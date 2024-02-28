import os
import cv2
import torch
import xlwt
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchmetrics
import math


class Train():
    def __init__(self, dev, n_epoch, batch_size, lr, net, k):
        super(Train, self).__init__()
        self.dev = dev
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.Dice = torchmetrics.Dice(zero_division=1.0, threshold=0.5).to(self.dev)
        self.Jaccard = torchmetrics.JaccardIndex(task='binary', threshold=0.5).to(self.dev) 
        self.criterion = nn.BCELoss().to(self.dev)
        self.optimizer = optim.Adam(net.parameters(), self.lr)
        self.k = k
        
    def plot_loss(self, train_loss, valid_loss, n_epochs, name, num):
        plt.figure(num)
        plt.plot([*range(0, n_epochs, 1)], train_loss)
        plt.plot([*range(0, n_epochs, 1)], valid_loss)
        plt.xlim((0, n_epochs))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(name+'.jpg')  
        
    def train(self, trainset, validationset, net):
        log_interval = 500
        train_loader = torch.utils.data.DataLoader(trainset, self.batch_size, shuffle=True)
        net.train(True)
        count = 0
        wb = xlwt.Workbook()
        ws = wb.add_sheet('Train')
        ws.write(0, 0, 'Epoch')
        ws.write(0, 1, 'Loss')
        ws.write(0, 2, 'DiceLoss')
        ws.write(0, 3, 'JaccardIndex')
        ws1 = wb.add_sheet('validation')
        ws1.write(0, 0, 'Epoch')
        ws1.write(0, 1, 'Loss')
        ws1.write(0, 2, 'DiceLoss')
        ws1.write(0, 3, 'JaccardIndex')
        
        best_jaccard = 0.0

        # train
        print('train')
        for epoch in range(1, self.n_epoch + 1):
            
            train_losses = []
            dice_losses = []
            jaccard_index = []
        
            for batch_idx, (data, target) in enumerate(train_loader):

                if self.dev.type == 'cuda':
                    data = data.to(self.dev)
                    target = target.to(self.dev)
                #data = data.float()
            
                self.optimizer.zero_grad()  # clear the gradient

                output = net(data)  # forward propagation
                loss = self.criterion(output, target)  # calculate loss
                
                loss.backward()  # current loss
                self.optimizer.step()  # update parameters

                dice_loss_value = self.Dice(output, target.int())
                jaccard_value = self.Jaccard(output, target.int())
                train_losses.append(loss.item())
                dice_losses.append(dice_loss_value.item())

                if math.isnan(jaccard_value.item()):
                    jaccard_index.append(1.0)
                else:
                    jaccard_index.append(jaccard_value.item())

                if batch_idx % log_interval == 0:
                    if batch_idx != 0:
                        print('Train Epoch: ' + str(epoch)
                            + " batch_idx: "
                            + str(batch_idx)
                            + "\tLoss: "
                            + str(round(np.average(train_losses[-log_interval]), 8))
                            + "\tDiceLoss: "
                            + str(round(np.average(dice_losses[-log_interval]), 8))
                            + "\tJaccardIndex: "
                            + str(round(np.average(jaccard_index[-log_interval]), 8)))

                count += 1
                
            ws.write(epoch+1, 0, 'epoch=' + str(epoch))
            ws.write(epoch+1, 1, np.average(train_losses))
            ws.write(epoch+1, 2, np.average(dice_losses))
            ws.write(epoch+1, 3, np.average(jaccard_index))
                        
            val_loss, val_dice, val_jaccard = self.validation(epoch, net, validationset)
            
            ws1.write(epoch+1, 0, 'epoch=' + str(epoch))
            ws1.write(epoch+1, 1, np.average(val_loss))
            ws1.write(epoch+1, 2, np.average(val_dice))
            ws1.write(epoch+1, 3, np.average(val_jaccard))
            
            if best_jaccard < np.average(val_jaccard):
                best_jaccard = np.average(val_jaccard)
                print('AZ EDDIGI LEGJOBB JACCARD A ' + str(epoch) + '. EPOCHBAN: ' + str(best_jaccard))
                torch.save(net, './results/' + self.k +  '/trained_net.pt')
            print('Act jaccard: ' + str(np.average(val_jaccard)))
            
        wb.save('./results/' + self.k + '/train_result.xls')
        
        return net

    def validation(self, epoch, net, validationset):
        #validation
        validation_loader = torch.utils.data.DataLoader(validationset, self.batch_size, shuffle=True)
        valid_train_losses = []
        valid_dice_losses = []
        valid_jaccard_index = []

        net.train(False)
        print('Validation')
        count = 0
        for batch_idx, (data, target) in enumerate(validation_loader):
            if self.dev.type == 'cuda':
                data = data.to(self.dev)
                target = target.to(self.dev)
            data = data.float()

            output = net(data)
            loss = self.criterion(output, target)
            dice_loss_value = self.Dice(output, target.int())
            jaccard_value = self.Jaccard(output, target.int())
            valid_train_losses.append(loss.item())
            valid_dice_losses.append(dice_loss_value.item())
            if math.isnan(jaccard_value.item()):
                valid_jaccard_index.append(1.0)
            else:
                valid_jaccard_index.append(jaccard_value.item())

            if (count % 100 == 0):
                if count == 0:
                    print('Train Epoch: ' + str(epoch)
                          + " batch_idx: " + str(batch_idx)
                          + ' Valid Loss: ' + str(round(valid_train_losses[0], 8))
                          + ' Valid DiceLoss: ' + str(round(valid_dice_losses[0], 8))
                          + ' Valid JaccardIndex: ' + str(round(valid_jaccard_index[0], 8)))
                else:
                    print('Train Epoch: ' + str(epoch)
                          + " batch_idx: " + str(batch_idx)
                          + ' Valid Loss: ' + str(round(np.average(valid_train_losses[-10]), 8))
                          + ' Valid DiceLoss: ' + str(round(np.average(valid_dice_losses[-10]), 8))
                          + ' Valid JaccardIndex: ' + str(round(np.average(valid_jaccard_index[-10]), 8)))

            count += 1
    
        return valid_train_losses, valid_dice_losses, valid_jaccard_index
            
    def run(self, trainset, validationset, net):
        net = self.train(trainset, validationset, net)
        return net