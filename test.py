import os
import math
import torch
import xlwt
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchmetrics
import metrics

class Test():
    def __init__(self, dev, batch_size, testset, net):
        super(Test, self).__init__()
        self.dev = dev
        self.batch_size = batch_size
        self.testset = testset
        self.net = net
        self.Dice = torchmetrics.Dice(zero_division=1.0, threshold=0.5).to(self.dev)
        self.Jaccard = torchmetrics.JaccardIndex(task='binary', threshold=0.5).to(self.dev) 
        self.criterion = nn.BCELoss().to(self.dev)
        
    def test(self):
        # To open Workbook
        wb = xlwt.Workbook()
        ws = wb.add_sheet('sheet')
        ws.write(0, 0, 'Valid loss')
        ws.write(0, 1, 'Value')

        test_loader = torch.utils.data.DataLoader(self.testset, self.batch_size, shuffle=False)
        criterion = nn.BCELoss()

        print('test' + '\n')
        test_losses = []
        test_dice_losses = []
        test_jaccard = []
        self.net.train(False)
        count = 0
        fpr = []
        tpr = []

        with torch.no_grad():
            for data, target in test_loader:
                if self.dev.type == 'cuda':
                    data = data.to(self.dev)
                    target = target.to(self.dev)
                data = data.float()

                output = self.net(data)
                loss = criterion(output, target)
                
                dice_loss_value = self.Dice(output, target.int())
                jaccard_value = self.Jaccard(output, target.int())
                test_losses.append(loss.item())
                test_dice_losses.append(dice_loss_value.item())
                if math.isnan(jaccard_value.item()):
                    test_jaccard.append(1.0)
                else:
                    test_jaccard.append(jaccard_value.item())
                    
                roc = torchmetrics.ROC(task='binary', thresholds=100)
                fpr_, tpr_, thresholds = roc(output, target.int())
                    
                if np.average(np.asarray(output.cpu().squeeze(1)))>0 and np.average(np.asarray(target.cpu().squeeze(1)))>0:
                    fpr.append(np.asarray(fpr_.cpu()))
                    tpr.append(np.asarray(tpr_.cpu()))

                if (count%10 == 0):
                    if count == 0:
                        print('Test Loss: ' + str(round(test_losses[0], 8)))
                        print('Test DiceLoss: ' + str(round(test_dice_losses[0], 8)))
                        print('Test JaccardIndex: ' + str(round(test_jaccard[0], 8)))

                    else:
                        print('Test Loss: ' + str(round(np.average(test_losses[-5]), 8)))
                        print('Test DiceLoss: ' + str(round(np.average(test_dice_losses[-5]), 8)))
                        print('Test JaccardIndex: ' + str(round(np.average(test_jaccard[-5]), 8)))

                ws.write(count+1, 0, 'count =' + str(count))
                ws.write(count+1, 1, test_losses[-1])
                ws.write(count+1, 2, test_dice_losses[-1])
                ws.write(count+1, 3, test_jaccard[-1])

                count += 1
                
        print(np.array(fpr).shape)
        print(np.array(tpr).shape)
                
        ws.write(count+5, 1, np.average(test_losses))
        ws.write(count+5, 2, np.average(test_dice_losses))
        ws.write(count+5, 3, np.average(test_jaccard))
        
        fpr_avg = np.average(np.asarray(fpr), axis=0)
        tpr_avg = np.average(np.asarray(tpr), axis=0)
        
        wb.save('test_losses.xls')
        
        plt.figure(7)
        auc=np.trapz(tpr_avg, x=fpr_avg, dx=0.01)
        plt.plot(fpr_avg, tpr_avg, label='AUC='+str(round(auc, 4)))
        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), label='baseline', linestyle='--')

        plt.title('ROC Curve', fontsize=18)
        plt.ylabel('TPR', fontsize=16)
        plt.xlabel('FPR', fontsize=16)
        plt.legend(fontsize=12, loc='lower right')
        plt.savefig('ROC_AUC.jpg')
                
    def run(self):
        self.test()