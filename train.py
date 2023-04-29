import os
import cv2
import torch
import xlwt
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import get_dataset as ds
import unet


def dice_loss(output, target):
    smooth = 1.0
    # flatten tensors
    iflat = output.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    #intersection and union
    intersection = (iflat * tflat).sum()
    a_sum = torch.sum(iflat * iflat)
    b_sum = torch.sum(tflat * tflat)
    diceloss = 1.0 - ((2.0 * intersection + smooth) / (a_sum + b_sum + smooth))
    return diceloss

def unet_loss(output, target, weight=1.0):
    bce_loss = nn.BCELoss() #Binary Cross Entropy (BCE)
    bce_logit_loss = nn.BCEWithLogitsLoss()
    dice_loss_value = dice_loss(output, target)
    #weighted sum of BCE and Dice Loss
    loss = weight * bce_logit_loss(output, target) + (1 - weight) * dice_loss_value
    #loss = bce_loss(sigmoid(output), target)
    
    return loss


def train(device, n_epoch, batch_size, lr, trainset, net):
    n_epochs = n_epoch
    log_interval = 100
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    train_losses = []
    dice_losses = []

    # train
    net.train()  # set mode of the NN
    for epoch in range(1, n_epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            if device == 'cuda:0':
                data = data.to(device).float()
                target = target.to(device)
            else:
                data = data.float()

            optimizer.zero_grad()  # clear the gradient
            output = net(data)  # forward propagation
            loss = unet_loss(output, target)  # calculation loss
            dice_loss_value = dice_loss(output, target)
            loss.backward()  # current loss
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: ' + str(epoch)
                      + " batch_idx: "
                      + str(batch_idx)
                      + "\tLoss: "
                      + str(dice_loss_value.item()))

                train_losses.append(loss.item())
                dice_losses.append(dice_loss_value.item())

            torch.cuda.empty_cache()

    plt.figure(1)
    plt.plot(dice_losses)
    plt.savefig('dice_losses.jpg')
    plt.figure(2)
    plt.plot(train_losses)
    plt.savefig('train_losses.jpg')


def test(device, batch_size, testset, net):
    loc = ("C:/Users/banko/Desktop/BME_VIK/I_felev/onlab1")
    # To open Workbook
    wb = xlwt.Workbook()
    ws = wb.add_sheet('sheet')
    ws.write(0, 0, 'Valid loss')
    ws.write(0, 1, 'Value')

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    print('test' + '\n')
    valid_losses = []
    valid_dice_losses = []
    net.train(False)
    count = 0

    with torch.no_grad():
        for data, target in test_loader:
            if device == 'cuda:0':
                data = data.to(device).float()
                target = target.to(device)
            else:
                data = data.float()

            output = net(data)
            loss = unet_loss(output, target)
            dice_loss_value = dice_loss(output, target)
            valid_losses.append(loss.item())
            valid_dice_losses.append(dice_loss_value.item())

            if count % 5 == 0:
                if count == 0:
                    print('Valid Loss: ' + str(valid_dice_losses[0]))
                else:
                    print('Valid Loss: ' + str(np.average(valid_dice_losses[-5])))

            ws.write(count + 1, 0, 'count =' + str(count))
            ws.write(count + 1, 1, str(valid_dice_losses[-1]))

            count += 1

            torch.cuda.empty_cache()

    plt.figure(3)
    plt.plot(valid_dice_losses)
    plt.savefig('valid_losses.jpg')
    plt.figure(4)
    plt.plot(valid_losses)
    plt.savefig('valid_dice_losses.jpg')
    wb.save('valid_dice_losses.xls')


if __name__ == '__main__':
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(dev)
    dataset = ds.GetDataset(128)
    train_len = int(len(dataset) * 0.8)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])

    net = unet.UNet(1, 1)

    if dev == 'cuda:0':
        net = net.to(dev)
    else:
        net = net.float()

    train(device=dev, n_epoch=2, batch_size=8, lr=0.001, trainset=train_set, net=net)
    test(device=dev, batch_size=8, testset=test_set, net=net)

