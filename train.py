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
import metrics

def train(dev, n_epoch, batch_size, lr, trainset, validationset, net):
    n_epochs = n_epoch
    log_interval = 100
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr)
    train_losses = []
    dice_losses = []
    jaccard_index = []
    valid_train_losses = []
    valid_dice_losses = []
    valid_jaccard_index = []

    # train
    for epoch in range(1, n_epochs + 1):
        net.train(True)

        for batch_idx, (data, target) in enumerate(train_loader):
            if dev != "cpu":
                data = data.to(dev)
                target = target.to(dev)

            data = data.float()

            optimizer.zero_grad()  # clear the gradient
            output = net(data)  # forward propagation
            loss = criterion(output, target)  # calculate loss
            dice_loss_value, jaccard_value = metrics.binary_classification(output, target)
            loss.backward()  # current loss
            optimizer.step()  # update parameters

            train_losses.append(loss.item())
            dice_losses.append(dice_loss_value)
            jaccard_index.append(jaccard_value)

            if batch_idx % log_interval == 0:
                if batch_idx == 0:
                    print('Train Epoch: ' + str(epoch)
                          + " batch_idx: "
                          + str(batch_idx)
                          + "\tLoss: "
                          + str(round(loss.item(), 8))
                          + "\tDiceLoss: "
                          + str(round(dice_loss_value, 8))
                          + "\tJaccardIndex: "
                          + str(round(jaccard_value, 8)))

                else:
                    print('Train Epoch: ' + str(epoch)
                          + " batch_idx: "
                          + str(batch_idx)
                          + "\tLoss: "
                          + str(round(np.average(train_losses[-log_interval]), 8))
                          + "\tDiceLoss: "
                          + str(round(np.average(dice_losses[-log_interval]), 8))
                          + "\tJaccardIndex: "
                          + str(round(np.average(jaccard_index[-log_interval]), 8)))

        net.train(False)
        print('Validation')
        count = 0
        for batch_idx, (data, target) in enumerate(validation_loader):
            if dev != "cpu":
                data = data.to(dev)
                target = target.to(dev)
            data = data.float()

            output = net(data)
            loss = criterion(output, target)
            dice_loss_value, jaccard_value = metrics.binary_classification(output, target)

            valid_train_losses.append(loss.item())
            valid_dice_losses.append(dice_loss_value)
            valid_jaccard_index.append(jaccard_value)

            if (count % 10 == 0):
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
    criterion = nn.BCELoss()

    print('test' + '\n')
    test_losses = []
    test_dice_losses = []
    test_jaccard = []
    net.train(False)
    count = 0

    with torch.no_grad():
        for data, target in test_loader:
            if dev != "cpu":
                data = data.to(dev)
                target = target.to(dev)

            data = data.float()

            output = net(data)
            # pred = torch.argmax(output.data, dim = 1).unsqueeze(1).float()
            loss = criterion(output, target)
            dice_loss_value, jaccard_value = metrics.binary_classification(output, target)

            test_losses.append(loss.item())
            test_dice_losses.append(dice_loss_value)
            test_jaccard.append(jaccard_value)

            if (count % 10 == 0):
                if count == 0:
                    print('Valid Loss: ' + str(round(test_losses[0], 8)))
                    print('Valid DiceLoss: ' + str(round(test_dice_losses[0], 8)))
                    print('Valid JaccardIndex: ' + str(round(test_jaccard[0], 8)))

                else:
                    print('Valid Loss: ' + str(round(np.average(test_losses[-5]), 8)))
                    print('Valid DiceLoss: ' + str(round(np.average(test_dice_losses[-5]), 8)))
                    print('Valid JaccardIndex: ' + str(round(np.average(test_jaccard[-5]), 8)))

            ws.write(count + 1, 0, 'count =' + str(count))
            ws.write(count + 1, 1, test_losses[-1])
            ws.write(count + 1, 2, test_dice_losses[-1])
            ws.write(count + 1, 3, test_jaccard[-1])

            count += 1

    ws.write(count + 5, 1, np.average(test_losses))
    ws.write(count + 5, 2, np.average(test_dice_losses))
    ws.write(count + 5, 3, np.average(test_jaccard))

