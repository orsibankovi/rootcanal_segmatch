import os
import cv2
import torch
from torch.utils.data import Dataset
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
    dice_loss_ = 1.0 - ((2.0 * intersection + smooth) / (a_sum + b_sum + smooth))
    return dice_loss_


def unet_loss(output, target, weight=1.0):
    sigmoid = nn.Sigmoid()
    bce_loss = nn.BCELoss()  # Binary Cross Entropy (BCE)

    # weighted sum of BCE and Dice Loss
    # loss = weight * bce_loss(sigmoid(output), target) + (1 - weight) * dice_loss_value
    loss = bce_loss(output, target)

    return loss


if __name__ == '__main__':
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ds.GetDataset()
    train_len = int(len(dataset) * 0.8)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])

    n_epochs = 1
    batch_size_train = 1
    log_interval = 100
    learning_rate = 0.001
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)

    net = unet.UNet(1, 1)
    net = net.to(dev)
    net = net.float()

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    train_losses = []
    dice_losses = []

    #train
    net.train()  # set mode of the NN

    for epoch in range(1, n_epochs + 1):

        for batch_idx, (data, target) in enumerate(train_loader):
            if dev == "cuda:0":
                data = data.to(dev).float()
                target = target.to(dev)
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

    plt.plot(dice_losses)

    #test
    print('test' + '\n')
    valid_losses = []
    valid_dice_losses = []
    net.train(False)
    count = 0

    with torch.no_grad():
        for data, target in test_loader:
            if dev == "cuda:0":
                data = data.to(dev).float()
                target = target.to(dev)
            else:
                data = data.float()

            output = net(data)
            dice_loss_value = dice_loss(output, target)
            valid_dice_losses.append(dice_loss_value.item())

            if count % 20 == 0:
                print('Valid Loss: ' + str(valid_dice_losses[-1]))

            torch.cuda.empty_cache()