import os
import cv2
import torch
import xlwt
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import get_dataset as ds
import unet
import torchvision
from torchvision import transforms
import metrics
import train as tr

preprocess_input = transforms.Compose([
                      transforms.Resize(256),
                      transforms.ToTensor(),
            ])

preprocess_target = transforms.Compose([
                      transforms.Resize(256, interpolation=Image.NEAREST),
                      transforms.ToTensor(),
            ])


def save_images(input_image, target_image, des_filename):
    target_tensor = preprocess_target(target_image).to(dev).unsqueeze(1).float()
    input_tensor = preprocess_input(input_image).float()
    input_tensor = input_tensor.to(dev).float()
    output = net(input_tensor.unsqueeze(1).float())

    metrics.ROC_curve(output.cpu().squeeze(1), target_tensor.cpu().squeeze(1))

    output_th = metrics.binary(output.cpu().squeeze(1))
    tf_tn_img = metrics.tf_fn_draw(output_th, target_tensor.squeeze(1))
    transform = torchvision.transforms.ToPILImage()
    image = transform(output_th)
    # tf = transform(tf_tn_img)
    tf_tn_img.save(des_filename + 'tf_tn_img' +  '.png')
    image.save(des_filename + 'unet_result' + '.png')


if __name__ == '__main__':
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(dev)
    dataset = ds.GetDataset(256)
    train_len = int(len(dataset) * 0.8)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])
    validation_len = int(len(test_set) * 0.5)
    validation_set, test_set = torch.utils.data.random_split(test_set, [validation_len, len(test_set) - validation_len])

    net = unet.UNet(1, 1)

    if dev == 'cuda:0':
        net = net.to(dev)
    else:
        net = net.float()

    tr.train(device=dev, n_epoch=2, batch_size=1, lr=0.001, trainset=train_set, net=net)
    tr.test(device=dev, batch_size=1, testset=test_set, net=net)

