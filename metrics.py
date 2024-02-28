import os
import cv2
import torch
from PIL import Image
import math
import numpy as np
from sklearn import metrics
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import get_dataset as ds

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):    
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = nn.BCELoss(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


def binary(output, threshold=0.5):
    binary = output > threshold

    binary = binary.float()

    return binary


def tf_fn_draw(output, target):
    target = target > 0  # assuming ground truth has non-zero values where foreground is present

    ima = np.zeros((output.shape[1], output.shape[2], 3))

    for i in range(target.shape[1]):
        for j in range(target.shape[2]):
            if target[0, i, j] == 0 and output[0, i, j] == 1:
                ima[i, j, 0] = 255
            elif target[0, i, j] == 1 and output[0, i, j] == 0:
                ima[i, j, 1] = 255
            elif target[0, i, j] == 1 and output[0, i, j] == 1:
                ima[i, j, :] = 255
            else:
                ima[i, j, :] = 0

    PIL_image = Image.fromarray(ima.astype('uint8'), 'RGB')

    return PIL_image


def find_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    else:
        return None, None

def center_of_canal(tensor):
    tensor = torch.squeeze(tensor, 0).numpy().transpose(1, 2, 0).astype(np.uint8)*255
    contours, _ = cv2.findContours(tensor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = [find_center(contour) for contour in contours]
    X = [center[0] for center in centers if center[0] is not None]
    Y = [center[1] for center in centers if center[1] is not None]

    return X, Y