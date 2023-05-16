import os
import cv2
import torch
from PIL import Image
import xlwt
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import get_dataset as ds
import unet


def binary_classification(output, target, threshold=0.5):
    output = output > threshold
    target = target > 0  # assuming ground truth has non-zero values where foreground is present

    if (torch.mean(output.float()) == 0 and torch.mean(target.float()) == 0) or (
            torch.mean(output.float()) == 1 and torch.mean(target.float()) == 1):
        return 1, 1

    smooth = 1e-6
    iflat = output.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    union = torch.sum(iflat) + torch.sum(tflat)
    dice = ((2. * intersection) / (A_sum + B_sum + smooth))
    jaccard = intersection / (union - intersection + smooth)

    return dice.item(), jaccard.item()

def ROC_values(output, target, thresh):
  binary = output
  binary = binary > thresh
  target = target > 0

  tp = ((binary == 1) & (target == 1)).sum()
  tn = ((binary == 0) & (target == 0)).sum()
  fp = ((binary == 1) & (target == 0)).sum()
  fn = ((binary == 0) & (target == 1)).sum()

  return tn, fp, fn, tp

def ROC_curve(output, target):
    roc_values = []
    for thresh in np.linspace(0, 1, 10):
        tn, fp, fn, tp = ROC_values(output, target, thresh)
        tpr = tp / (tp + fn + 1e-15)
        fpr = fp / (fp + tn + 1e-15)
        roc_values.append([fpr, tpr])

    roc_values = np.array(roc_values)

    plt.figure(7)
    plt.plot(roc_values[:, 0], roc_values[:, 1])
    plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), label='baseline', linestyle='--')

    plt.title('Receiver Operating Characteristic Curve', fontsize=18)
    plt.ylabel('TPR', fontsize=16)
    plt.xlabel('FPR', fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig('ROC_AUC.jpg')


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

