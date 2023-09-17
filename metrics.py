import os
import cv2
import torch
from PIL import Image
import xlwt
import numpy as np
from sklearn import metrics
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import get_dataset as ds


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

