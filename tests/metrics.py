import cv2
import torch
from PIL import Image
import math
import numpy as np
from torch import Tensor

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

def center_of_canal(tensor: Tensor) -> tuple[list, list]:
    tensor = torch.squeeze(tensor, 0).numpy().transpose(1, 2, 0).astype(np.uint8)*255
    contours, _ = cv2.findContours(tensor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = [find_center(contour) for contour in contours]
    X = [center[0] for center in centers if center[0] is not None]
    Y = [center[1] for center in centers if center[1] is not None]

    return X, Y

def centers_of_canals(output_tensor: Tensor, target_tensor: Tensor) -> float:
    output_x, output_y = center_of_canal(output_tensor)
    target_x, target_y = center_of_canal(target_tensor)

    if len(output_x) != 0 and len(target_x) != 0:
        distances = find_min_dist(output_x, target_x, output_y, target_y)
    else:
        distances = 'nan'

    return distances

def find_min_dist(output_x: list, target_x: list, output_y: list, target_y: list) -> float:
    distances = []
    shorter_x = min(output_x, target_x, key=len)
    longer_x = max(output_x, target_x, key=len)
    shorter_y = min(output_y, target_y, key=len)
    longer_y = max(output_y, target_y, key=len)

    for i in range(len(shorter_x)):
        min_difference = -1
        for j in range(len(longer_x)):
            dist = math.sqrt((shorter_x[i] - longer_x[j])**2 + (shorter_y[i] - longer_y[j])**2)
            if min_difference == -1 or dist < min_difference:
                min_difference = dist
        distances.append(min_difference)

    return np.sum(distances) / len(distances)