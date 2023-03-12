import cv2
from natsort import natsorted
# import sys
# mport pathlib
import os
import numpy as np
# import pandas as pd
from tqdm import tqdm

path = 'fogak/CBCT 201307231443'

dfrows = []


def getOriginalFiles(path):
    filenames = list(os.listdir(path))
    png_filenames = list(filter(lambda x: x.endswith(".png"), filenames))
    sorted_filenames = natsorted(png_filenames)
    imgs = []
    for imgfile in sorted_filenames:
        img_temp = cv2.imread(path + '/' + imgfile, cv2.IMREAD_GRAYSCALE)
        imgs.append(img_temp)
    return imgs


def GammaTransform(path, gamma):
    imgs = getOriginalFiles(path)
    imgGamma = []
    # gamma = 0.16  # set!!
    invGamma = 1.0 / gamma

    for img in imgs:
        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        gamma = cv2.LUT(img, table)
        imgGamma.append(gamma)

    return imgGamma


def neighbourAvg(path):
    ksize = (4, 4)
    imgs = GammaTransform(path, 0.12)
    imgsAvg = []
    for img in imgs:
        blur = cv2.blur(img, ksize)
        imgsAvg.append(blur)

    return imgsAvg


def gradient(path):
    imgs = GammaTransform(path, 0.1)
    imgsGrad = []
    for img in imgs:
        laplacian = cv2.Laplacian(img, cv2.CV_64F, 10)
        imgsGrad.append(laplacian)
    return imgsGrad


def binMixedMap(path):
    f = GammaTransform(path, 0.2)
    g = gradient(path)
    n = neighbourAvg(path)
    mixedmap = []
    for img in range(len(f)):
        ret, img_temp = cv2.threshold(f[img] + g[img] + n[img], 0, 255, cv2.THRESH_BINARY)
        mixedmap.append(img_temp)

    return mixedmap


def MinMaxZ(img):
    avg = np.average(np.array(img).flatten())
    return avg


def Process(path):
    #os.chdir(path)
    imgO = getOriginalFiles(path)
    bins = binMixedMap(path)

    target_path = 'fogak/segmentation/'

    for i in range(len(bins)):
        if (MinMaxZ(bins[i]) > 0):
            cv2.imwrite(target_path + "CBCT 201307231443" + "_" + str(i) + "_" + "binary.png", bins[i])
            cv2.imwrite(target_path + "CBCT 201307231443" + "_" + str(i) + "_" + "original.png", imgO[i])

    #cv2.imshow('MixedMap', bins[140])
    #cv2.imshow('Original', imgO[140])
    #cv2.waitKey(0)


Process(path)
