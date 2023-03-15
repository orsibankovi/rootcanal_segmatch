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
        laplacian = cv2.Laplacian(img, cv2.CV_8UC1, 10)
        imgsGrad.append(laplacian)
    return imgsGrad


def binMixedMap(path):
    f = GammaTransform(path, 0.2)
    g = gradient(path)
    n = neighbourAvg(path)
    mixedmap = []
    for img in range(len(f)):
        img_temp_ = f[img] + g[img] + n[img]
        ret, img_temp = cv2.threshold(img_temp_, 0, 255, cv2.THRESH_BINARY)
        mixedmap.append(img_temp)
    return mixedmap


def MinMaxZ(img):
    avg = np.average(np.array(img).flatten())
    return avg


def ConvexHull(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Konvex burkok rajzolása a képre
    convex_hulls = []
    for contour in contours:
        convex_hull = cv2.convexHull(contour)
        convex_hulls.append(convex_hull)
        cv2.drawContours(img, [convex_hull], 0, (255, 255, 255), 8)

    return img

def floodfill(img):
    for x in range(img.shape[1]):
        # Fill dark top pixels:
        if img[0, x] == 0:
            cv2.floodFill(img, None, seedPoint=(x, 0), newVal=255, loDiff=3,
                          upDiff=3)  # Fill the background with white color

        # Fill dark bottom pixels:
        if img[-1, x] == 0:
            cv2.floodFill(img, None, seedPoint=(x, img.shape[0] - 1), newVal=255, loDiff=3,
                          upDiff=3)  # Fill the background with white color

    for y in range(img.shape[0]):
        # Fill dark left side pixels:
        if img[y, 0] == 0:
            cv2.floodFill(img, None, seedPoint=(0, y), newVal=255, loDiff=3,
                          upDiff=3)  # Fill the background with white color

        # Fill dark right side pixels:
        if img[y, -1] == 0:
            cv2.floodFill(img, None, seedPoint=(img.shape[1] - 1, y), newVal=255, loDiff=3,
                          upDiff=3)  # Fill the background with white color
    return img

def Process(path):
    #os.chdir(path)
    imgO = getOriginalFiles(path)
    bins = binMixedMap(path)

    target_path = 'fogak/segmentation/'

    '''
    closeBins = []
    for img in bins:
        closeBins.append(ConvexHull(img))
    '''

    for i in range(len(bins)):
        if (MinMaxZ(bins[i]) > 0):
            cv2.imwrite(target_path + 'binary/' + "CBCT 201307231443" + "_" + str(i) + "_" + "binary.png", bins[i])
            cv2.imwrite(target_path + 'original/' + "CBCT 201307231443" + "_" + str(i) + "_" + "original.png", imgO[i])
            cv2.imwrite(target_path + 'inverse/' + "CBCT 201307231443" + "_" + str(i) + "_" + "inverse.png", cv2.bitwise_not(floodfill(bins[i])))

    cv2.imshow('MixedMap', bins[182])
    cv2.imshow('FloodFill', cv2.bitwise_not(floodfill(bins[182])))
    cv2.imshow('Original', imgO[182])
    floodfill(bins[85])
    cv2.waitKey(0)


Process(path)
