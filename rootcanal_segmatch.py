import cv2
from natsort import natsorted
# import sys
# mport pathlib
import os
import numpy as np
# import pandas as pd
from tqdm import tqdm

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


def ConvexHull(img, contour):
    # Konvex burkok rajzolása a képre
    if not cv2.isContourConvex(contour):
        convex_hull = cv2.convexHull(contour)
        cv2.drawContours(img, [convex_hull], 0, (255, 255, 255), 0)

    return img

def NumberOfHoles(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for c, h in zip(contours, hierarchy[0]):
        # If there is at least one interior contour, find out how many there are
        if h[2] == -1:
            img = ConvexHull(img, c)
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

def OriginalWithRootCanal(path):
    pathOriginal = path + '/axis/eredeti'
    filenames = list(os.listdir(pathOriginal))
    png_filenames = list(filter(lambda x: x.endswith(".png"), filenames))
    sorted_filenames = natsorted(png_filenames)
    imgs = []
    for imgfile in sorted_filenames:
        img_temp = cv2.imread(pathOriginal + '/' + imgfile, cv2.IMREAD_GRAYSCALE)
        imgs.append(img_temp)
    return imgs


def matchOriginal(imgsRootCanal, imgs):
    diffs = []
    for i in range(len(imgs)):
        diff = np.average((imgsRootCanal-imgs[i])**2)
        diffs.append(diff)
    m = np.array(diffs)
    return m

def writeMins(path, start, l):
    p = path + "/axis/range.txt"
    with open(str(p), "w") as f:
        f.write(str(start) + " " + str(l) + "\n")

def calcStart(E):
    b = E.shape[0]
    n = E.shape[1]
    ss = []
    for i in range(n-b+1):
        ss.append(np.trace(E, i))
    a = np.array(ss)
    m = np.argmin(a)
    return m


def Process(path):
    os.chdir(path)
    imgO = getOriginalFiles(path)
    bins = binMixedMap(path)
    imgRoot = OriginalWithRootCanal(path)

    rows = []
    for i in range(len(imgRoot)):
        mi = matchOriginal(imgRoot[i], imgO)
        rows.append(mi)

    E = np.array(rows)
    s = calcStart(E)

    writeMins(path, s, len(imgRoot))
    print(s-1)
    print(len(imgRoot))

    target_path = 'C:/Users/banko/Desktop/BME_VIK/I_felev/onlab1/fogak/segmentation/'
    os.chdir(target_path)

    for i in range(s-1, s+len(imgRoot)):
        print(i)
        cv2.imwrite(target_path + 'binary/' + "CBCT 201307231443" + "_" + str(i) + "_" + "binary.png", NumberOfHoles(bins[i]))
        cv2.imwrite(target_path + 'original/' + "CBCT 201307231443" + "_" + str(i) + "_" + "original.png", imgO[i])
        cv2.imwrite(target_path + 'inverse/' + "CBCT 201307231443" + "_" + str(i) + "_" + "rootcanal.png", cv2.bitwise_not(floodfill(bins[i])))

    cv2.imshow('MixedMap', NumberOfHoles(bins[80]))
    cv2.imshow('Original', imgO[80])
    cv2.imshow('FloodFill', cv2.bitwise_not(floodfill(bins[80])))
    cv2.waitKey(0)


if __name__ == '__main__':
    path = 'C:/Users/banko/Desktop/BME_VIK/I_felev/onlab1/fogak/CBCT 201307231443/'
    Process(path)
