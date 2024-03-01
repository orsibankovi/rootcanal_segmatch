import cv2
from natsort import natsorted
import os
import numpy as np
import xlwt


def getoriginalfiles(path):
    files_ = os.listdir(path)
    filenames = list(files_)
    filtered = filter(lambda x: x.endswith(".png"), filenames)
    png_filenames = list(filtered)
    sorted_filenames = natsorted(png_filenames)
    images = []
    for png_files in sorted_filenames:
        img_temp = cv2.imread(path + '/' + png_files, cv2.IMREAD_GRAYSCALE)
        images.append(img_temp)
    return images


def gammatransform(path, gamma):
    images = getoriginalfiles(path)
    img_gamma = []
    # gamma = 0.16  # set!!
    inv_gamma = 1.0 / gamma

    for img in images:
        table = [((j / 255) ** inv_gamma) * 255 for j in range(256)]
        table = np.array(table, np.uint8)
        gamma = cv2.LUT(img, table)
        img_gamma.append(gamma)

    return img_gamma


def neighbour_avg(path):
    k_size = (2, 2)
    images = gammatransform(path, 0.12)
    images_avg = []
    for img in images:
        blur = cv2.blur(img, k_size)
        images_avg.append(blur)

    return images_avg


def gradient(path):
    images = gammatransform(path, 0.1)
    images_grad = []
    for img in images:
        laplacian = cv2.Laplacian(img, cv2.CV_8UC1, 10)
        images_grad.append(laplacian)
    return images_grad

def getoriginalbinary(path):
    path_binary = path + '/axis/binaris'
    filenames = list(os.listdir(path_binary))
    png_filenames = list(filter(lambda x: x.endswith(".png"), filenames))
    sorted_filenames = natsorted(png_filenames)
    images = []
    for image_file in sorted_filenames:
        img_temp = cv2.imread(path_binary + '/' + image_file, cv2.IMREAD_GRAYSCALE)
        images.append(img_temp)
    return images

def bin_mixedmap(path):
    f = gammatransform(path, 0.2)
    g = gradient(path)
    n = neighbour_avg(path)
    mixedmap = []
    for img in range(len(f)):
        img_temp_ = f[img] + g[img] + n[img]
        ret, img_temp = cv2.threshold(img_temp_, 0, 255, cv2.THRESH_BINARY)
        mixedmap.append(img_temp)
    return mixedmap


def convexhull(img, contour, hierarchy):
    for c, h in zip(contour, hierarchy[0]):
        if h[2] != -1:
            convex_hull = cv2.convexHull(c)
            cv2.drawContours(img, [convex_hull], 0, (255, 255, 255), thickness=cv2.FILLED)

    return img


def number_of_holes(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if isinstance(hierarchy, np.ndarray):
        img = convexhull(img, contours, hierarchy)
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


def original_with_root_canal(path):
    path_original = path + '/axis/eredeti'
    filenames = list(os.listdir(path_original))
    png_filenames = list(filter(lambda x: x.endswith(".png"), filenames))
    sorted_filenames = natsorted(png_filenames)
    images = []
    for image_file in sorted_filenames:
        img_temp = cv2.imread(path_original + '/' + image_file, cv2.IMREAD_GRAYSCALE)
        images.append(img_temp)
    return images

def count_white_pixels(img):
    count = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 255:
                count += 1
    return count


def match_original(images_rootcanal, images):
    diffs = []
    for j in range(len(images)):
        diff = np.average((images_rootcanal - images[j]) ** 2)
        diffs.append(diff)
    m = np.array(diffs)
    return m


def write_min(path, start, l):
    p = path + "/axis/range.txt"
    with open(str(p), "w") as f:
        f.write(str(start) + " " + str(l) + "\n")


def calculate_start(e):
    b, n = e.shape
    ss = []
    for j in range(n - b + 1):
        ss.append(np.trace(e, j))
    a = np.array(ss)
    m = np.argmin(a)
    return m


def process(path, name, sheet, count):
    os.chdir(path)
    original_images = getoriginalfiles(path)
    # bins = bin_mixedmap(path)
    binaris = getoriginalbinary(path)
    img_root = original_with_root_canal(path)

    rows = []
    for j in range(len(img_root)):
        mi = match_original(img_root[j], original_images)
        rows.append(mi)

    E = np.array(rows)
    s = calculate_start(E)

    write_min(path, s, len(img_root))
    print(s - 1)
    print(len(img_root))

    target_path = 'c:/Users/orsolya.bankovi/Documents/uni/rootcanal_segmatch/all/'
    os.chdir(target_path)

    for j in range(len(original_images)):
        if s <= j < s + len(img_root):
            cv2.imwrite(target_path + 'original/' + name + "_" + str(j + 1) + "_" + "original.png", original_images[j])
            # cv2.imwrite(target_path + 'binary/' + name + "_" + str(j + 1) + "_" + "binary.png", number_of_holes(bins[j]))
            # cv2.imwrite(target_path + 'binary/' + name + "_" + str(j + 1) + "_" + "binary.png", number_of_holes(binaris[j-s]))
            inverse = cv2.bitwise_not(floodfill(binaris[j-s]))
            cv2.imwrite(target_path + 'inverse/' + name + "_" + str(j + 1) + "_" + "rootcanal.png", inverse)
            sheet.write(count, 0, '/all/' + name + "/" + str(j + 1) + "_" + ".png")
            sheet.write(count, 1, '/all/inverse/' + name + "_" + str(j + 1) + "_" + "rootcanal.png")
            sheet.write(count, 2, count_white_pixels(inverse))
            count += 1

        elif j % 5 == 0:
            cv2.imwrite(target_path + 'original/' + name + "_" + str(j + 1) + "_" + "original.png", original_images[j])
            inverse = np.zeros(original_images[j].shape)
            #cv2.imwrite(target_path + 'binary/' + name + "_" + str(j + 1) + "_" + "binary.png", black)
            cv2.imwrite(target_path + 'inverse/' + name + "_" + str(j + 1) + "_" + "rootcanal.png", inverse)
            sheet.write(count, 0, '/all/' + name + "/" + str(j + 1) + "_" + ".png")
            sheet.write(count, 1, '/all/inverse/' + name + "_" + str(j + 1) + "_" + "rootcanal.png")
            sheet.write(count, 2, count_white_pixels(inverse))
            count += 1     
    return count
        
            
    # cv2.imshow('MixedMap', NumberOfHoles(bins[146]))
    # cv2.imshow('Original', original_images[146])
    # cv2.imshow('FloodFill', cv2.bitwise_not(floodfill(bins[146])))
    # cv2.waitKey(0)

if __name__ == '__main__':
    rootdir = 'c:/Users/orsolya.bankovi/Documents/uni/rootcanal_segmatch/fogak'
    ws = xlwt.Workbook()
    sheet = ws.add_sheet('Images')
    sheet.write(0, 0, 'Original')
    sheet.write(0, 1, 'Rootcanal')
    sheet.write(0, 2, 'Number of white pixels')

    dirs = [[]]
    count = 1
    for subdir, dir_, files in os.walk(rootdir):
        dirs.append(dir_)
    for i in dirs[1]:
        if i != 'segmentation_rootcanal_only' and i != 'segmentation' and i != 'segmentation_all':
            Path = rootdir + '/' + i + '/'
            count = process(Path, i, sheet, count)
            
    ws.save('c:/Users/orsolya.bankovi/Documents/uni/rootcanal_segmatch/fogak/segmentation/' + 'summary' + '.xls')
