import cv2
import numpy as np
from natsort import natsorted
import os

def process_image(input_path, output_path):
    # Olvassuk be a képet
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Állítsuk be a küszöbértéket, hogy bináris képet kapjunk
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Másoljuk a bináris képet, hogy ne módosítsuk az eredetit
    result_image = binary_image.copy()

    # Definiáljuk a 8 szomszédot
    neighbors = [(i, j) for i in range(-2, 3) for j in range(-2, 3) if (i != 0 or j != 0)]

    # Végigmegyünk a képen
    for i in range(1, binary_image.shape[0] - 1):
        for j in range(1, binary_image.shape[1] - 1):
            white_neighbors = 0
            if binary_image[i, j] == 0:
                continue
            if binary_image[i, j] != 0 and binary_image[i, j] != 255:
                print("Nem bináris kép!")
                binary_image[i, j] = 0
            for ni, nj in neighbors:
                if binary_image[i + ni, j + nj] == 255:
                    white_neighbors += 1
            if binary_image[i, j] == 255 and white_neighbors <= 1:
                result_image[i, j] = 0

    # Mentsük el az eredményt
    cv2.imwrite(output_path, result_image)

if __name__ == "__main__":
        path = 'c:/Users/orsolya.bankovi/Documents/uni/rootcanal_segmatch/all/processed'
        files_ = os.listdir(path)
        filenames = list(files_)
        filtered = filter(lambda x: x.endswith(".png"), filenames)
        png_filenames = list(filtered)
        sorted_filenames = natsorted(png_filenames)
        images = []
        for png_files in sorted_filenames:
            path_png = path + '/' + png_files
            target_path = 'c:/Users/orsolya.bankovi/Documents/uni/rootcanal_segmatch/all/fuck/' + png_files
            process_image(path_png, target_path)
