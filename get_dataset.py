import os
import torch
import cv2
from torch.utils.data import Dataset


class GetDataset(Dataset):
    def __init__(self, size):
        super(GetDataset, self).__init__()
        self.path_input = "C:/Users/banko/Desktop/BME_VIK/I_felev/onlab1/fogak/segmentation/original"
        self.input = os.listdir(self.path_input)
        self.input_names = list(filter(lambda x: x.endswith(".png"), list(self.input)))
        self.input_images = []
        for images in self.input_names:
            img_temp = cv2.imread(self.path_input + '/' + images, cv2.IMREAD_GRAYSCALE)
            img_temp = cv2.resize(img_temp, (size, size))
            img_temp = torch.from_numpy(img_temp.reshape((1, size, size))).float()
            self.input_images.append(img_temp)

        self.path_target = "C:/Users/banko/Desktop/BME_VIK/I_felev/onlab1/fogak/segmentation/inverse"
        self.target = os.listdir(self.path_target)
        self.target_names = list(filter(lambda x: x.endswith(".png"), list(self.target)))
        self.target_images = []
        for images in self.target_names:
            img_temp = cv2.imread(self.path_target + '/' + images, cv2.IMREAD_GRAYSCALE)
            img_temp = cv2.resize(img_temp, (size, size))
            img_temp = torch.from_numpy(img_temp.reshape((1, size, size))).float()
            self.target_images.append(img_temp)

    def __getitem__(self, index):
        return self.input_images[index], self.target_images[index]

    def __len__(self):
        return len(self.input_images)