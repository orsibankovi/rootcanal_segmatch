import os
import unet
import rootcanal_segmatch
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class GetDataset(Dataset):
    def __init__(self):
        super(GetDataset, self).__init__()
        self.path_input = "C:/Users/banko/Desktop/BME_VIK/I_felev/onlab1/fogak/segmentation/original"
        self.input = os.listdir(self.path_input)
        self.input_names = list(filter(lambda x: x.endswith(".png"), list(self.input)))
        self.input_images = []
        for images in self.input_names:
            img_temp = cv2.imread(self.path_input + '/' + images, cv2.IMREAD_GRAYSCALE)
            img_temp = torch.from_numpy(img_temp.reshape((1, 501, 501))).int()
            self.input_images.append(img_temp)

        self.path_target = "C:/Users/banko/Desktop/BME_VIK/I_felev/onlab1/fogak/segmentation/inverse"
        self.target = os.listdir(self.path_target)
        self.target_names = list(filter(lambda x: x.endswith(".png"), list(self.target)))
        self.target_images = []
        for images in self.target_names:
            img_temp = cv2.imread(self.path_target + '/' + images, cv2.IMREAD_GRAYSCALE)
            img_temp = torch.from_numpy(img_temp.reshape((1, 501, 501))).int()
            self.target_images.append(img_temp)

    def __getitem__(self, index):
        return self.input_images[index], self.target_images[index]

    def __len__(self):
        return len(self.input_images)


trainset = GetDataset()
i = trainset.__len__()
print(i)
x, y = trainset.__getitem__(100)
print(x)
print(y)

n_epochs = 3
batch_size_train = 64

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)

net = unet.UNet()