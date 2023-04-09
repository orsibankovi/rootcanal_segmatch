import rootcanal_segmatch
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class GetDataset(Dataset):
    def __init__(self):
        super(GetDataset, self).__init__()
        self.path_input = "/fogak/segmentation/original"
        self.input_names = list(filter(lambda x: x.endswith(".png"), list(self.path_input)))
        self.input_images = []
        for images in self.input_names:
            img_temp = cv2.imread(self.path_input + '/' + images, cv2.IMREAD_GRAYSCALE)
            img_temp = torch.from_numpy(img_temp.reshape((1, 501, 501))).double()
            self.input_images.append(img_temp)

        self.path_target = "/fogak/segmentation/inverse"
        self.target_names = list(filter(lambda x: x.endswith(".png"), list(self.path_target)))
        self.target_images = []
        for images in self.target_names:
            img_temp = cv2.imread(self.path_input + '/' + images, cv2.IMREAD_GRAYSCALE)
            img_temp = torch.from_numpy(img_temp.reshape((1, 501, 501))).double()
            self.target_images.append(img_temp)

    def __getitem__(self, index):
        return self.input_images[index], self.target_images[index]

    def __len__(self):
        return len(self.input_images)