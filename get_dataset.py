import os
import torch
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from natsort import natsorted


preprocess_input = transforms.Compose([
                      transforms.Resize(256),
                      transforms.ToTensor(),
            ])

preprocess_target = transforms.Compose([
                      transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
                      transforms.ToTensor(),
            ])

augmentation = transforms.Compose([
        transforms.RandomRotation(35, interpolation=Image.NEAREST), 
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256, interpolation=Image.NEAREST),
        transforms.ToTensor()])

class GetDataset(Dataset):
    def __init__(self):
        super(GetDataset, self).__init__()
        self.path_input = "./fogak/segmentation/original"
        self.input = os.listdir(self.path_input)
        self.input_names = list(filter(lambda x: x.endswith(".png"), list(self.input)))
        self.sorted_input = natsorted(self.input_names)
        self.input_images = []
        for count, images in enumerate(self.sorted_input[:]):
            input_image = Image.open(self.path_input + '/' + images)
            input_tensor = preprocess_input(input_image).float()
            self.input_images.append(input_tensor)
            if count%3 == 0:
                input_tensor = augmentation(input_image).float()
                self.input_images.append(input_tensor)
                
            # print(torch.max(input_tensor))

        self.path_target = "./fogak/segmentation/inverse"
        self.target = os.listdir(self.path_target)
        self.target_names = list(filter(lambda x: x.endswith(".png"), list(self.target)))
        self.sorted_target = natsorted(self.target_names)
        self.target_images = []
        for count, images in enumerate(self.sorted_target[:]):
            target_image = Image.open(self.path_target + '/' + images)
            #print(np.asarray(target_image).max())
            target_tensor = preprocess_target(target_image).float()
            self.target_images.append(target_tensor)
            if count%3 == 0:
                target_tensor = augmentation(target_image).float()
                self.target_images.append(target_tensor)
            # print(torch.max(target_tensor))
        print(self.target_images[0].shape)

    def __getitem__(self, index):
        return self.input_images[index], self.target_images[index]

    def __len__(self):
        return len(self.input_images)