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
        transforms.RandomRotation(30, interpolation=Image.NEAREST), 
        transforms.Resize(256, interpolation=Image.NEAREST),
        transforms.ToTensor()])

class GetDataset(Dataset):
    def __init__(self, input_path, target_path, bool_augmentation):
        super(GetDataset, self).__init__()
        self.input_images = self.load_images(input_path, bool_augmentation, preprocess_input)
        self.target_images = self.load_images(target_path, bool_augmentation, preprocess_target)
        print(self.target_images[0].shape)

    def load_images(self, path, bool_augmentation, preprocess_func):
        images = os.listdir(path)
        image_names = list(filter(lambda x: x.endswith(".png"), images))
        sorted_images = natsorted(image_names)
        loaded_images = []

        for count, image_name in enumerate(sorted_images):
            image = Image.open(os.path.join(path, image_name)).convert('L')
            image_tensor = preprocess_func(image).float()
            loaded_images.append(image_tensor)

            if bool_augmentation and count % 3 == 0:
                image_tensor = augmentation(image).float()
                loaded_images.append(image_tensor)

        return loaded_images

    def __getitem__(self, index):
        return self.input_images[index], self.target_images[index]

    def __len__(self):
        return len(self.input_images)