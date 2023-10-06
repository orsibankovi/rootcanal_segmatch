import torch
import numpy as np
from PIL import Image
import get_dataset as ds
import unet
import torchvision
from torchvision import transforms
import metrics
import train
import test
import os
os.environ['CUDNN_BACKEND'] = 'tegra'

preprocess_input = transforms.Compose([
                      transforms.Resize(256),
                      transforms.ToTensor(),
            ])

preprocess_target = transforms.Compose([
                      transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
                      transforms.ToTensor(),
            ])

def save_images(input_image, target_image, des_filename, net):
    target_tensor = preprocess_target(target_image).to(dev).unsqueeze(1).float()
    input_tensor = preprocess_input(input_image).float()
    input_tensor = input_tensor.to(dev).float()
    output = net(input_tensor.unsqueeze(1).float())
    
    output_th = metrics.binary(output.cpu().squeeze(1))
    tf_tn_img = metrics.tf_fn_draw(output_th, target_tensor.squeeze(1))
    transform = torchvision.transforms.ToPILImage()
    image = transform(output_th)
    # tf = transform(tf_tn_img)
    tf_tn_img.save(des_filename + 'tf_tn_img' +  '.png')
    image.save(des_filename + 'unet_result' + '.png')


if __name__ == '__main__':
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(dev)
    dataset=ds.GetDataset()
    train_len = int(len(dataset)*0.9)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_len, len(dataset)-train_len])
    torch.save(train_set, 'train_set.pt')
    validation_len = int(len(test_set)*0.5)
    validation_set, test_set = torch.utils.data.random_split(test_set, [validation_len, len(test_set)-validation_len])
    torch.save(test_set, 'test_set.pt')
    torch.save(validation_set, 'validation_set.pt')

    net = unet.UNet(1, 1)

    if dev.type == 'cuda':
        net = net.to(dev)
    else:
        net = net.float()
    
    Train = train.Train(dev=dev, n_epoch=20, batch_size=4, lr=0.001, net=net)
    trained_net = Train.run(trainset=train_set, validationset=validation_set, net=net)
    Test = test.Test(dev=dev, batch_size=1, testset=test_set, net=trained_net)
    Test.run()
    input_image = Image.open('./fogak/segmentation/original/CBCT 7d_100_239_original.png')
    target_image = Image.open('./fogak/segmentation/inverse/CBCT 7d_100_239_rootcanal.png')
    save_images(input_image, target_image, 'CBCT_7d_100_239_', trained_net)

