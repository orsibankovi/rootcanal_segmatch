import torch

import preprocess.get_dataset as ds
from models import unet, deeplabv3
import torchvision
from torchvision import transforms
from tests import test, metrics
from train import train
import test
import os
from natsort import natsorted
os.environ['CUDNN_BACKEND'] = 'tegra'

preprocess_input = transforms.Compose([
                      transforms.Resize(256),
                      transforms.ToTensor(),
            ])

preprocess_target = transforms.Compose([
                      transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
                      transforms.ToTensor(),
            ])

def save_images(input_image, target_image, des_filename, net, k):
    target_tensor = preprocess_target(target_image).to(dev).unsqueeze(1).float()
    input_tensor = preprocess_input(input_image).float()
    input_tensor = input_tensor.to(dev).float()
    output = net(input_tensor.unsqueeze(1).float())
    
    output_th = metrics.binary(output.cpu().squeeze(1))
    tf_tn_img = metrics.tf_fn_draw(output_th, target_tensor.squeeze(1))
    transform = torchvision.transforms.ToPILImage()
    image = transform(output_th)
    # tf = transform(tf_tn_img)
    tf_tn_img.save('./results/' + k + '/visualization/tf_tn/' + des_filename + '.png')
    image.save('./results/' + k + './visualization/unet_result/' + des_filename + '.png')


if __name__ == '__main__':
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #dev = torch.device("cpu")
    k = 'k_7'
    print(dev)
    '''
    train_dataset=ds.GetDataset(input_path='./fogak/segmentation/kfold/' + k + '/train/original', 
                                target_path='./fogak/segmentation/kfold/' + k + '/train/inverse', 
                                bool_augmentation=True)
                                '''
    #print('train_dataset: ', len(train_dataset))
    #train_len = int(len(train_dataset)*0.98)
    #train_set, validation_set = torch.utils.data.random_split(train_dataset, [train_len, len(train_dataset)-train_len])
    test_set=ds.GetDataset(input_path='./fogak/segmentation_rootcanal_only/original', 
                            target_path='./fogak/segmentation_rootcanal_only/inverse', 
                            bool_augmentation=False)
    print('test_dataset: ', len(test_set))

    net = unet.UNet(1, 1)
    
    net = torch.load('./trained_net.pt', map_location=dev)
    #train_set = torch.load('./finetune_after24epoch/train_set.pt')
    #validation_set = torch.load('./finetune_after24epoch/validation_set.pt')
    #net = deeplabv3.deeplabv3_resnet50(num_classes=2)
    
    if dev.type == 'cuda':
        net = net.to(dev)
    else:
        net = net.to(dev).float()
    
    #Train = train.Train(dev=dev, n_epoch=20, batch_size=8, lr=0.001, net=net, k=k)
    #trained_net = Train.run(trainset=train_set, validationset=validation_set, net=net)
    #torch.save(trained_net, './results/' + k + '/finished_trained_net.pt')
    dev = torch.device("cpu")
    net = net.to(dev)
    Test = test.Test(dev=dev, batch_size=1, testset=test_set, net=net, k=k)
    Test.run()

    '''
    #net = torch.load('./results/' + k + '/trained_net.pt')
    dev = torch.device("cpu")
    net = net.to(dev)
    original = './fogak/segmentation_rootcanal_only/original'
    orig_filenames = list(os.listdir(original))
    orig_png_filenames = list(filter(lambda x: x.endswith(".png"), orig_filenames))
    orig_png_filenames = natsorted(orig_png_filenames)
    inverse = './fogak/segmentation/kfold/' + k + '/test/inverse'
    inv_filenames = list(os.listdir(inverse))
    inv_png_filenames = list(filter(lambda x: x.endswith(".png"), inv_filenames))
    inv_png_filenames = natsorted(inv_png_filenames)
    for i in range(len(orig_png_filenames)):
        input_image = Image.open(f'./fogak/segmentation/kfold/k_7/test/original/{orig_png_filenames[i]}')
        target_image = Image.open(f'./fogak/segmentation/kfold/k_7/test/inverse/{inv_png_filenames[i]}')
        input_filename = orig_png_filenames[i].split('.')[0]  # Extract the filename without the extension
        target_filename = inv_png_filenames[i].split('.')[0]
        save_images(input_image, target_image, input_filename, net, k)
        '''

