import torch
import torch.nn as nn
import torch.optim as optim
from get_dataset import GetDataset as dataset
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if using bilinear interpolation, the output size is half the input size
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/ncullen93/pytorch-roi-align/issues/36
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


if __name__ == '__main__':
    trainset = dataset()
    i = trainset.__len__()
    print(i)
    x, y = trainset.__getitem__(80)
    print(x)
    print(y)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(dev)

    n_epochs = 1
    batch_size_train = 64
    log_interval = 10
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
    learning_rate = 0.001

    net = UNet(1, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    train_losses = []

    net.train()  # set mode of the NN

    for epoch in range(1, n_epochs + 1):  # hanyadik epochnal tartunk

        for batch_idx, (data, target) in enumerate(train_loader):

            data = data.to(dev).float()
            target = target.to(dev)

            optimizer.zero_grad()  # clear the gradient
            output = net(data)  # forward propagation
            loss = criterion(output, target)  # calculation loss
            loss.backward()  # current loss
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: ' + str(epoch)
                      + " batch_idx: "
                      + str(batch_idx)
                      + "\tLoss: "
                      + str(loss.item()))

                train_losses.append(loss.item())


    plt.plot(train_losses)