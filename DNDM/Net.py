import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Net, self).__init__()
        self.input_nc = input_nc

        self.conv1_1 = nn.Conv2d(input_nc, 32, 3, padding=1)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(2)

        self.deconv8 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=1)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, output_nc, 1)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            init.xavier_uniform_(m.bias.data)

    def forward(self, input):
        x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))  
        conv1 = x
        x = self.max_pool1(x)

        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        x = self.max_pool2(x)


        conv8 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        up9 = torch.cat([self.deconv8(conv8), conv1], 1)
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))

        latent = self.conv10(conv9)

        return latent