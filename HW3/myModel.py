import torch.nn as nn
from torchvision import models

class Resnet(nn.Module):
    def __init__(self, numChannel):
        super(Resnet, self).__init__()
        self.numChannel = numChannel
        self.conv = nn.Sequential(
            nn.Conv2d(numChannel, numChannel, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(numChannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(32)
        self.resnet1 = Resnet(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(64)
        self.resnet2 = Resnet(64)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(128)
        self.resnet3 = Resnet(128)
        self.conv4   = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(256)
        self.resnet4 = Resnet(256)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(512)
        self.resnet5 = Resnet(512)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):

        x1 = self.bnd1(self.relu(self.conv1(x)))
        x1_1 = self.resnet1(x1)
        x2 = self.bnd2(self.relu(self.conv2(x1 + x1_1)))
        x2_2 = self.resnet2(x2)
        x3 = self.bnd3(self.relu(self.conv3(x2 + x2_2)))
        x3_3 = self.resnet3(x3)
        x4 = self.bnd4(self.relu(self.conv4(x3 + x3_3)))
        x4_4 = self.resnet4(x4)
        out_encoder = self.bnd5(self.relu(self.conv5(x4 + x4_4)))
        out_encoder_5 = self.resnet5(out_encoder)

        y1 = self.bn1(self.relu(self.deconv1(out_encoder + out_encoder_5)))    
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        out_decoder = self.bn5(self.relu(self.deconv5(y4)))
        
        score = self.classifier(out_decoder)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)
