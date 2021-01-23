import torch.nn as nn
from UpperSampling import UpSampling

'''
    input 256*256 images
'''


class SimpleUnet(nn.Module):
    def __init__(self):
        super(SimpleUnet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3),  # 254*254*64
            nn.Conv2d(64, 64, 3),  # 252*252*64
            nn.MaxPool2d(2),  # 126*126
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),  # 124*124*128
            nn.Conv2d(128, 128, 3),  # 122*122*128
            nn.MaxPool2d(2),  # 56*56*128
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3),  # 54*54*256
            nn.Conv2d(256, 256, 3),  # 52*52*256
            nn.MaxPool2d(2),  # 26*26*128
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3),  # 24*24*512
            nn.Conv2d(512, 512, 3),  # 22*22*512
            nn.MaxPool2d(2),  # 11*11*512
        )
        self.middleConv = nn.Sequential(
            nn.Conv2d(512, 1024, 3),  # 9*9*1024
            nn.Conv2d(1024, 1024, 3),  # 7*7*1024
            nn.ConvTranspose2d(1024, 512, 2, 2),  # 14*14*512
        )
        self.tranconv4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3),  # 12*12*512
            nn.Conv2d(512, 512, 3),  # 24*24*512
            nn.ConvTranspose2d(512, 256, 2, 2),  # 48*48*256
        )
        self.tranconv3 = nn.Sequential(
            nn.Conv2d(512, 256, 3),  # 46*46*256
            nn.Conv2d(256, 256, 3),  # 44*44*256
            nn.ConvTranspose2d(256, 128, 2, 2)  # 88*88*128
        )
        self.tranconv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3),  # 86*86*128
            nn.Conv2d(128, 128, 3),  # 84*84*128
            nn.ConvTranspose2d(128, 64, 2, 2)  # 168*168*64
        )
        self.tranconv1 = nn.Sequential(
            nn.Conv2d(128, 64, 3),  # 166*166*64
            nn.Conv2d(64, 64, 3),  # 164*164*64
            nn.Conv2d(64, 2, 3),  # 162*162*2
        )
        self.upsample4 = UpSampling(self.tranconv4)
        self.upsample3 = UpSampling(self.tranconv3)
        self.upsample2 = UpSampling(self.tranconv2)
        self.upsample1 = UpSampling(self.tranconv1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        xm = self.middleConv(x4)

        tx4 = self.upsample4(xm, x4)
        tx3 = self.upsample4(tx4, x3)
        tx2 = self.upsample4(tx3, x2)
        tx1 = self.upsample4(tx2, x1)

        return tx1
