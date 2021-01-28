import torch.nn as nn
from UpperSampling import UpSampling

'''
    input 256*256 images
'''


class SimpleUnet(nn.Module):
    def __init__(self):
        super(SimpleUnet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3),  # 510*510*64
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),  # 508*508*64
            nn.ReLU(),

        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2),  # 254*254
            nn.Conv2d(64, 128, 3),  # 252*252*128
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),  # 250*250*128
            nn.ReLU(),

        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2),  # 125*125*128
            nn.Conv2d(128, 256, 3),  # 123*123*256
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),  # 121*121*256
            nn.ReLU(),

        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2),  # 60*60*128
            nn.Conv2d(256, 512, 3),  # 58*58*512
            nn.ReLU(),
            nn.Conv2d(512, 512, 3),  # 56*56*512
            nn.ReLU(),

        )
        self.middleConv = nn.Sequential(
            nn.MaxPool2d(2),  # 28*28*512
            nn.Conv2d(512, 1024, 3),  # 26*26*1024
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3),  # 24*24*1024
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 2, 2),  # 48*48*512
            nn.ReLU(),
        )
        self.tranconv4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3),  # 46*46*512
            nn.ReLU(),
            nn.Conv2d(512, 512, 3),  # 44*44*512
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, 2),  # 88*88*256
            nn.ReLU(),
        )
        self.tranconv3 = nn.Sequential(
            nn.Conv2d(512, 256, 3),  # 86*86*256
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),  # 84*84*256
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, 2),  # 168*168*128
            nn.ReLU(),

        )
        self.tranconv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3),  # 166*166*128
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),  # 164*164*128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, 2),  # 328*328*64
            nn.ReLU(),
        )
        self.tranconv1 = nn.Sequential(
            nn.Conv2d(128, 64, 3),  # 326*326*64
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),  # 324*324*64
            nn.ReLU(),
            nn.Conv2d(64, 2, 3),  # 322*322*2
            nn.ReLU(),
        )
        self.upsample = UpSampling()


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        xm = self.middleConv(x4)
        tx4 = self.tranconv4(self.upsample(xm, x4))
        tx3 = self.tranconv3(self.upsample(tx4, x3))
        tx2 = self.tranconv2(self.upsample(tx3, x2))
        tx1 = self.tranconv1(self.upsample(tx2, x1))

        return tx1
