import torch.nn as nn
import torch as tr
import unet_model.Sampling as sampling


class Unet(nn.Module):
    def __init__(self, num_cat=2):
        super(Unet, self).__init__()
        self.down1 = nn.Sequential(
            sampling.ConvLayer(1, 64),
        )
        self.down2 = nn.Sequential(
            sampling.ConvLayer(64, 128),
            sampling.DownSample(),
        )
        self.down3 = nn.Sequential(
            sampling.ConvLayer(128, 256),
            sampling.DownSample(),
        )
        self.down4 = nn.Sequential(
            sampling.ConvLayer(256, 512),
            sampling.DownSample(),
        )
        self.middle = nn.Sequential(
            sampling.ConvLayer(512, 1024),
            sampling.UpSample(1024, 512),
        )

        self.cat4 = sampling.Cat()
        self.up4 = nn.Sequential(
            sampling.ConvLayer(1024, 512),
            sampling.UpSample(512, 256),
        )

        self.cat3 = sampling.Cat()
        self.up3 = nn.Sequential(
            sampling.ConvLayer(512, 256),
            sampling.UpSample(256, 128),
        )

        self.cat2 = sampling.Cat()
        self.up2 = nn.Sequential(
            sampling.ConvLayer(256, 128),
            sampling.UpSample(128, 64),
        )

        self.cat1 = sampling.Cat()
        self.up1 = nn.Sequential(
            sampling.ConvLayer(128, 64),
        )

        self.output = nn.Sequential(
            nn.Conv2d(64, num_cat, 3),
            nn.BatchNorm2d(num_cat),
            nn.ReLU(),
        )

    def forward(self, x):
        x1d = self.down1(x)
        x2d = self.down2(x1d)
        x3d = self.down3(x2d)
        x4d = self.down4(x3d)
        x_m = self.middle(x4d)
        x = self.up4(self.cat4(x_m, x4d))
        x = self.up3(self.cat3(x, x3d))
        x = self.up2(self.cat2(x, x2d))
        x = self.up1(self.cat1(x, x1d))
        x = self.output(x)
        return x
