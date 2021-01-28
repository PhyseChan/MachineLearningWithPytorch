import torch.nn as nn
import torch.nn.functional as F
import torch as tr

class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(self, ConvLayer).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x


class DownSample(nn.Module):
    def __int__(self):
        super(DownSample, self).__int__()
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel)

    def forward(self, x):
        x = self.up(x)
        return x


class Cat(nn.Module):
    def __init__(self):
        super(Cat, self).__init__()

    def forward(self, x_from_down, x_from_up):
        x_from_down_width, x_from_down_length = x_from_down.shape[2], x_from_down.shape[1]
        x_from_up_width, x_from_up_length = x_from_up.shape[2], x_from_up.shape[1]
        diff_width = x_from_down_width - x_from_up_width
        diff_length = x_from_down_length - x_from_up_length
        x_from_up = F.pad(x_from_up, [
            diff_width // 2, diff_width - diff_width // 2,
            diff_length // 2, diff_length - diff_length // 2
        ])
        x_from_up = tr.cat([x_from_up, x_from_down], dim=0)
        return x_from_up
