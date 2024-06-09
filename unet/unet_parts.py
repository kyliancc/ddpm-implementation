import torch
import torch.nn as nn

import math


class TimeEmbedding(nn.Module):
    def __init__(self, t_channels):
        super().__init__()
        self.t_channels = t_channels

    def forward(self, time_step):
        time_step.unsqueeze_(-1)
        frac_len = math.ceil(self.t_channels / 2)
        frac = time_step / (torch.pow(10000, torch.arange(0, frac_len, 1, device=time_step.device) / self.t_channels))
        ret = torch.zeros([time_step.size(0), self.t_channels], device=time_step.device)
        ret[:, 0::2] = torch.sin(frac)
        ret[:, 1::2] = torch.cos(frac)
        return ret


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, t_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.time = nn.Sequential(
            nn.Linear(t_channels, mid_channels)
        )

    def forward(self, x, t):
        x = self.conv1(x)
        t = self.time(t)
        t = torch.reshape(t, [t.size(0), -1, 1, 1])
        x = x + t
        x = self.conv2(x)
        return x


class Downsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(x)


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        return self.upsample(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, t_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.down = Downsample()
        self.conv = DoubleConv(in_channels, out_channels, mid_channels, t_channels)

    def forward(self, x, t):
        x = self.down(x)
        x = self.conv(x, t)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, t_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.up = Upsample()
        self.conv = DoubleConv(in_channels, out_channels, mid_channels, t_channels)

    def forward(self, x, x_skip, t):
        x = self.up(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv(x, t)
        return x


class In(nn.Module):
    def __init__(self, in_channels, out_channels, t_channels):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.double_conv = DoubleConv(out_channels, out_channels, out_channels, t_channels)

    def forward(self, x, t):
        x = self.in_conv(x)
        x = self.double_conv(x, t)
        return x


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 5, 1, 0)
        )

    def forward(self, x):
        return self.conv(x)
