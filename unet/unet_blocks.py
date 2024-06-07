import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        """
        :param n_channels: Embedding dimensions of time step.
        """
        super().__init__()
        self.n_channels = n_channels

    def forward(self, t):
        """
        :param t: Time step with size [batch_size,].
        :return: Embedded time step with size [batch_size, n_channels].
        """
        t.unsqueeze_(-1)
        frac_len = math.ceil(self.n_channels / 2)
        frac = t / (torch.pow(10000, torch.arange(0, frac_len, 1) / self.n_channels))
        ret = torch.zeros([t.size(0), self.n_channels])
        ret[:,0::2] = torch.sin(frac)
        ret[:,1::2] = torch.cos(frac)
        return ret


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_channels):
        """
        :param in_channels: Input channels.
        :param out_channels: Output channels.
        :param t_channels: Time step embedding dimensions.
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.time = nn.Sequential(
            TimeEmbedding(t_channels),
            nn.Linear(t_channels, out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, t):
        """
        :param x: Input feature map with size [batch_size, in_channels, height, width].
        :param t: Input time steps with size [batch_size,].
        :return: Output feature map with size [batch_size, out_channels, height, width].
        """
        t = self.time(t)
        t = torch.reshape(t, [t.size(0), -1, 1, 1])
        x1 = self.conv1(x)
        x1 = x1 + t
        x1 = self.conv2(x1)
        x = self.shortcut(x) + x1
        return x


class AttentionBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

    def forward(self, x):
        # Dot product
        prod = torch.sum(x.mul(x), dim=1)
        score = F.softmax(torch.reshape(prod, [prod.size(0), -1]) / self.n_channels, dim=1)
        score = torch.reshape(score, [score.size(0), 1, x.size(2), x.size(3)])
        return torch.mul(score, x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 2, 1, 0, bias=False),
            nn.BatchNorm2d(n_channels)
        )

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(n_channels, n_channels, 2, 1, 0, bias=False),
            nn.BatchNorm2d(n_channels)
        )

    def forward(self, x):
        return self.down(x)
