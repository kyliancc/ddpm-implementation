import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class TimeEmbedding(nn.Module):
    def __init__(self, t_channels, device):
        super().__init__()
        self.t_channels = t_channels
        self.device = device

    def forward(self, time_step):
        """
        :param time_step: Time steps with size [batch_size,].
        :return: Embedded time steps with size [batch_size, n_channels].
        """
        time_step.unsqueeze_(-1)
        frac_len = math.ceil(self.t_channels / 2)
        frac = time_step / (torch.pow(10000, torch.arange(0, frac_len, 1, device=self.device) / self.t_channels))
        ret = torch.zeros([time_step.size(0), self.t_channels], device=self.device)
        ret[:, 0::2] = torch.sin(frac)
        ret[:, 1::2] = torch.cos(frac)
        return ret


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        :param x: Input with size [batch_size, in_channels, height, width].
        :return: Output with size [batch_size, out_channels, height, width].
        """
        return self.conv(x)


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, t_channels):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels)
        self.conv2 = Conv(out_channels, out_channels)
        self.time = nn.Sequential(
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
        :param x: Input feature maps with size [batch_size, in_channels, height, width].
        :param t: Input time embeddings with size [batch_size, t_channels].
        :return: Output feature maps with size [batch_size, out_channels, height, width].
        """
        t = self.time(t)
        t = torch.reshape(t, [t.size(0), -1, 1, 1])
        x1 = self.conv1(x)
        x1 = x1 + t
        x1 = self.conv2(x1)
        x = self.shortcut(x) + x1
        return x


class Attention(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

    def forward(self, x):
        """
        :param x: Input feature maps with size [batch_size, n_channels, height, width].
        :return: Output feature maps with size [batch_size, n_channels, height, width].
        """
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
        """
        :param x: Input feature maps with size [batch_size, n_channels, height, width].
        :return: Output feature maps with size [batch_size, n_channels, height, width].
        """
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(n_channels, n_channels, 2, 1, 0, bias=False),
            nn.BatchNorm2d(n_channels)
        )

    def forward(self, x):
        """
        :param x: Input feature maps with size [batch_size, n_channels, height, width].
        :return: Output feature maps with size [batch_size, n_channels, height, width].
        """
        return self.down(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_channels):
        super().__init__()
        self.residual = Residual(in_channels, out_channels, t_channels)
        self.attention = Attention(out_channels)

    def forward(self, x, t):
        """
        :param x: Input feature maps with size [batch_size, in_channels, height, width].
        :param t: Input time embeddings with size [batch_size, t_channels].
        :return: Output feature maps with size [batch_size, out_channels, height, width].
        """
        x = self.residual(x, t)
        x = self.attention(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_channels):
        super().__init__()
        self.residual = Residual(in_channels, out_channels, t_channels)
        self.attention = Attention(out_channels)

    def forward(self, x, t):
        """
        :param x: Input feature maps with size [batch_size, in_channels, height, width].
        :param t: Input time embeddings with size [batch_size, t_channels].
        :return: Output feature maps with size [batch_size, out_channels, height, width].
        """
        x = self.residual(x, t)
        x = self.attention(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, t_channels):
        super().__init__()
        self.residual1 = Residual(in_channels, mid_channels, t_channels)
        self.residual2 = Residual(mid_channels, out_channels, t_channels)
        self.attention = Attention(mid_channels)

    def forward(self, x, t):
        """
        :param x: Input feature maps with size [batch_size, in_channels, height, width].
        :param t: Input time embeddings with size [batch_size, t_channels].
        :return: Output feature maps with size [batch_size, out_channels, height, width].
        """
        x = self.residual1(x, t)
        x = self.attention(x)
        x = self.residual2(x, t)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, t_channels):
        super().__init__()
        self.down1 = DownBlock(in_channels, out_channels, t_channels)
        self.down2 = DownBlock(out_channels, out_channels, t_channels)

    def forward(self, x1, t):
        """
        :param x1: Input with size [batch_size, in_channels, height, width].
        :param t: Input time embeddings with size [batch_size, t_channels].
        :return: Outputs to feed to decoder.
        """
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        return x1, x2, x3


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, t_channels):
        super().__init__()
        self.up1 = UpBlock(2 * in_channels, in_channels, t_channels)
        self.up2 = UpBlock(2 * in_channels, in_channels, t_channels)
        self.up3 = UpBlock(in_channels + out_channels, out_channels, t_channels)

    def forward(self, x, x1, x2, x3, t):
        """
        :param x: Input with size [batch_size, in_channels, height, width].
        :param x1: From encoder.
        :param x2: From encoder.
        :param x3: From encoder.
        :param t: Input time embeddings with size [batch_size, t_channels].
        :return: Output with size [batch_size, out_channels, height, width].
        """
        x = torch.cat([x, x3], dim=1)
        x = self.up1(x, t)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, t)
        x = torch.cat([x, x1], dim=1)
        x = self.up3(x, t)
        return x
