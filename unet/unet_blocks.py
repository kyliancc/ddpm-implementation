import torch
import torch.nn as nn

import math


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        
    def forward(self, t):
        frac_len = math.ceil(self.n_channels / 2)
        frac = t / (torch.pow(10000, torch.arange(0, frac_len, 1) / self.n_channels))
        ret = torch.zeros([self.n_channels, 1, 1])
        ret[0::2,0,0] = torch.sin(frac)
        ret[1::2,0,0] = torch.cos(frac)
        return ret


class DownBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return
