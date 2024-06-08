from unet.unet_blocks import *


class DDPMUNet(nn.Module):
    def __init__(self, device, in_channels=3, out_channels=3, t_channels=128):
        super().__init__()
        self.time_embedding = TimeEmbedding(t_channels, device)
        self.in_conv = Conv(in_channels, 64)
        self.down1 = Down(64, 64, t_channels)
        self.down_sample1 = Downsample(64)
        self.down2 = Down(64, 128, t_channels)
        self.down_sample2 = Downsample(128)
        self.down3 = Down(128, 256, t_channels)
        self.down_sample3 = Downsample(256)
        self.down4 = Down(256, 1024, t_channels)
        self.mid = MiddleBlock(1024, 1024, 1024, t_channels)
        self.up4 = Up(1024, 256, t_channels)
        self.up_sample3 = Upsample(256)
        self.up3 = Up(256, 128, t_channels)
        self.up_sample2 = Upsample(128)
        self.up2 = Up(128, 64, t_channels)
        self.up_sample1 = Upsample(64)
        self.up1 = Up(64, 64, t_channels)
        self.out_conv = Conv(64, out_channels)

    def forward(self, x, time_step):
        t = self.time_embedding(time_step)
        x11, x12, x13 = self.down1(self.in_conv(x), t)
        x21, x22, x23 = self.down2(self.down_sample1(x13), t)
        x31, x32, x33 = self.down3(self.down_sample2(x23), t)
        x41, x42, x43 = self.down4(self.down_sample3(x33), t)
        x = self.up4(self.mid(x43, t), x41, x42, x43, t)
        x = self.up3(self.up_sample3(x), x31, x32, x33, t)
        x = self.up2(self.up_sample2(x), x21, x22, x23, t)
        x = self.up1(self.up_sample1(x), x11, x12, x13, t)
        x = self.out_conv(x)
        return x
