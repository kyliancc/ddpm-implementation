from unet.unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, t_channels=64):
        super().__init__()
        self.time_embedding = TimeEmbedding(t_channels)
        self.in_conv = In(in_channels, 64, t_channels)
        self.down1 = Down(64, 128, t_channels)
        self.down2 = Down(128, 256, t_channels)
        self.down3 = Down(in_channels=256, mid_channels=512, out_channels=256, t_channels=t_channels)
        self.up3 = Up(in_channels=512, mid_channels=256, out_channels=128, t_channels=t_channels)
        self.up2 = Up(in_channels=256, mid_channels=128, out_channels=64, t_channels=t_channels)
        self.up1 = Up(in_channels=128, out_channels=64, t_channels=t_channels)
        self.out_conv = Out(64, out_channels)

    def forward(self, x, time_step):
        t = self.time_embedding(time_step)
        x1 = self.in_conv(x, t)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x = self.down3(x3, t)
        x = self.up3(x, x3, t)
        x = self.up2(x, x2, t)
        x = self.up1(x, x1, t)
        return self.out_conv(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet()
model.to(device)

y = model(torch.randn([4, 1, 28, 28]).to(device), torch.tensor([3., 5., 7., 1.]).to(device))
print(y.shape)
