import torch
import torch.nn as nn
import torch.nn.functional as F
from gdn import GDN


class Decoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.decoder1 = Decoder1(device)
        self.decoder2 = Decoder2(device)
        self.decoder3 = Decoder3(device)
    
    def forward(self, x):
        assert x.shape[1] == 96, "number of channels must be 32"

        x1, x2, x3 = torch.split(x, 32, dim=1)

        output1 = self.decoder1(x1)
        output2 = self.decoder2(x2)
        output3 = self.decoder3(x3)
        return output1, output2, output3

class Decoder1(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.igdn_16 = GDN(16, device, inverse=True)
        self.igdn_32 = GDN(32, device, inverse=True)
        self.conv_32_32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv_32_16 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv_16_16 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(16, 9, kernel_size=1)
        

    def forward(self, x):
        x = self.conv_32_32(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.igdn_32(x)
        x = self.conv_32_32(x)
        x = self.igdn_32(x)
        x = self.conv_32_32(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_32_32(x)
        x = self.igdn_32(x)
        x = self.conv_32_32(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_32_16(x)
        x = self.igdn_16(x)
        x = self.conv_16_16(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_16_16(x)
        x = self.igdn_16(x)
        x = self.conv_16_16(x)
        x = self.out_conv(x)
        return x


class Decoder2(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.igdn_16 = GDN(16, device, inverse=True)
        self.igdn_32 = GDN(32, device, inverse=True)
        self.conv_32_32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv_32_16 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv_16_16 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(16, 9, kernel_size=1)

    def forward(self, x):
        x = self.conv_32_32(x)
        x = self.igdn_32(x)
        x = self.conv_32_32(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_32_32(x)
        x = self.igdn_32(x)
        x = self.conv_32_32(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_32_16(x)
        x = self.igdn_16(x)
        x = self.conv_16_16(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_16_16(x)
        x = self.igdn_16(x)
        x = self.conv_16_16(x)
        x = self.out_conv(x)
        return x


class Decoder3(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.igdn_16 = GDN(16, device, inverse=True)
        self.igdn_32 = GDN(32, device, inverse=True)
        self.conv_32_32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv_32_16 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv_16_16 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(16, 12, kernel_size=1)

    def forward(self, x):
        x = self.conv_32_32(x)
        x = self.igdn_32(x)
        x = self.conv_32_32(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_32_32(x)
        x = self.igdn_32(x)
        x = self.conv_32_32(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_32_16(x)
        x = self.igdn_16(x)
        x = self.conv_16_16(x)
        x = self.out_conv(x)
        return x

if __name__ == "__main__":
    e = torch.rand(10, 96, 8, 8)
    net = Decoder("cpu")
    s1, s2, s3 = net(e)
    print(s1.shape)
    print(s2.shape)
    print(s3.shape)
