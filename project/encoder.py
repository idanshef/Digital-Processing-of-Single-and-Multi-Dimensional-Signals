import torch
import torch.nn as nn
import torch.nn.functional as F
from gdn import GDN


class Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.encoder1 = Encoder1(device)
        self.encoder2 = Encoder2(device)
        self.encoder3 = Encoder3(device)
    
    def forward(self, x_s1, x_s2, x_s3):
        output1 = self.encoder1(x_s1)
        output2 = self.encoder2(x_s2)
        output3 = self.encoder3(x_s3)
        output = torch.hstack((output1, output2, output3))
        return output

class Encoder1(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.gdn_16 = GDN(16, device)
        self.gdn_32 = GDN(32, device)
        self.conv_9_16 = nn.Conv2d(9, 16, kernel_size=3, padding=1)
        self.conv_16_16 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv_16_32 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv_32_32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        

    def forward(self, x):
        x = self.conv_9_16(x)
        x = self.gdn_16(x)
        x = self.conv_16_16(x)
        x = self.conv_16_16(x)
        x = F.interpolate(x, scale_factor=0.5)
        x = self.gdn_16(x)
        x = self.conv_16_16(x)
        x = self.conv_16_32(x)
        x = F.interpolate(x, scale_factor=0.5)
        x = self.gdn_32(x)
        x = self.conv_32_32(x)
        x = self.conv_32_32(x)
        x = F.interpolate(x, scale_factor=0.5)
        x = self.gdn_32(x)
        x = self.conv_32_32(x)
        x = self.gdn_32(x)
        x = self.conv_32_32(x)
        x = F.interpolate(x, scale_factor=0.5)
        return x


class Encoder2(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.gdn_16 = GDN(16, device)
        self.gdn_32 = GDN(32, device)
        self.conv_9_16 = nn.Conv2d(9, 16, kernel_size=3, padding=1)
        self.conv_16_16 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv_16_32 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv_32_32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_9_16(x)
        x = self.gdn_16(x)
        x = self.conv_16_16(x)
        x = self.conv_16_16(x)
        x = F.interpolate(x, scale_factor=0.5)
        x = self.gdn_16(x)
        x = self.conv_16_16(x)
        x = self.conv_16_32(x)
        x = F.interpolate(x, scale_factor=0.5)
        x = self.gdn_32(x)
        x = self.conv_32_32(x)
        x = self.conv_32_32(x)
        x = F.interpolate(x, scale_factor=0.5)
        x = self.gdn_32(x)
        x = self.conv_32_32(x)
        return x


class Encoder3(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.gdn_16 = GDN(16, device)
        self.gdn_32 = GDN(32, device)
        self.conv_12_16 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.conv_16_16 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv_16_32 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv_32_32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_12_16(x)
        x = self.gdn_16(x)
        x = self.conv_16_16(x)
        x = self.conv_16_16(x)
        x = F.interpolate(x, scale_factor=0.5)
        x = self.gdn_16(x)
        x = self.conv_16_16(x)
        x = self.conv_16_32(x)
        x = F.interpolate(x, scale_factor=0.5)
        x = self.gdn_32(x)
        x = self.conv_32_32(x)
        return x

if __name__ == "__main__":
    net = Encoder("cpu")
    s3 = torch.rand(1, 12, 32, 32)
    s2 = torch.rand(1, 9, 64, 64)
    s1 = torch.rand(1, 9, 128, 128)
    out = net(s1, s2, s3)
    print(out.shape)
