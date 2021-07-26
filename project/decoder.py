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
        # TODO: split x to 3?
        output1 = self.decoder1(x)
        output2 = self.decoder2(x)
        output3 = self.decoder3(x)
        return output1, output2, output3

class Decoder1(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.igdn_16 = GDN(16, device, inverse=True)
        self.igdn_32 = GDN(32, device, inverse=True)
        self.conv_32_32 = nn.Conv2d(32, 32, kernel_size=3, stride=0, padding=0)
        self.conv_32_16 = nn.Conv2d(32, 16, kernel_size=3, stride=0, padding=0)
        self.conv_16_16 = nn.Conv2d(16, 16, kernel_size=3, stride=0, padding=0)
        self.conv_16_1 = nn.Conv2d(16, 1, kernel_size=3, stride=0, padding=0) # TODO:?
        

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
        return x


class Decoder2(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.igdn_16 = GDN(16, device, inverse=True)
        self.igdn_32 = GDN(32, device, inverse=True)
        self.conv_32_32 = nn.Conv2d(32, 32, kernel_size=3, stride=0, padding=0)
        self.conv_32_16 = nn.Conv2d(32, 16, kernel_size=3, stride=0, padding=0)
        self.conv_16_16 = nn.Conv2d(16, 16, kernel_size=3, stride=0, padding=0)
        self.conv_16_1 = nn.Conv2d(16, 1, kernel_size=3, stride=0, padding=0) # TODO:?

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
        return x


class Decoder3(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.igdn_16 = GDN(16, device, inverse=True)
        self.igdn_32 = GDN(32, device, inverse=True)
        self.conv_32_32 = nn.Conv2d(32, 32, kernel_size=3, stride=0, padding=0)
        self.conv_32_16 = nn.Conv2d(32, 16, kernel_size=3, stride=0, padding=0)
        self.conv_16_16 = nn.Conv2d(16, 16, kernel_size=3, stride=0, padding=0)
        self.conv_16_1 = nn.Conv2d(16, 1, kernel_size=3, stride=0, padding=0) # TODO:?

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
        return x