import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder
from pytorch_wavelets import DWTForward, DWTInverse


class WaveletAutoEncoder(nn.Module):
    def __init__(self, device, num_bits=10, wavelet_string="db1"):
        super().__init__()
        self.num_bits = num_bits
        self.noise_scale = 1. / 2. ** (num_bits - 1)
        self.wavelet_levels = 3
        self.wavelet_decomp = DWTForward(J=self.wavelet_levels, wave=wavelet_string)
        self.wavelet_recon = DWTInverse(wave=wavelet_string)
        self.encoder = Encoder(device)
        self.decoder = Decoder(device)
    
    def wavelet_decompose_image(self, img):
        yl, yh = self.wavelet_decomp(img)
        img_s3 = torch.cat([yl, *torch.unbind(yh[-1], dim=2)], dim=1)
        img_s2 = torch.cat(torch.unbind(yh[1], dim=2), dim=1)
        img_s1 = torch.cat(torch.unbind(yh[0], dim=2), dim=1)
        return img_s1, img_s2, img_s3

    def wavelet_reconstruct_image(self, out_s1, out_s2, out_s3):
        yl = out_s3[:, :3, ...]
        resolutions = []
        resolutions.append(torch.stack(torch.split(out_s1, 3, dim=1), dim=2))
        resolutions.append(torch.stack(torch.split(out_s2, 3, dim=1), dim=2))
        resolutions.append(torch.stack(torch.split(out_s3[:, 3:, ...], 3, dim=1), dim=2))
        return torch.tanh(self.wavelet_recon((yl, resolutions)))

        
    def forward(self, img):

        x_s1, x_s2, x_s3 = self.wavelet_decompose_image(img)
        
        y = self.encoder(x_s1, x_s2, x_s3)
        if self.training:
            quantization_noise = (2. * torch.rand_like(y) - 1.) * self.noise_scale
            y += quantization_noise
        else:
            y = ((0.5 * y + 0.5) * 2**self.num_bits).type(torch.int16)
            y = 2. * (y.float() / 2**self.num_bits) - 1.

        out_s1, out_s2, out_s3 = self.decoder(y)
        out = self.wavelet_reconstruct_image(out_s1, out_s2, out_s3)

        # wavelet_loss = F.mse_loss(out_s1, x_s1) + F.mse_loss(out_s2, x_s2) + F.mse_loss(out_s3, x_s3)
        return out, y#, wavelet_loss


if __name__ == "__main__":
    img = torch.rand(5, 3, 256, 256)
    net = WaveletAutoEncoder("cpu")
    out, y = net(img)
    print(out.shape)
    print(y.shape)