import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class AutoEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.encoder = Encoder(device)
        self.decoder = Decoder(device)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x