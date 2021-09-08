import torch
from torch import nn


class CompressionLoss(nn.Module):
    def __init__(self, weight=0.0025):
        super().__init__()

        self.reconstruction_loss = nn.MSELoss()
        self.weight = weight

    @staticmethod
    def entropy_loss(symbols, probabilities, reduction_type='mean'):
        entropy = symbols * torch.log2(probabilities)
        if reduction_type == 'mean':
            return entropy.mean()
        if reduction_type == 'sum':
            return entropy.sum()
        
        return entropy

    def forward(self, x_recon, x_gt, symbols, probablities):
        recon_loss = self.reconstruction_loss(x_recon, x_gt)
        entropy_loss = self.entropy_loss(symbols, probablities)
        full_loss = recon_loss + self.weight * entropy_loss
        return  full_loss, recon_loss.item(), entropy_loss.item()