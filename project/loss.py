import torch
from torch import nn


def compute_gaussian_probability(symbols, mu=0., var=1.):
    tmp = torch.pow(symbols - mu, 2) * 0.5 / var
    return torch.exp(-tmp)


class CompressionLoss(nn.Module):
    def __init__(self, weight=0.0025):
        super().__init__()
        self.reconstruction_loss = nn.MSELoss()
        self.weight = weight

    @staticmethod
    def entropy_loss(symbols, reduction_type='mean'):
        batch_size = symbols.shape[0]

        probabilities = compute_gaussian_probability(symbols)
        probabilities_hat = torch.softmax(symbols.reshape(batch_size, -1), dim=-1)
        entropy = -probabilities_hat * torch.log2(probabilities.reshape(batch_size, -1))
        if reduction_type == 'mean':
            return entropy.mean()
        if reduction_type == 'sum':
            return entropy.sum()
        
        return entropy

    def forward(self, x_recon, x_gt, symbols):

        recon_loss = self.reconstruction_loss(x_recon, x_gt)
        entropy_loss = self.entropy_loss(symbols)
        full_loss = recon_loss + self.weight * entropy_loss

        return  full_loss, recon_loss.item(), entropy_loss.item()
