import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset import CLICDataset
from wavelet_ae import WaveletAutoEncoder
from loss import CompressionLoss


class WaveletType:
    Daubechies1 = 0


def create_dataloaders(data_dir, batch_size):
    data_loaders = dict()

    for set_type in ['train', 'valid']:
        dataset_dir = os.path.join(data_dir, set_type)
        if not os.path.exists(dataset_dir):
            raise RuntimeError("Cannot find dataset at - %s" % dataset_dir)
        dataset = CLICDataset(dataset_dir)
        data_loaders[set_type] = DataLoader(dataset, shuffle=True, batch_size=batch_size,
                                            num_workers=4, pin_memory=True)
    return data_loaders


def compute_gaussian_probability(symbols, mu=0., var=1.):
    tmp = torch.pow(symbols - mu, 2) * 0.5 / var
    return torch.exp(-tmp)


def train_model(epochs, data_loader, model, loss_func, optimizer, device):
    
    for epoch in range(epochs):
        print("Training epoch %d/%d" % (epoch + 1, epochs))
        model.train()
        accumulated_loss = 0.
        num_batches = len(data_loader['train'])
        for batch_idx, patches in enumerate(data_loader["train"]):
            print("Training on batch %d/%d" % (batch_idx + 1, num_batches))
            optimizer.zero_grad()

            patches = patches.to(device)
            recon_patch, symbols = model(patches)

            full_loss, recon_loss_val, entropy_loss_val = loss_func(recon_patch, patches, symbols, compute_gaussian_probability(symbols))
            
            full_loss.backward()
            optimizer.step()

            full_loss_val = full_loss.item()
            
            print(f"\tFull loss: {full_loss_val}, recon loss: {recon_loss_val}, entropy loss: {entropy_loss_val}")
            accumulated_loss += full_loss_val
    
    return model


if __name__ == "__main__":
    data_dir = "/home/orweiser/university/Digital-Processing-of-Single-and-Multi-Dimensional-Signals/data"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = WaveletAutoEncoder(device)
    model = model.to(device=device)
    loss = CompressionLoss()
    # optimizer = optim.SGD(model.parameters(), weight_decay=1e-8, lr=0.01)
    optimizer = optim.Adam(model.parameters(), 1e-4)
    epochs = 100
    batch_size = 8
    
    dataloaders = create_dataloaders(data_dir, batch_size)
    train_model(epochs, dataloaders, model, loss, optimizer, device)