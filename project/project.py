import os
import pywt
import itertools
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from dataset import CLICDataset
from autoencoder import AutoEncoder


class WaveletType:
    Daubechies1 = 0

def load_data(data_dir, batch_size):
    train_dir = os.path.join(data_dir, "professional_train_2020")
    val_dir = os.path.join(data_dir, "professional_valid_2020")
    
    train_dataset = CLICDataset(train_dir)
    val_dataset = CLICDataset(val_dir)

    data_loaders = dict()
    data_loaders['train'] = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4,
                                       pin_memory=True)
    data_loaders['val'] = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4,
                                     pin_memory=True)
    
    return data_loaders


def split_to_patches(img, M, N):
    w_count = img.shape[0] // M
    h_count = img.shape[1] // N
    patches = [img[h_idx * N : (h_idx + 1) * N, w_idx * M : (w_idx + 1) * M, :] for h_idx, w_idx in itertools.product(range(h_count), range(w_count))]
    return patches


def normalize_patches(patches):
    patches = [2 * ((patch - patch.min()) / (patch.max() - patch.min())) - 1 for patch in patches]
    return patches
        
def wavelet_transform(patches, wavelet_type=WaveletType.Daubechies1):
    if wavelet_type == WaveletType.Daubechies1:
        wavelet = pywt.Wavelet('db1')
    else:
        return
    r_channel = patches[:, :, 2]
    g_channel = patches[:, :, 1]
    b_channel = patches[:, :, 0]
    
    r_coeffs = pywt.wavedec2(r_channel, wavelet, level=3)
    g_coeffs = pywt.wavedec2(g_channel, wavelet, level=3)
    b_coeffs = pywt.wavedec2(b_channel, wavelet, level=3)
    return r_coeffs, g_coeffs, b_coeffs


def algo(data_loader, M, N, model, loss_func, optimizer):
    for batch in data_loader:
        patches = split_to_patches(img, M, N)
        patches = normalize_patches(patches)
        
        model.train()
        
        channel2lvl3 = lambda patch: torch.stack((torch.tensor(patch[0]), torch.tensor(patch[1][0]),
                                                    torch.tensor(patch[1][1]), torch.tensor(patch[1][2])))
        channel2lvl = lambda patch, lvl: torch.stack((torch.tensor(patch[4 - lvl][0]), torch.tensor(patch[4 - lvl][1]), 
                                                        torch.tensor(patch[4 - lvl][2])))
        for patch in patches:
            patch_r, patch_g, patch_b = wavelet_transform(patch)
            lvl1tensor = torch.cat((channel2lvl(patch_r, 1), channel2lvl(patch_g, 1), channel2lvl(patch_b, 1))).unsqueeze(0).to(torch.float)
            lvl2tensor = torch.cat((channel2lvl(patch_r, 2), channel2lvl(patch_g, 2), channel2lvl(patch_b, 2))).unsqueeze(0).to(torch.float)
            lvl3tensor = torch.cat((channel2lvl3(patch_r), channel2lvl3(patch_g), channel2lvl3(patch_b))).unsqueeze(0).to(torch.float)
            
            optimizer.zero_grad()
            pred_patch = model(lvl1tensor, lvl2tensor, lvl3tensor)
            loss = loss_func(pred_patch, patch)
            
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    data_dir = "/home/idansheffer/data_others"
    N, M = 256, 256
    
    #         M
    #     ---------
    #    |         |
    #  N |         |
    #    |         |
    #     ---------
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AutoEncoder(device)
    model = model.to(device=device) # TODO: is it needed?
    loss = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), weight_decay=1e-8, lr=0.01)
    optimizer = optim.Adam(model.parameters(), 1e-4)
    epochs = 100
    batch_size = 8
    
    dataset = load_data(data_dir, batch_size)
    algo(dataset, M, N, model, loss, optimizer)