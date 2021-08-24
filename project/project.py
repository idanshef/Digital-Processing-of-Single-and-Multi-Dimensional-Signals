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
    train_dir = os.path.join(data_dir, "professional_train_2020_256_256")
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
    normalized_patches = torch.zeros_like(patches)
    for i, patch in enumerate(patches):
        normalized_patches[i] = 2 * ((patch - patch.min()) / (patch.max() - patch.min())) - 1 
    return patches
        
def wavelet_transform(patches, wavelet_type=WaveletType.Daubechies1):
    if wavelet_type == WaveletType.Daubechies1:
        wavelet = pywt.Wavelet('db1')
    else:
        return
    
    r_channels = patches[:, :, :, 2]
    g_channels = patches[:, :, :, 1]
    b_channels = patches[:, :, :, 0]
    
    r_coeffs, g_coeffs, b_coeffs = [], [], []
    for r_channel, g_channel,b_channel in zip(r_channels, g_channels, b_channels):
        r_coeffs.append(pywt.wavedec2(r_channel, wavelet, level=3))
        g_coeffs.append(pywt.wavedec2(g_channel, wavelet, level=3))
        b_coeffs.append(pywt.wavedec2(b_channel, wavelet, level=3))
    return r_coeffs, g_coeffs, b_coeffs


def train_model(data_loader, M, N, model, loss_func, optimizer, device):
    channel2lvl3 = lambda patch: torch.stack((torch.tensor(patch[0]), torch.tensor(patch[1][0]),
                                                torch.tensor(patch[1][1]), torch.tensor(patch[1][2])))
    channel2lvl = lambda patch, lvl: torch.stack((torch.tensor(patch[4 - lvl][0]), torch.tensor(patch[4 - lvl][1]), 
                                                    torch.tensor(patch[4 - lvl][2])))
    
    for batch in data_loader["train"]:
        patches = batch["image"].type(torch.float32)
        patches = normalize_patches(patches)
        
        model.train()
        r_coeffs_lst, g_coeffs_lst, b_coeffs_lst = wavelet_transform(patches)
        lvl1tensor_batch, lvl2tensor_batch, lvl3tensor_batch = None, None, None
        for r_coeffs, g_coeffs, b_coeffs in zip(r_coeffs_lst, g_coeffs_lst, b_coeffs_lst):
            lvl1tensor = torch.cat((channel2lvl(r_coeffs, 1), channel2lvl(g_coeffs, 1), channel2lvl(b_coeffs, 1))).unsqueeze(0).to(torch.float)
            lvl2tensor = torch.cat((channel2lvl(r_coeffs, 2), channel2lvl(g_coeffs, 2), channel2lvl(b_coeffs, 2))).unsqueeze(0).to(torch.float)
            lvl3tensor = torch.cat((channel2lvl3(r_coeffs), channel2lvl3(g_coeffs), channel2lvl3(b_coeffs))).unsqueeze(0).to(torch.float)
            if lvl1tensor_batch is None:
                lvl1tensor_batch = lvl1tensor
                lvl2tensor_batch = lvl2tensor
                lvl3tensor_batch = lvl3tensor
            else:
                lvl1tensor_batch = torch.vstack((lvl1tensor_batch, lvl1tensor))
                lvl2tensor_batch = torch.vstack((lvl2tensor_batch, lvl2tensor))
                lvl3tensor_batch = torch.vstack((lvl3tensor_batch, lvl3tensor))
            
        optimizer.zero_grad()
        lvl1tensor_batch = lvl1tensor_batch.to(device=device)
        lvl2tensor_batch = lvl2tensor_batch.to(device=device)
        lvl3tensor_batch = lvl3tensor_batch.to(device=device)
        
        pred_patch = model(lvl1tensor_batch, lvl2tensor_batch, lvl3tensor_batch)
        loss = loss_func(pred_patch, patches)
        
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
    train_model(dataset, M, N, model, loss, optimizer, device)