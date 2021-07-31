import cv2
import pywt
import itertools
import torch
from autoencoder import AutoEncoder


class WaveletType:
    Daubechies1 = 0

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


def algo(img, M, N):
    patches = split_to_patches(img, M, N)
    patches = normalize_patches(patches)
    for patch in patches:
        patch_r, patch_g, patch_b = wavelet_transform(patch)
        channel2lvl1 = lambda patch: torch.stack((torch.tensor(patch[0]).unsqueeze(0), torch.tensor(patch[1][0]).unsqueeze(0),
                                                  torch.tensor(patch[1][1]).unsqueeze(0), torch.tensor(patch[1][2]).unsqueeze(0)))
        channel2lvl = lambda patch, lvl: torch.stack((torch.tensor(patch[lvl][0]).unsqueeze(0), torch.tensor(patch[lvl][1]).unsqueeze(0), 
                                                      torch.tensor(patch[lvl][2]).unsqueeze(0)))
        
        lvl1tensor = torch.cat((channel2lvl1(patch_r), channel2lvl1(patch_g), channel2lvl1(patch_b)))
        lvl2tensor = torch.cat((channel2lvl(patch_r, 2), channel2lvl(patch_g, 2), channel2lvl(patch_b, 2)))
        lvl3tensor = torch.cat((channel2lvl(patch_r, 3), channel2lvl(patch_g, 3), channel2lvl(patch_b, 3)))
        
        print("hi")

if __name__ == "__main__":
    img_path = r"C:\Users\isheffer\OneDrive - Intel Corporation\Desktop\university\Digital-Processing-of-Single-and-Multi-Dimensional-Signals\finalProject\Lenna.png"
    N, M = 256, 256
    
    #         M
    #     ---------
    #    |         |
    #  N |         |
    #    |         |
    #     ---------
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder(device)
    model = model.to(device=device)
    # TODO: loss 
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    algo(img, M, N)