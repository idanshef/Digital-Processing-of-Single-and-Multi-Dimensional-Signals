import os
import cv2
from torch.utils.data import Dataset


def get_image_paths(data_dir):
    image_paths = []
    for root, _, files in os.walk(data_dir):
        image_paths = [os.path.join(root, x) for x in files]
    return image_paths


class CLICDataset(Dataset):
    def __init__(self, data_dir):
        self.image_paths = get_image_paths(data_dir)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        return {'image': image}