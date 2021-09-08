import os
from glob import glob
from torch.utils.data import Dataset
from torchvision.io import read_image


class CLICDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.image_paths = glob(os.path.join(data_dir, "*"))
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = 2. * read_image(image_path).float() / 255. - 1.

        if self.transforms:
            image = self.transforms(image)
        
        return image

if __name__ == "__main__":
    dataset_dir = "/home/orweiser/university/Digital-Processing-of-Single-and-Multi-Dimensional-Signals/data/valid"
    dataset = CLICDataset(dataset_dir)
    # img = dataset[5]
    print(len(dataset))