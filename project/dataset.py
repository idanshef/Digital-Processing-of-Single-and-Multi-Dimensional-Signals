import os
from glob import glob
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader


class CLICDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.image_paths = glob(os.path.join(data_dir, "*"))
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = read_image(image_path).float() / 255.  # image is [0., 1.]

        if self.transforms:
            image = self.transforms(image)
        
        return image


def create_dataloaders(data_dir, batch_size):
    data_loaders = dict()
    batch_sizes = dict()
    
    if type(batch_size) == int:
        batch_sizes['train'] = batch_size
        batch_sizes['valid'] = batch_size
    else:
        batch_sizes = batch_size

    for set_type in ['train', 'valid']:
        dataset_dir = os.path.join(data_dir, set_type)

        if not os.path.exists(dataset_dir):
            raise RuntimeError("Cannot find dataset at - %s" % dataset_dir)
        
        dataset = CLICDataset(dataset_dir)
        
        data_loaders[set_type] = DataLoader(dataset, shuffle=True, batch_size=batch_sizes[set_type],
                                            num_workers=4, pin_memory=True)
    return data_loaders


if __name__ == "__main__":
    dataset_dir = "/home/orweiser/university/Digital-Processing-of-Single-and-Multi-Dimensional-Signals/data/valid"
    dataset = CLICDataset(dataset_dir)
    # img = dataset[5]
    print(len(dataset))