
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        item['mask'] = np.array(item['mask'])

        if self.transform:
            transformed = self.transform(image=item['target'], mask=item['mask'])
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(item['target'])
            mask = torch.from_numpy(item['mask'])

        mask = mask[:,:,0]

        return image, mask
    

