import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import pandas as pd
from typing import Dict, Tuple

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    
])


class Imagenet_3Channel_Dataset(Dataset):
    def __init__(self, data_df:pd.DataFrame, transform=None):
        """
        Args:
            data_df (pd.data_df): data_df containing 'image_path' and 'label' columns.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_df = data_df
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        image_path = self.data_df.iloc[idx]['image_path']
        label = self.data_df.iloc[idx]['label']

        # Load .npy image and convert to float32
        image = np.load(image_path).astype(np.float32).T
        # Ensure image shape is (H, W, 3)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)  # Stack the same image across 3 channels
        elif image.shape[-1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {image.shape[0]} channels in {image_path}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(data_config:Dict)->Tuple[DataLoader, DataLoader, DataLoader]:
    dataset_type:str = data_config['dataset_type']  
    train_df, val_df , test_df = (
        pd.read_csv(data_config['train_path']),
        pd.read_csv(data_config['val_path']),
        pd.read_csv(data_config['test_path'])
    )  
    if dataset_type.lower()=="imagenet_3channel":
        tr_dataset, val_dataset, test_dataset = (
                                                    Imagenet_3Channel_Dataset(train_df, transform=transform),
                                                    Imagenet_3Channel_Dataset(val_df),
                                                    Imagenet_3Channel_Dataset(test_df)
                                                )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}, please choose from ['imagenet_3channel']")

    tr_loader, val_loader, test_loader = (
        DataLoader(tr_dataset, batch_size=data_config['batch_size'], num_workers=data_config['num_workers'] ,shuffle=True),
        DataLoader(val_dataset, batch_size=data_config['batch_size'], num_workers=data_config['num_workers'], shuffle=False),
        DataLoader(test_dataset, batch_size=data_config['batch_size'], num_workers=data_config['num_workers'], shuffle=False)
    )
    
    return tr_loader, val_loader, test_loader
