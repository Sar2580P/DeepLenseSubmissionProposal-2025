from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from functools import partial
import pandas as pd
import torchvision.transforms as T

transform = T.Compose([
    T.ToPILImage(),          # Convert numpy array (150x150) to PIL Image.
    T.Pad(padding=1),        # Pad 1 pixel on all sides to get 152x152.
    T.ToTensor()             # Convert back to tensor.
])


class DiffusionDataset(Dataset):
    def __init__(self, data_csv_path:str, transform=transform):
        """
        Custom PyTorch Dataset
        
        Args:
            data_path (str): Path to the data file
            transform (callable, optional): Optional transform to be applied to samples
        """
        self.transform = transform     
        self.df = pd.read_csv(data_csv_path)   
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['img_path']
        img = np.load(img_path).transpose(1, 2, 0)
        if self.transform:
            img = self.transform(img)
        
        return img
    
    
def get_dataloaders(data_config:dict):
    """
    Utility function to get train and test dataloaders
    
    Args:
        data_config (dict): Configuration dictionary for data
    """
    
    train_dataset = DiffusionDataset(data_csv_path=data_config['train_path'])
    val_dataset = DiffusionDataset(data_csv_path=data_config['val_path'])
    test_dataset = DiffusionDataset(data_csv_path=data_config['test_path'])
    train_loader = DataLoader(train_dataset, batch_size=data_config['BATCH_SIZE'],
                              shuffle=True, num_workers=data_config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=data_config['BATCH_SIZE'],
                            shuffle=False, num_workers=data_config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=data_config['BATCH_SIZE'],
                             shuffle=False, num_workers=data_config['num_workers'])
    
    return train_loader, val_loader,  test_loader