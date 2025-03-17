from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from functools import partial

class DiffusionDataset(Dataset):
    def __init__(self, data_dir:str, total_samples:int, 
                 mode:str='train' , split_ratio:float=0.75,  transform=None):
        """
        Custom PyTorch Dataset
        
        Args:
            data_path (str): Path to the data file
            transform (callable, optional): Optional transform to be applied to samples
        """
        self.total_samples = total_samples
        self.mode = mode
        self.transform = transform
        self.data_dir = data_dir
        
        self.data_samples_ct , self.data = self.get_data_splits(split_ratio)
        
    def get_data_splits(self, split_ratio:float):
        """
        Utility function to split the data into train and test
        
        Args:
            split_ratio (float): Ratio to split the data
        """
        np.random.seed(42)
        train_samples = int(self.total_samples*split_ratio)
        val_samples = (self.total_samples - train_samples)//2
        test_samples = self.total_samples - train_samples - val_samples
        
        train_data = np.random.choice(self.total_samples, train_samples, replace=False)
        rem_data = np.setdiff1d(np.arange(self.total_samples), train_data)
        val_data = np.random.choice(rem_data, val_samples, replace=False)
        test_data = np.setdiff1d(rem_data, val_data)
        
        if self.mode == 'train':
            return train_samples , train_data
        elif self.mode == 'val':
            return val_samples, val_data
        elif self.mode == 'test':
            return test_samples, test_data
        else:
            raise ValueError("Invalid mode for DiffusionDataset... Please choose from ['train', 'val', 'test']")
        
    def __len__(self):
        return self.data_samples_ct
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, f'sample{self.data[idx]}.npy')
        img = np.load(img_path)
        
        if self.transform:
            img = self.transform(img)
        
        return img
    
    
def get_dataloaders(data_config:dict):
    """
    Utility function to get train and test dataloaders
    
    Args:
        data_config (dict): Configuration dictionary for data
    """
    dataset = partial(DiffusionDataset, data_dir=data_config['data_dir'], 
                      total_samples=data_config['total_samples'], split_ratio=data_config['split_ratio'])
    
    train_dataset, val_dataset,  test_dataset = dataset(mode="train") , dataset(mode="val") , dataset(mode="test")
    
    train_loader = DataLoader(train_dataset, batch_size=data_config['BATCH_SIZE'],
                              shuffle=True, num_workers=data_config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=data_config['BATCH_SIZE'],
                            shuffle=False, num_workers=data_config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=data_config['BATCH_SIZE'],
                             shuffle=False, num_workers=data_config['num_workers'])
    
    return train_loader, val_loader,  test_loader