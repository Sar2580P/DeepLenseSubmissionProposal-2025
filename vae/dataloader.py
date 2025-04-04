from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class VAEDataset(Dataset):
    def __init__(self, data_csv_path:str , transform=None):
        self.transform = transform
        self.data_csv = pd.read_csv(data_csv_path)
            
    
    def __len__(self):
        return self.data_csv.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.data_csv.iloc[idx]['image_path']
        img = np.load(img_path).astype(np.float32)

        if self.transform:
            img = self.transform(img)
        
        return img
    
def get_dataloaders(data_config:dict):
    """
    Utility function to get train, validation and test dataloaders for VAE training
    
    Args:
        data_config (dict): Configuration dictionary for data
    """
    
    tr_dataset = VAEDataset(data_csv_path=data_config['tr_path']) 
    val_dataset= VAEDataset(data_csv_path=data_config['val_path'])
    tst_dataset = VAEDataset(data_csv_path=data_config['tst_path'])
    train_loader = DataLoader(
                        tr_dataset, 
                        batch_size=data_config['batch_size'],
                        shuffle=True, 
                        num_workers=data_config['num_workers'],
                    )
    
    val_loader = DataLoader(
                    val_dataset, 
                    batch_size=data_config['batch_size'],
                    shuffle=False, 
                    num_workers=data_config['num_workers'],
                )
    
    test_loader = DataLoader(
                    tst_dataset, 
                    batch_size=data_config['batch_size'],
                    shuffle=False, 
                    num_workers=data_config['num_workers'],
                )
    return train_loader, val_loader, test_loader