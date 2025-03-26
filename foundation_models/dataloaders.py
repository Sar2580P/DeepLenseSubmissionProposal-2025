from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torchvision import transforms
import torch 

class MAEDataset(Dataset):
    def __init__(self, data_csv_path:str , transform=None, objective="pretraining"):
        self.transform = transform or transforms.Compose([
           					 transforms.ToTensor()  # Ensures NumPy array is converted to a PyTorch tensor
        				])
        self.data_csv = pd.read_csv(data_csv_path)
        self.objective = objective   
    
    def __len__(self):
        return self.data_csv.shape[0]
    
    def __getitem__(self, idx):
        if self.objective=="pretraining":
          img_path = self.data_csv.iloc[idx]['img_path']
          label=None
        elif self.objective=="finetuned_classification":
          img_path = self.data_csv.iloc[idx]['image_path']
          label = self.data_csv.iloc[idx]['label']
          label = torch.tensor(label, dtype=torch.long)
       
        img = np.load(img_path, allow_pickle=True)
        if 'axion' in img_path:
           # img.shape : (2,) , img[0]-> (64,64) , img[1]-> np.float64
           img = img[0]
        img = np.expand_dims(img, axis=2).astype(np.float32)

        if self.transform:
           img = self.transform(img)

        return img, label if label is not None else img
    
def get_dataloaders(data_config:dict):
    """
    Utility function to get train, validation and test dataloaders for VAE training
    
    Args:
        data_config (dict): Configuration dictionary for data
    """
    
    tr_dataset = MAEDataset(data_csv_path=data_config['tr_path'], objective=data_config['objective']) 
    val_dataset= MAEDataset(data_csv_path=data_config['val_path'], objective=data_config['objective'])
    tst_dataset = MAEDataset(data_csv_path=data_config['tst_path'], objective=data_config['objective'])
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
