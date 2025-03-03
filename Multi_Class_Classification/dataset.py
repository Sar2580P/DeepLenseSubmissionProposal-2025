import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import pandas as pd
from typing import Dict, Tuple
import os,  glob
from sklearn.model_selection import train_test_split

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
        image = np.load(image_path).astype(np.float32)

        # Ensure image shape is (3, H, W)
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)  # Stack the same image across 3 channels
        elif image.shape[0] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {image.shape[0]} channels in {image_path}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def split_data(data_dir: str , split_ratio: float = 0.2) -> None:
    label0_files = glob.glob(os.path.join(data_dir, 'no', '*.npy'))
    label1_files = glob.glob(os.path.join(data_dir, 'sphere', '*.npy'))
    label2_files = glob.glob(os.path.join(data_dir, 'vort', '*.npy'))
    
    # Split data into train, val, test
    label0_train, label0_test = train_test_split(label0_files, test_size=split_ratio, random_state=42, shuffle=True)
    label1_train, label1_test = train_test_split(label1_files, test_size=split_ratio, random_state=42, shuffle=True)
    label2_train, label2_test = train_test_split(label2_files, test_size=split_ratio, random_state=42, shuffle=True)
    
    label0_train, label0_val = train_test_split(label0_train, test_size=split_ratio, random_state=42, shuffle=True)
    label1_train, label1_val = train_test_split(label1_train, test_size=split_ratio, random_state=42, shuffle=True)
    label2_train, label2_val = train_test_split(label2_train, test_size=split_ratio, random_state=42, shuffle=True)
    
    # Create dataframes
    train_df = pd.DataFrame({'image_path': label0_train + label1_train + label2_train, 
                             'label': [0]*len(label0_train) + [1]*len(label1_train) + [2]*len(label2_train)})
    val_df = pd.DataFrame({'image_path': label0_val + label1_val + label2_val,
                           'label': [0]*len(label0_val) + [1]*len(label1_val) + [2]*len(label2_val)})
    test_df = pd.DataFrame({'image_path': label0_test + label1_test + label2_test,
                            'label': [0]*len(label0_test) + [1]*len(label1_test) + [2]*len(label2_test)})
    
    # Save dataframes
    train_df.to_csv('data/dataset/train_df.csv', index=False)
    val_df.to_csv('data/dataset/val_df.csv', index=False)
    test_df.to_csv('data/dataset/test_df.csv', index=False)
    
    return
    
    
def get_dataloaders(data_config:Dict)->Tuple[DataLoader, DataLoader, DataLoader]:
    dataset_type:str = data_config['dataset_type']  
    train_df, val_df , test_df = (
        pd.read_csv(data_config['train_df_path']),
        pd.read_csv(data_config['val_df_path']),
        pd.read_csv(data_config['test_df_path'])
    )  
    if dataset_type.lower()=="imagenet_3channel":
        tr_dataset, val_dataset, test_dataset = (
                                                    Imagenet_3Channel_Dataset(train_df),
                                                    Imagenet_3Channel_Dataset(val_df),
                                                    Imagenet_3Channel_Dataset(test_df)
                                                )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}, please choose from ['imagenet_3channel']")
    
    tr_loader, val_loader, test_loader = (
        DataLoader(tr_dataset, **data_config['dataloader_params'], shuffle=True),
        DataLoader(val_dataset, **data_config['dataloader_params'], shuffle=False),
        DataLoader(test_dataset, **data_config['dataloader_params'], shuffle=False)
    )
    
    return tr_loader, val_loader, test_loader

if __name__ == '__main__':
    split_data('data/dataset/train', split_ratio=0.2)
    