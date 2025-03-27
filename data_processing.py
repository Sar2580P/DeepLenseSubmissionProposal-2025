import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def common_classification_dataset(data_dir: str, save_dir:str, split_ratio: float = 0.1) -> None:
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
                             'label': [0]*len(label0_train) + [1]*len(label1_train) + [2]*len(label2_train)}).sample(frac=1).reset_index(drop=True)
    val_df = pd.DataFrame({'image_path': label0_val + label1_val + label2_val,
                           'label': [0]*len(label0_val) + [1]*len(label1_val) + [2]*len(label2_val)}).sample(frac=1).reset_index(drop=True)
    test_df = pd.DataFrame({'image_path': label0_test + label1_test + label2_test,
                            'label': [0]*len(label0_test) + [1]*len(label1_test) + [2]*len(label2_test)}).sample(frac=1).reset_index(drop=True)
    
    # Save dataframes
    dir = os.path.join(save_dir, 'common_classification_dataset')
    os.makedirs(dir, exist_ok=True)
    train_df.to_csv(os.path.join(dir, 'train_df.csv'), index=False)
    val_df.to_csv(os.path.join(dir, 'val_df.csv'), index=False)
    test_df.to_csv(os.path.join(dir, 'test_df.csv'), index=False)

    return

def diffusion_dataset(save_dir:str):
    SOURCE_DIR = 'data/Samples'

    files = glob.glob(os.path.join(SOURCE_DIR, '*.npy'))
    train_files, test_files = train_test_split(files, test_size=0.1, random_state=42, shuffle=True)
    train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42, shuffle=True)
    
    # save files as 'img_path' col in dataframe
    train_df = pd.DataFrame({'img_path': train_files})
    val_df = pd.DataFrame({'img_path': val_files})
    test_df = pd.DataFrame({'img_path': test_files})
    
    # save dataframes
    dir = os.path.join(save_dir, 'diffusion_dataset')
    os.makedirs(dir, exist_ok=True)
    train_df.to_csv(os.path.join(dir, 'train_df.csv'), index=False)
    val_df.to_csv(os.path.join(dir, 'val_df.csv'), index=False)
    test_df.to_csv(os.path.join(dir, 'test_df.csv'), index=False)
    
    return

def foundation_models_pretraining_dataset(save_dir:str):
    
    SOURCE_DIR = 'data/Dataset/no_sub'
    files = glob.glob(os.path.join(SOURCE_DIR, '*.npy'))
    train_files, test_files = train_test_split(files, test_size=0.1, random_state=42, shuffle=True)
    train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42, shuffle=True)
    
    # save files as 'img_path' col in dataframe
    train_df = pd.DataFrame({'img_path': train_files})
    val_df = pd.DataFrame({'img_path': val_files})
    test_df = pd.DataFrame({'img_path': test_files})
    
    # save dataframes
    dir = os.path.join(save_dir, 'foundation_models_pretraining_dataset')
    os.makedirs(dir, exist_ok=True)
    train_df.to_csv(os.path.join(dir, 'train_df.csv'), index=False)
    val_df.to_csv(os.path.join(dir, 'val_df.csv'), index=False)
    test_df.to_csv(os.path.join(dir, 'test_df.csv'), index=False)
    
    return

def foundation_models_classification_dataset(save_dir:str):
    
    SOURCE_DIR = 'data/Dataset'
    
    no_sub_files = glob.glob(os.path.join(SOURCE_DIR, 'no_sub', '*.npy'))   
    cdm_files = glob.glob(os.path.join(SOURCE_DIR, 'cdm', '*.npy'))
    axion_files = glob.glob(os.path.join(SOURCE_DIR, 'axion', '*.npy'))
    
    # Split data into train, val, test
    no_sub_train, no_sub_test = train_test_split(no_sub_files, test_size=0.1, random_state=42, shuffle=True)
    cdm_train, cdm_test = train_test_split(cdm_files, test_size=0.1, random_state=42, shuffle=True)
    axion_train, axion_test = train_test_split(axion_files, test_size=0.1, random_state=42, shuffle=True)
    
    no_sub_train, no_sub_val = train_test_split(no_sub_train, test_size=0.1, random_state=42, shuffle=True)
    cdm_train, cdm_val = train_test_split(cdm_train, test_size=0.1, random_state=42, shuffle=True)  
    axion_train, axion_val = train_test_split(axion_train, test_size=0.1, random_state=42, shuffle=True)
    
    # Create dataframes
    train_df = pd.DataFrame({'image_path': no_sub_train + cdm_train + axion_train, 
                             'label': [0]*len(no_sub_train) + [1]*len(cdm_train) + [2]*len(axion_train)}).sample(frac=1).reset_index(drop=True)
    val_df = pd.DataFrame({'image_path': no_sub_val + cdm_val + axion_val,
                           'label': [0]*len(no_sub_val) + [1]*len(cdm_val) + [2]*len(axion_val)}).sample(frac=1).reset_index(drop=True)
    test_df = pd.DataFrame({'image_path': no_sub_test + cdm_test + axion_test,
                            'label': [0]*len(no_sub_test) + [1]*len(cdm_test) + [2]*len(axion_test)}).sample(frac=1).reset_index(drop=True)
    
    # save dataframes
    dir = os.path.join(save_dir, 'foundation_models_classification_dataset')
    os.makedirs(dir, exist_ok=True)
    train_df.to_csv(os.path.join(dir, 'train_df.csv'), index=False)
    val_df.to_csv(os.path.join(dir, 'val_df.csv'), index=False)    
    test_df.to_csv(os.path.join(dir, 'test_df.csv'), index=False)
    
    return

def foundation_models_SuperRes_dataset(save_dir:str):
    
    SOURCE_DIR = 'data/SR/Dataset'
    low_res_files = glob.glob(os.path.join(SOURCE_DIR, 'LR', '*.npy'))
    high_res_files = glob.glob(os.path.join(SOURCE_DIR, 'HR', '*.npy'))
    
    # Split data into train, val, test
    low_res_train, low_res_test = train_test_split(low_res_files, test_size=0.1, random_state=42, shuffle=True)
    high_res_train, high_res_test = train_test_split(high_res_files, test_size=0.1, random_state=42, shuffle=True)
    
    low_res_train, low_res_val = train_test_split(low_res_train, test_size=0.1, random_state=42, shuffle=True)
    high_res_train, high_res_val = train_test_split(high_res_train, test_size=0.1, random_state=42, shuffle=True)
    
    # Create dataframes
    train_df = pd.DataFrame({'low_res_img_path': low_res_train, 'high_res_img_path': high_res_train}).sample(frac=1).reset_index(drop=True)
    val_df = pd.DataFrame({'low_res_img_path': low_res_val, 'high_res_img_path': high_res_val}).sample(frac=1).reset_index(drop=True)
    test_df = pd.DataFrame({'low_res_img_path': low_res_test, 'high_res_img_path': high_res_test}).sample(frac=1).reset_index(drop=True)
    
    # save dataframes
    dir = os.path.join(save_dir, 'foundation_models_superres_dataset')
    os.makedirs(dir, exist_ok=True)
    train_df.to_csv(os.path.join(dir, 'train_df.csv'), index=False)
    val_df.to_csv(os.path.join(dir, 'val_df.csv'), index=False)
    test_df.to_csv(os.path.join(dir, 'test_df.csv'), index=False)
    
    return

if __name__=="__main__":
    save_dir = 'data/dataframes'
    common_classification_dataset('data/dataset/train', save_dir)
    diffusion_dataset(save_dir)
    foundation_models_pretraining_dataset(save_dir)
    foundation_models_classification_dataset(save_dir)
    foundation_models_SuperRes_dataset(save_dir)
    
    print('Datasets created successfully!')