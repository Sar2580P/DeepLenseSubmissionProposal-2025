from utils.utils import read_yaml
from diffusion.architecture.modules import Unet
from diffusion.architecture.model import CustomGaussianDiffusion
from diffusion.metrics.fid import FIDEvaluation
from diffusion.dataloader import DiffusionDataset
from torch.utils.data import DataLoader
import torch
import numpy as np 

def initialise_diffusion(ckpt_path:str, map_to_cpu:bool=False):
  config = read_yaml('diffusion/config.yaml')
  tr_config = config['train_config']
  data_config = config['data_config']

  unet = Unet(**config['UNet_params'])
  if tr_config['model_name'].lower()=='Vanilla_Gaussian_Diffusion'.lower():
    model = CustomGaussianDiffusion(unet, **config['GaussianDiffusion_params'])
  else:
    raise ValueError(f"Unknown model name: {tr_config['model_name']}")
  
  if map_to_cpu:
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
  else : 
    checkpoint = torch.load(ckpt_path)
  #print(checkpoint['state_dict'].keys())
  model_weights = {k.replace("model.", "", 1) : v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
  try: 
    model.load_state_dict(model_weights)
    print("Loaded model weights successfully")
  except Exception as e:
    print("Failed to load model weights", e)
    
  return model

def get_dataloader(BATCH_SIZE:int , num_workers:int , csv_path:str):
  dataset = DiffusionDataset(data_csv_path=csv_path)
  total_samples = dataset.df.shape[0]
  loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=num_workers)

  return total_samples , loader


if __name__=="__main__":
  batch_size , num_workers = 16, 8
  channels = 1
  stats_dir = "results/Diffusion/Vanilla_Gaussian_Diffusion" 
  total_samples, loader = get_dataloader(batch_size , num_workers , 
                                        "data/dataframes/diffusion_dataset/test_df.csv")
  
  ckpt_path="results/Diffusion/Vanilla_Gaussian_Diffusion/ckpts/epoch=22 | val_MSE_loss=0.024.ckpt"
  diffusion_model = initialise_diffusion(ckpt_path, map_to_cpu= False)
 
  fid_eval = FIDEvaluation(batch_size,
        dl=iter(loader),
        sampler= diffusion_model,
        channels=channels,
        accelerator=None,
        stats_dir=stats_dir,
        device="cuda",
        num_fid_samples=total_samples)

  res = fid_eval.fid_score()
  # save the result in .npy
  print(f"FID score : {res}")
  np.save(f"{stats_dir}/final_fid_res", res)
  print("Completed calculation for FID score")
  
