import torch

ckpt_path = 'results/MAE/MAE_ViT_pretraining/ckpts/epoch=119 | val_MAE_loss=0.010.ckpt'

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
print(ckpt['state_dict'].keys())

