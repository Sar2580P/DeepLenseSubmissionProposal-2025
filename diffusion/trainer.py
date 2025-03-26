from diffusion.train_loop import DiffusionTrainLoop
from diffusion.architecture.model import CustomGaussianDiffusion
from diffusion.architecture.modules import Unet
from diffusion.dataloader import get_dataloaders
from utils.utils import read_yaml
from utils.callbacks import get_callbacks
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import torch
import os

config = read_yaml('diffusion/config.yaml')
tr_config = config['train_config']
data_config = config['data_config']

unet = Unet(**config['UNet_params'])
if tr_config['model_name'].lower()=='Vanilla_Gaussian_Diffusion'.lower():
    model = CustomGaussianDiffusion(unet, **config['GaussianDiffusion_params'])
else:
    raise ValueError(f"Unknown model name: {tr_config['model_name']}")

model_obj = DiffusionTrainLoop(model, config=config)

tr_loader, val_loader, tst_loader = get_dataloaders(data_config)
#___________________________________________________________________________________________________________________

RUN_NAME = f"{tr_config['model_name']}"
RESULT_DIR = os.path.join(tr_config['dir'], RUN_NAME)
os.makedirs(RESULT_DIR, exist_ok=True)

early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary, lr_monitor = get_callbacks(config['callbacks_config'])
checkpoint_callback.dirpath = os.path.join(RESULT_DIR, 'ckpts')
checkpoint_callback.filename = tr_config['ckpt_file_name']

wandb_logger = WandbLogger(project= "DeepLense_Diffusion_Task", name = RUN_NAME)
csv_logger = CSVLogger(RESULT_DIR+'/logs/'+ tr_config['ckpt_file_name'])
#___________________________________________________________________________________________________________________

torch.set_float32_matmul_precision('high')
trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary, lr_monitor],
                accelerator = tr_config['accelerator'] ,accumulate_grad_batches=tr_config['accumulate_grad_batches'] , 
                max_epochs=tr_config['MAX_EPOCHS'], logger=[wandb_logger, csv_logger], devices=2, num_nodes=1, strategy="ddp")

trainer.fit(model_obj, tr_loader, val_loader)
trainer.test(model_obj, tst_loader)
