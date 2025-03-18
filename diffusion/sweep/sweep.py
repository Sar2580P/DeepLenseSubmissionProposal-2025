import torch
from utils.utils import read_yaml
import wandb
from diffusion.train_loop import DiffusionTrainLoop
from diffusion.dataloader import get_dataloaders
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from utils.callbacks import get_callbacks
import yaml
import time
from omegaconf import OmegaConf
from diffusion.architecture.modules import Unet
from diffusion.architecture.model import CustomGaussianDiffusion

sweep_config =read_yaml('diffusion/sweep/sweep_config.yaml')
sweep_config = OmegaConf.to_container(sweep_config, resolve=True)
sweep_id = wandb.sweep(sweep_config, project="DeepLense_Diffusion_Sweep")

# Define the training function
def train(config=None):
    with wandb.init(config=config):
        wandb_config = wandb.config

        original_config_filepath = "diffusion/sweep/sweep_full_config.yaml"

        # ✅ First, read the existing YAML config
        original_config = yaml.safe_load(open(original_config_filepath, 'r'))

        # ✅ Then, update and write it back
        original_config['sweep_config'].update(wandb_config)
        # print(original_config)

        with open(original_config_filepath, "w") as f:
            yaml.dump(original_config, f)

        time.sleep(0.05)  # Ensure the file is closed before reading it
        config = read_yaml(original_config_filepath)

        tr_config = config['train_config']
        data_config = config['data_config']

        unet = Unet(**config['Unet_params'])
        if tr_config['model_name'].lower()=='Vanilla_Gaussian_Diffusion'.lower():
            model = CustomGaussianDiffusion(unet, **config['GaussianDiffusion_params'])
        else:
            raise ValueError(f"Unknown model name: {tr_config['model_name']}")

        model_obj = DiffusionTrainLoop(model, config=config)

        tr_loader, val_loader, tst_loader = get_dataloaders(data_config)
        early_stop_callback, _, rich_progress_bar, rich_model_summary, lr_monitor = get_callbacks(config['callbacks_config'])

        #___________________________________________________________________________________________________________________

        torch.set_float32_matmul_precision('high')
        trainer = Trainer(callbacks=[early_stop_callback, rich_progress_bar, rich_model_summary, lr_monitor],
                        accelerator = tr_config['accelerator'] ,accumulate_grad_batches=tr_config['accumulate_grad_batches'] , 
                        max_epochs=tr_config['MAX_EPOCHS'])

        trainer.fit(model_obj, tr_loader, val_loader)
        trainer.test(model_obj, tst_loader)


# Run the sweep
wandb.agent(sweep_id, function=train, count = 50)