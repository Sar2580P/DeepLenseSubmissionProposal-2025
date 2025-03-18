import torch
from utils.utils import read_yaml
import wandb
from vae.train_loop import VAETrainLoop
from vae.dataloader import get_dataloaders
from pytorch_lightning import Trainer
from utils.callbacks import get_callbacks
import yaml
import time
from omegaconf import OmegaConf

sweep_config =read_yaml('vae/sweep/sweep_config.yaml')
sweep_config = OmegaConf.to_container(sweep_config, resolve=True)
sweep_id = wandb.sweep(sweep_config, project="DeepLense_VAE_Analysis_Sweep")

# Define the training function
def train(config=None):
    with wandb.init(config=config):
        wandb_config = wandb.config

        original_config_filepath = "vae/sweep/sweep_full_config.yaml"

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

        if tr_config['model_name'].lower()=='VanillaVAE'.lower():
            from vae.architectures.vanilla_vae import VanillaVAE
            model = VanillaVAE(**config['VanillaVAE_params'])
        elif tr_config['model_name'].lower()=='BetaVAE'.lower():
            from vae.architectures.beta_vae import BetaVAE
            model = BetaVAE(**config['BetaVAE_params'])
        else:
            raise ValueError(f"Unknown model name: {tr_config['model_name']}, please choose from VanillaVAE or BetaVAE")

        model_obj = VAETrainLoop(model, config=config)

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