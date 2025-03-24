import torch
import pytorch_lightning as pl
from diffusion.architecture.model import CustomGaussianDiffusion
import wandb
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger

class DiffusionTrainLoop(pl.LightningModule):
    def __init__(self, model, config:dict):
        super().__init__()
        self.model:CustomGaussianDiffusion = model
        self.config = config
        self.loss_lambda = config['train_config']['loss_lambda'] 

    def training_step(self, batch, batch_idx):
        x0_samples = batch
        loss, *_ = self.model.forward(x0_samples)
        loss = loss.mean()*self.loss_lambda
        self.log("train_MSE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x0_samples = batch
        if self.config['train_config']['should_log_images'] and batch_idx==0 and  \
            self.current_epoch % self.config['train_config']['log_images_every_n_epochs'] == 0:
                
            loss , x_start, x , model_output = self.model.forward(x0_samples, log_generation_results=True)
            # log the generated samples
            max_samples_to_log = min(5, x_start.shape[0])
            self.log_images({"x_start": x_start[:max_samples_to_log], "x": x[:max_samples_to_log],
                            "model_output": model_output[:max_samples_to_log]})
        else:
            loss, *_ = self.model.forward(x0_samples)
        loss = loss.mean()*self.loss_lambda
        self.log("val_MSE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True)

        return loss 

    def test_step(self, batch, batch_idx):
        x0_samples = batch
        loss, *_ = self.model.forward(x0_samples)
        loss = loss.mean()*self.loss_lambda
        self.log("test_MSE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        tr_config = self.config['train_config']
        schdlr_config = self.config['scheduler_params']
        optim =  torch.optim.Adam(self.model.parameters(), lr = tr_config['lr'], 
                                weight_decay = tr_config['weight_decay'])   
        scheduler_params = schdlr_config[f"{schdlr_config['scheduler_name']}_params"]

        if schdlr_config['scheduler_name'] == 'exponential_decay_lr_scheduler':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, **scheduler_params)

        elif schdlr_config['scheduler_name'] == 'cosine_decay_lr_scheduler':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, **scheduler_params)

        return [optim], [{'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'train_MSE_loss',
                        'name': schdlr_config['scheduler_name']}]
    
    def log_images(self, images_dict: dict):
        """
        Log grayscale images to Weights & Biases during training, organizing each image type in its own row.

        Args:
            images_dict (dict): Dictionary of torch tensors to log.
                Expected keys: "x_start", "x", "model_output"
        """

        if not hasattr(self, "logger") or not isinstance(self.logger, WandbLogger):
            print("WandbLogger not found. Skipping image logging.")
            return

        processed_images = {}

        for key, tensor in images_dict.items():
            images = tensor.detach().cpu().numpy()

            # Convert grayscale images from [B, 1, H, W] -> [B, H, W]
            if images.shape[1] == 1:
                images = np.squeeze(images, axis=1)

            # Min-max normalization per image
            min_vals = images.min(axis=(1, 2), keepdims=True)
            max_vals = images.max(axis=(1, 2), keepdims=True)
            images = (images - min_vals) / (max_vals - min_vals)

            processed_images[key] = images

        num_samples = list(processed_images.values())[0].shape[0]

        # Creating a W&B image grid
        columns = ["Type"] + [f"Sample-{i}" for i in range(num_samples)]
        data = [
            ["Original (x_start)"] + [wandb.Image(processed_images["x_start"][i], mode="L") for i in range(num_samples)],
            ["Noisy (x)"] + [wandb.Image(processed_images["x"][i], mode="L") for i in range(num_samples)],
            ["Model Output"] + [wandb.Image(processed_images["model_output"][i], mode="L") for i in range(num_samples)]
        ]

        table = wandb.Table(columns=columns, data=data)

        # ✅ Corrected logging method
        self.logger.experiment.log({
            f"visualization (epoch-> {self.current_epoch})": table
        })
