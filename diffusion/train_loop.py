import torch
import pytorch_lightning as pl
from diffusion.architecture.model import CustomGaussianDiffusion
import wandb
import numpy as np

class DiffusionTrainLoop(pl.LightningModule):
    def __init__(self, model, config:dict):
        super().__init__()
        self.model:CustomGaussianDiffusion = model
        self.config = config

    def training_step(self, batch, batch_idx):
        x0_samples = batch
        loss, *_ = self.model.forward(x0_samples)
        loss = loss.mean()
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
        loss = loss.mean()
        self.log("val_MSE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True)

        return loss 

    def test_step(self, batch, batch_idx):
        x0_samples = batch
        loss, *_ = self.model.forward(x0_samples)
        loss = loss.mean()
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
        
        # Convert tensors to numpy and prepare for visualization
        processed_images = {}
        
        for key, tensor in images_dict.items():
            # Ensure tensor is detached from computational graph and on CPU
            images = tensor.detach().cpu()
            
            # Convert to numpy array
            images = images.numpy()
            
            # For grayscale images...
            if images.shape[1] == 1: images = np.squeeze(images, axis=1)  # [B, C=1, H, W] -> [B, H, W]
            
            # apply min-max normalisation for all images in a batch separately
            min = images.min(axis=(1, 2), keepdims=True)
            max = images.max(axis=(1, 2), keepdims=True)
            images = (images - min) / (max - min)
            processed_images[key] = images
        
        # Create a grid with each row representing a different image type
        num_samples = list(processed_images.values())[0].shape[0]
        
        # Create a wandb image grid using columns
        columns = [f"Sample-{i}" for i in range(num_samples)]
        
        # Log to wandb using a table format
        wandb.log({
            f"visualization (epoch-> {self.current_epoch})": wandb.Table(
                columns=columns,
                data=[
                    [wandb.Image(processed_images["x_start"][i], mode="L") for i in range(num_samples)],
                    [wandb.Image(processed_images["x"][i], mode="L") for i in range(num_samples)],
                    [wandb.Image(processed_images["model_output"][i], mode="L") for i in range(num_samples)]
                ],
                rows=["Original (x_start)", "Noisy (x)", "Model Output"]
            )
        })
