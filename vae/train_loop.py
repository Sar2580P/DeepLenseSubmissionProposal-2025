import torch
import pytorch_lightning as pl
from vae.architectures.base_vae import BaseVAE

class VAETrainLoop(pl.LightningModule):
    def __init__(self, model, config:dict):
        super().__init__()
        self.model:BaseVAE = model
        self.config = config

    def training_step(self, batch, batch_idx):
        x0_samples = batch
        recons, input, mu, log_var = self.model.forward(x0_samples)
        loss = self.model.loss_function(recons, input, mu, log_var, M_N = 1)
        self.log("train_VAE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x0_samples = batch
        recons, input, mu, log_var = self.model.forward(x0_samples)
        loss = self.model.loss_function(recons, input, mu, log_var, M_N = 1)
        self.log("val_VAE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        x0_samples = batch
        recons, input, mu, log_var = self.model.forward(x0_samples)
        loss = self.model.loss_function(recons, input, mu, log_var, M_N = 1)
        self.log("test_VAE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True)

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

        return [optim], [{'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'train_VAE_loss',
                        'name': schdlr_config['scheduler_name']}]