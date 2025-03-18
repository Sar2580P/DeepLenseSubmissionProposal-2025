import torch
import pytorch_lightning as pl
from foundation_models.architectures.mae import MAE

class MAETrainLoop(pl.LightningModule):
    def __init__(self, model, config:dict):
        super().__init__()
        self.model:MAE = model
        self.config = config

    def training_step(self, batch, batch_idx):
        x_samples = batch
        loss = self.model.forward(x_samples)
        self.log("train_MAE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_samples = batch
        loss = self.model.forward(x_samples)
        self.log("val_MAE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x_samples = batch
        loss = self.model.forward(x_samples)
        self.log("test_MAE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True)

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

        return [optim], [{'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'train_MAE_loss',
                        'name': schdlr_config['scheduler_name']}]