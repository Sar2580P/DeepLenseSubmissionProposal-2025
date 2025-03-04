import torch
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
import torch.nn as nn
from typing import Dict
import wandb 
import numpy as np 
import matplotlib.pyplot as plt

class Classifier(pl.LightningModule):
  def __init__(self, model_obj:nn.Module, config:Dict):
    super().__init__()
    self.model_obj = model_obj
    self.layer_lr = model_obj.layer_lr
    self.tr_config = config['train_config']  
    self.lr_scheduler_config = config['lr_scheduler_params']

    self.tr_kappa = torchmetrics.CohenKappa(task = 'multiclass' , num_classes =self.tr_config['num_classes'], weights = 'quadratic')
    self.val_kappa = torchmetrics.CohenKappa(task = 'multiclass' , num_classes =self.tr_config['num_classes'], weights = 'quadratic')
    self.tst_kappa = torchmetrics.CohenKappa(task = 'multiclass' , num_classes =self.tr_config['num_classes'], weights = 'quadratic')

    self.tr_accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes =self.tr_config['num_classes'])
    self.val_accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes =self.tr_config['num_classes'])
    self.tst_accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes =self.tr_config['num_classes'])

    self.val_conf_mat = MulticlassConfusionMatrix(num_classes =self.tr_config['num_classes'])

    self.criterion = torch.nn.CrossEntropyLoss()
    self.tst_y_hat , self.tst_y_true = [], []
    self.save_hyperparameters(ignore=['model_obj', 'layer_lr', 'tst_y_hat', 'tst_y_true'])

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model_obj.forward(x)
    ce_loss = self.criterion(y_hat, y.long())      # compute CE loss

    self.tr_accuracy(y_hat, y)
    self.tr_kappa(y_hat, y)
    self.log("train_ce_loss", ce_loss,on_step = False ,on_epoch=True, prog_bar=True, logger=True)
    self.log("train_kappa", self.tr_kappa, on_step=False , on_epoch=True, prog_bar=True, logger=True)
    self.log("train_acc", self.tr_accuracy, on_step = False, on_epoch=True,prog_bar=True, logger=True)

    return ce_loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model_obj.forward(x)
    ce_loss = self.criterion(y_hat, y.long())      # compute CE loss
    
    self.val_accuracy(y_hat, y)
    self.val_kappa(y_hat, y)
    self.log("val_ce_loss", ce_loss, on_epoch=True, on_step=False,  prog_bar=True, logger=True)
    self.log("val_kappa", self.val_kappa,on_step = False, on_epoch=True, prog_bar=True, logger=True)
    self.log("val_acc", self.val_accuracy, on_step = False, on_epoch=True,prog_bar=True, logger=True)
    
    
    # logic for logging confusion matrix plot while training the model...
    if self.current_epoch%10==0:
        if batch_idx<4:
            y_pred = torch.argmax(y_hat, dim=1)
            self.val_conf_mat.update(y_pred, y)
        elif batch_idx==4:
            fig, ax = self.val_conf_mat.plot(labels=['no' , 'sphere' ,'vort'])
            ax.set_title("Sample Plot")
            # Extract image array from the figure
            fig.canvas.draw()
            image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # Log to Weights & Biases
            wandb.log({"plot": wandb.Image(image_array, caption=f"Confusion Matrix Plot (Validation- epoch={self.current_epoch})")})
            plt.close(fig)  # Close the figure to free memory
            self.val_conf_mat.reset()
            
    return ce_loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model_obj.forward(x)
    ce_loss = self.criterion(y_hat, y.long())      # compute CE loss

    self.tst_accuracy(y_hat, y)
    self.tst_kappa(y_hat, y)
    self.log("test_ce_loss", ce_loss,on_step = True,  on_epoch=True, prog_bar=True, logger=True)
    self.log("test_kappa", self.tst_kappa,on_step = True,  on_epoch=True, prog_bar=True, logger=True)
    self.log("test_acc", self.tst_accuracy, on_step = True ,on_epoch=True,prog_bar=True, logger=True)
    
    self.tst_y_hat.append(y_hat.detach().cpu().numpy())
    self.tst_y_true.append(y.detach().cpu().numpy())
    return ce_loss

  def configure_optimizers(self):
    optim =  torch.optim.Adam(self.layer_lr, lr =self.tr_config['lr'], weight_decay =self.tr_config['weight_decay'])   # https://pytorch.org/docs/stable/optim.html
    scheduler_params =self.lr_scheduler_config[f"{self.lr_scheduler_config['lr_scheduler_name']}_params"]


    if self.lr_scheduler_config['lr_scheduler_name'] == 'exponential_decay_lr_scheduler':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, **scheduler_params)

    elif self.lr_scheduler_config['lr_scheduler_name'] == 'cosine_decay_lr_scheduler':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, **scheduler_params)
    else : 
        raise ValueError(f"Unknown scheduler name: {self.lr_scheduler_config['lr_scheduler_name']}, please choose from ['exponential_decay_lr_scheduler', 'cosine_decay_lr_scheduler']")

    return [optim], [{'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'train_ce_loss', 'name':self.lr_scheduler_config['lr_scheduler_name']}]