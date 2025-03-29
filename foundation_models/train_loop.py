import torch
import pytorch_lightning as pl
from foundation_models.architectures.mae import MAE
import torchmetrics
from foundation_models.architectures.vit import ViT
from foundation_models.architectures.super_resolution import SuperResolutionAE
import wandb
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure

class MAETrainLoop(pl.LightningModule):
    def __init__(self, model, config:dict):
        super().__init__()
        self.model:MAE = model
        self.config = config

    def training_step(self, batch, batch_idx):
        x_samples = batch
        loss = self.model.forward(x_samples)
        self.log("train_MSE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_samples = batch
        loss = self.model.forward(x_samples)
        self.log("val_MSE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x_samples = batch
        loss = self.model.forward(x_samples)
        self.log("test_MSE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

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


class ViT_Classifier_TrainLoop(pl.LightningModule):
  def __init__(self, model, config:dict):
    super().__init__()
    self.model: ViT = model
    self.config = config

    num_classes = self.config['ViT_params']['num_classes']
    self.tr_kappa = torchmetrics.CohenKappa(task = 'multiclass' , num_classes =num_classes, weights = 'quadratic')
    self.val_kappa = torchmetrics.CohenKappa(task = 'multiclass' , num_classes =num_classes, weights = 'quadratic')
    self.tst_kappa = torchmetrics.CohenKappa(task = 'multiclass' , num_classes =num_classes, weights = 'quadratic')

    self.tr_accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes =num_classes)
    self.val_accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes =num_classes)
    self.tst_accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes =num_classes)

    self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    self.tst_y_hat , self.tst_y_true = [], []
    self.save_hyperparameters(ignore=['model', 'layer_lr', 'tst_y_hat', 'tst_y_true'])

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model.forward(x)['classification_head_logits']
    ce_loss = self.criterion(y_hat, y.long())      # compute CE loss

    self.tr_accuracy(y_hat, y)
    self.tr_kappa(y_hat, y)
    self.log("train_ce_loss", ce_loss,on_step = False ,on_epoch=True, prog_bar=True, logger=True,  sync_dist=True)
    self.log("train_kappa", self.tr_kappa, on_step=False , on_epoch=True, prog_bar=True, logger=True,  sync_dist=True)
    self.log("train_acc", self.tr_accuracy, on_step = False, on_epoch=True,prog_bar=True, logger=True,  sync_dist=True)

    return ce_loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model.forward(x)['classification_head_logits']

    ce_loss = self.criterion(y_hat, y.long())      # compute CE loss
    
    self.val_accuracy(y_hat, y)
    self.val_kappa(y_hat, y)
    self.log("val_ce_loss", ce_loss, on_epoch=True, on_step=False,  prog_bar=True, logger=True,  sync_dist=True)
    self.log("val_kappa", self.val_kappa,on_step = False, on_epoch=True, prog_bar=True, logger=True,  sync_dist=True)
    self.log("val_acc", self.val_accuracy, on_step = False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
    return ce_loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model.forward(x)['classification_head_logits']
    ce_loss = self.criterion(y_hat, y.long())      # compute CE loss

    self.tst_accuracy(y_hat, y)
    self.tst_kappa(y_hat, y)
    self.log("test_ce_loss", ce_loss,on_step = True,  on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    self.log("test_kappa", self.tst_kappa,on_step = True,  on_epoch=True, prog_bar=True, logger=True,  sync_dist=True)
    self.log("test_acc", self.tst_accuracy, on_step = True ,on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
    
    self.tst_y_hat.append(y_hat.detach().cpu().numpy())
    self.tst_y_true.append(y.detach().cpu().numpy())
    return ce_loss

  def configure_optimizers(self):
    tr_config = self.config['train_config']
    lr_scheduler_config = self.config['scheduler_params']
   
    base_params = [p for n, p in self.model.named_parameters() if "mlp_head" not in n]
    layer_lr = [
                 {'params': base_params, 'lr': tr_config['lr']/10},
                 {'params': self.model.mlp_head.parameters() }
               ]

    optim =  torch.optim.Adam(layer_lr, lr = tr_config['lr'], weight_decay =tr_config['weight_decay'])   # https://pytorch.org/docs/stable/optim.html
    scheduler_params = lr_scheduler_config[f"{lr_scheduler_config['scheduler_name']}_params"]


    if lr_scheduler_config['scheduler_name'] == 'exponential_decay_lr_scheduler':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, **scheduler_params)

    elif lr_scheduler_config['scheduler_name'] == 'cosine_decay_lr_scheduler':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, **scheduler_params)
    else : 
        raise ValueError(f"Unknown scheduler name: {lr_scheduler_config['scheduler_name']}, please choose from ['exponential_decay_lr_scheduler', 'cosine_decay_lr_scheduler']")

    return [optim], [{'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'train_ce_loss', 'name':lr_scheduler_config['scheduler_name']}]



class SuperResAE_TrainLoop(pl.LightningModule):
    def __init__(self, model, config:dict):
      super().__init__()
      self.model:SuperResolutionAE = model
      self.config = config
      
      self.tr_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
      self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
      self.tst_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
      self.mse = torch.nn.MSELoss()
      self.tr_psnr = torchmetrics.PeakSignalNoiseRatio()
      self.val_psnr = torchmetrics.PeakSignalNoiseRatio()
      self.tst_psnr = torchmetrics.PeakSignalNoiseRatio()

    def training_step(self, batch, batch_idx):
      low_res, high_res = batch
      _, pred_high_res = self.model.forward(low_res)
      
      recon_loss = self.mse(pred_high_res, high_res)
      psnr = self.tr_psnr(pred_high_res, high_res)
      ssim = self.tr_ssim(pred_high_res, high_res)
      self.log("train_MSE_loss", recon_loss, on_step = False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log("train_PSNR", psnr, on_step = False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log("train_SSIM", ssim, on_step = False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      return recon_loss

    def validation_step(self, batch, batch_idx):
      low_res, high_res = batch
      restored_low_res_from_patches, pred_high_res = self.model.forward(low_res)
      
      recon_loss = self.mse(pred_high_res, high_res)
      psnr = self.val_psnr(pred_high_res, high_res)
      ssim = self.val_ssim(pred_high_res, high_res)
      self.log("val_MSE_loss", recon_loss, on_step = False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log("val_PSNR", psnr, on_step = False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log("val_SSIM", ssim, on_step = False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      
      # logging images for qualitative evaluation
      if self.config['train_config']['should_log_images'] and batch_idx==0 and  \
        self.current_epoch % self.config['train_config']['log_images_every_n_epochs'] == 0:
            
        # log the generated samples
        max_samples_to_log = min(5, high_res.shape[0])
        self.log_images({"low_res_images": low_res[:max_samples_to_log], "original_high_res_images": high_res[:max_samples_to_log],
                        "pred_high_res_images": pred_high_res[:max_samples_to_log], 
                        "restored_low_res_images": restored_low_res_from_patches[:max_samples_to_log]})

      return recon_loss
    
    def test_step(self, batch, batch_idx):
      low_res, high_res = batch
      _, pred_high_res = self.model.forward(low_res)
      
      recon_loss = self.mse(pred_high_res, high_res)
      psnr = self.tst_psnr(pred_high_res, high_res)
      ssim = self.tst_ssim(pred_high_res, high_res)
      self.log("test_SSIM", ssim, on_step = False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log("test_PSNR", psnr, on_step = False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log("test_MSE_loss", recon_loss, on_step = False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      return recon_loss
    
    def configure_optimizers(self):
      tr_config = self.config['train_config']
      schdlr_config = self.config['scheduler_params']
      
      encoder_param_ids = {id(p) for p in self.model.encoder.parameters()}
      base_params = [p for n, p in self.model.named_parameters() if "encoder" not in n and id(p) not in encoder_param_ids]
      layer_lr = [
                  {'params': base_params},
                  {'params': self.model.encoder.parameters(), 'lr': tr_config['lr'] /50.0}
                ]
    
      optim =  torch.optim.Adam(layer_lr, lr = tr_config['lr'], 
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
              Expected keys: "low_res_images", "original_high_res_images", "pred_high_res_images"
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
          ["Original (low_res)"] + [wandb.Image(processed_images["low_res_images"][i], mode="L") for i in range(num_samples)],
          ["Original (high_res)"] + [wandb.Image(processed_images["original_high_res_images"][i], mode="L") for i in range(num_samples)],
          ["Predicted (high res)"] + [wandb.Image(processed_images["pred_high_res_images"][i], mode="L") for i in range(num_samples)]
      ]

      table = wandb.Table(columns=columns, data=data)

      # ✅ Corrected logging method
      self.logger.experiment.log({
          f"visualization (epoch-> {self.current_epoch})": table
      })
