import torch
import pytorch_lightning as pl
from foundation_models.architectures.mae import MAE
import torchmetrics

class MAETrainLoop(pl.LightningModule):
    def __init__(self, model, config:dict):
        super().__init__()
        self.model:MAE = model
        self.config = config

    def training_step(self, batch, batch_idx):
        x_samples = batch
        loss = self.model.forward(x_samples)
        self.log("train_MAE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_samples = batch
        loss = self.model.forward(x_samples)
        self.log("val_MAE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x_samples = batch
        loss = self.model.forward(x_samples)
        self.log("test_MAE_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

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

    self.criterion = torch.nn.CrossEntropyLoss()
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
                 {'params': base_params},
                 {'params': self.model.mlp_head.parameters(), 'lr': tr_config['lr'] * 5}
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

