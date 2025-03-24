from foundation_models.train_loop import MAETrainLoop
from foundation_models.dataloaders import get_dataloaders
from utils.utils import read_yaml
from utils.callbacks import get_callbacks
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import torch
import os
from pytorch_lightning.strategies import DDPStrategy


config = read_yaml('foundation_models/config.yaml')
tr_config = config['train_config']
data_config = config['data_config']

if tr_config['model_name'].lower()=='MAE_ViT'.lower():
    from foundation_models.architectures.mae import MAE
    from foundation_models.architectures.vit import ViT
    encoder = ViT(**config['ViT_params'])
    mae_model = MAE(encoder=encoder, **config['MAE_params'])
else:
    raise ValueError(f"Unknown model name: {tr_config['model_name']}, please choose from ['MAE_ViT']")

model_obj = MAETrainLoop(mae_model, config=config)

tr_loader, val_loader, tst_loader = get_dataloaders(data_config)
#___________________________________________________________________________________________________________________

RUN_NAME = f"{tr_config['model_name']}_{tr_config['dataset_objective']}"
RESULT_DIR = os.path.join(tr_config['dir'], RUN_NAME)
os.makedirs(RESULT_DIR, exist_ok=True)

early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary, lr_monitor = get_callbacks(config['callbacks_config'])
checkpoint_callback.dirpath = os.path.join(RESULT_DIR, 'ckpts')
checkpoint_callback.filename = tr_config['ckpt_file_name']

wandb_logger = WandbLogger(project= "DeepLense_FoundationModels_Analysis", name = RUN_NAME)
csv_logger = CSVLogger(RESULT_DIR+'/logs/'+ tr_config['ckpt_file_name'])
#___________________________________________________________________________________________________________________

torch.set_float32_matmul_precision('high')
trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary, lr_monitor],
                accelerator = tr_config['accelerator'] ,accumulate_grad_batches=tr_config['accumulate_grad_batches'] , 
                strategy=DDPStrategy(find_unused_parameters=True), max_epochs=tr_config['MAX_EPOCHS'], logger=[wandb_logger, csv_logger])

trainer.fit(model_obj, tr_loader, val_loader)
trainer.test(model_obj, tst_loader)
