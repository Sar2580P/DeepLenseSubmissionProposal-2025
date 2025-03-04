from Multi_Class_Classification.train_loop import Classifier
from Multi_Class_Classification.dataset import get_dataloaders
from utils.utils import read_yaml
from utils.callbacks import get_callbacks
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import pickle
import torch
import os

config = read_yaml('Multi_Class_Classification/config.yaml')
tr_config = config['train_config']
data_config = config['data_config']

if tr_config['model_name'].lower()=='ImagenetModels'.lower():
    from Multi_Class_Classification.models import ImagenetModels
    model = ImagenetModels(**config['ImagenetModels_params'])
else:
    raise ValueError(f"Unknown model name: {tr_config['model_name']}")

model_obj = Classifier(model, config=config)

tr_loader, val_loader, tst_loader = get_dataloaders(data_config)
#___________________________________________________________________________________________________________________

RUN_NAME = f"{tr_config['model_name']}--data={data_config['dataset_type']}"
RESULT_DIR = os.path.join(tr_config['dir'], RUN_NAME)
os.makedirs(RESULT_DIR, exist_ok=True)

early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary, lr_monitor = get_callbacks(config['callbacks_config'])
filename = f"submodel={config[tr_config['model_name']+ '_params']['model_name']}"
checkpoint_callback.dirpath = os.path.join(RESULT_DIR, 'ckpts')
checkpoint_callback.filename = filename +'--'+ tr_config['ckpt_file_name']

run_name = RUN_NAME + f"--submodel={filename}"
wandb_logger = WandbLogger(project= "DeepLenseClassificationTask", name = run_name)
csv_logger = CSVLogger(RESULT_DIR+'/logs/'+ filename)
#___________________________________________________________________________________________________________________

torch.set_float32_matmul_precision('high')
trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary, lr_monitor],
                accelerator = tr_config['accelerator'] ,accumulate_grad_batches=tr_config['accumulate_grad_batches'] , 
                max_epochs=tr_config['MAX_EPOCHS'], logger=[wandb_logger, csv_logger])

trainer.fit(model_obj, tr_loader, val_loader)
trainer.test(model_obj, tst_loader)
#___________________________________________________________________________________________________________________

os.makedirs(os.path.join(RESULT_DIR, 'evaluations'), exist_ok=True)

with open(os.path.join(RESULT_DIR, 'evaluations', f"{filename}__predictions.pkl"), 'wb') as f:
    dict_ = {'y_hat': model_obj.tst_y_hat, 'y_true': model_obj.tst_y_true , 'config' : config}
    pickle.dump(dict_, f)
    
#___________________________________________________________________________________________________________________
