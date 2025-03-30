from foundation_models.train_loop import MAETrainLoop, ViT_Classifier_TrainLoop, SuperResAE_TrainLoop
from foundation_models.dataloaders import get_dataloaders
from utils.utils import read_yaml
from utils.callbacks import get_callbacks
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import torch
import os
from pytorch_lightning.strategies import DDPStrategy
from foundation_models.architectures.vit import ViT
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Trainer with configurable YAML")
    parser.add_argument("--config", type=str, required=True,
                        help="Name of the config file (e.g., super_res_config.yaml, config2.yaml, config3.yaml)")
    return parser.parse_args()

args = parse_args()
if args.config=="pretraining":
  config = read_yaml('foundation_models/configs/pre_training_config.yaml')
elif args.config=="Task-4A":
  config = read_yaml('foundation_models/configs/classification_finetuning_config.yaml')
elif args.config=="Task-4B":
  config = read_yaml('foundation_models/configs/super_res_config.yaml')
else :
  raise ValueError(f'Provide right config, one of [pretraining, Task-4A, Task-4B], but provided {args.config}')

#__________________________________________________________________________________________________________________________

tr_config = config['train_config']
data_config = config['data_config']
device="cuda" if torch.cuda.is_available() else "cpu"

if tr_config['model_name'].lower()=='MAE_ViT'.lower():
    from foundation_models.architectures.mae import MAE
    encoder = ViT(**config['ViT_params'])
    model = MAE(encoder=encoder, **config['MAE_params'])
    model_obj = MAETrainLoop(model, config=config)

elif tr_config["model_name"].lower()=='Task-4A_ViT'.lower():
    model = ViT(**config['ViT_params'])  
    checkpoint = torch.load(tr_config['pretrained_mae_ckpt_path'], map_location=torch.device(device))
    model_weight = {k.replace("model.encoder.", "", 1):v for k, v in checkpoint["state_dict"].items() if k.startswith("model.encoder.")}
    try: 
      model.load_state_dict(model_weight)
      print("Successfully loaded the model weights for classification")
    except Exception as e:
      print(f"Failed to load the checkpoint for classification... initialising model randomly")
    model_obj = ViT_Classifier_TrainLoop(model=model , config=config)

elif tr_config["model_name"].lower()=="Task-4B_SuperRes".lower():
    from foundation_models.architectures.super_resolution import SuperResolutionAE

    encoder = ViT(**config['ViT_params'])

    checkpoint = torch.load(tr_config['pretrained_mae_ckpt_path'], map_location=torch.device(device))
    model_weight = {k.replace("model.encoder.", "", 1):v for k, v in checkpoint["state_dict"].items() if k.startswith("model.encoder.")}
    try:
      encoder.load_state_dict(model_weight)
      print("Successfully loaded the encoder weights for Super Resolution")
    except Exception as e:
      print(f"Failed to load the checkpoint for super resolution... initialising model randomly")
      print(e)
    

    model = SuperResolutionAE(encoder=encoder, **config["SuperRes_params"])
    model_obj = SuperResAE_TrainLoop(model, config)
else:
    raise ValueError(f"Unknown model name: {tr_config['model_name']}")

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
                accelerator = tr_config['accelerator'] ,accumulate_grad_batches=tr_config['accumulate_grad_batches'], devices=2 , num_nodes=1, 
                strategy=DDPStrategy(find_unused_parameters=tr_config['find_unused_parameters']), max_epochs=tr_config['MAX_EPOCHS'], 
                logger=[wandb_logger, csv_logger])

trainer.fit(model_obj, tr_loader, val_loader)
trainer.test(model_obj, tst_loader)

#____________________________________________________________________________________________________________________
# for Task-4A, saving the test predictions for ROC curve valuations
if tr_config["model_name"].lower()=='Task-4A_ViT_Classifier'.lower():
  import pickle
  with open(os.path.join(RESULT_DIR,  f"predictions.pkl"), 'wb') as f:
    dict_ = {'y_hat': model_obj.tst_y_hat, 'y_true': model_obj.tst_y_true , 'config' : config}
    pickle.dump(dict_, f)

