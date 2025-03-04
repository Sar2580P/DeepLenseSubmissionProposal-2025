import torch
from utils.utils import read_yaml
import wandb
from Multi_Class_Classification.train_loop import Classifier
from Multi_Class_Classification.dataset import get_dataloaders
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from utils.callbacks import get_callbacks
import yaml
import time
from omegaconf import OmegaConf

sweep_config =read_yaml('Multi_Class_Classification/sweep/sweep_config.yaml')
sweep_config = OmegaConf.to_container(sweep_config, resolve=True)
sweep_id = wandb.sweep(sweep_config, project="DeepLenseClassificationSweep")

# Define the training function
def train(config=None):
    with wandb.init(config=config):
        wandb_config = wandb.config

        original_config_filepath = "Multi_Class_Classification/sweep/sweep_full_config.yaml"

        # ✅ First, read the existing YAML config
        original_config = yaml.safe_load(open(original_config_filepath, 'r'))

        # ✅ Then, update and write it back
        original_config['train_config'].update(wandb_config)
        # print(original_config)

        with open(original_config_filepath, "w") as f:
            yaml.dump(original_config, f)

        time.sleep(0.05)
        config = read_yaml(original_config_filepath)

        tr_config = config['train_config']
        data_config = config['data_config']

        if tr_config['model_name'].lower()=='ImagenetModels'.lower():
            from Multi_Class_Classification.models import ImagenetModels
            model = ImagenetModels(**config['ImagenetModels_params'])
        else:
            raise ValueError(f"Unknown model name: {tr_config['model_name']}")

        model_obj = Classifier(model, train_config=tr_config)

        tr_loader, val_loader, tst_loader = get_dataloaders(data_config)
        
        RUN_NAME = f"{tr_config['model_name']}--data={data_config['dataset_type']}"
        run_name = RUN_NAME + f"--submodel={config[tr_config['model_name']+ '_params']['model_name']}"
        wandb_logger = WandbLogger(project= "DeepLenseClassificationSweep", name = run_name)

        early_stop_callback, _, rich_progress_bar, rich_model_summary, lr_monitor = get_callbacks(config['callbacks_config'])
        torch.set_float32_matmul_precision('high')
        trainer = Trainer(callbacks=[early_stop_callback, rich_progress_bar, rich_model_summary, lr_monitor],
                accelerator = tr_config['accelerator'] ,accumulate_grad_batches=tr_config['accumulate_grad_batches'] , 
                max_epochs=tr_config['MAX_EPOCHS'], logger=[wandb_logger])

        trainer.fit(model_obj, tr_loader, val_loader)
        trainer.test(model_obj, tst_loader)

# Run the sweep
wandb.agent(sweep_id, function=train, count = 50)