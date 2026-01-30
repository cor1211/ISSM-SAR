import torch
from src import ISSM_SAR, MultiTempSARDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import yaml
import argparse
from trainer_2 import Trainer
from pathlib import Path
import sys
import random
import numpy as np


def load_config(config_path: str):
    with open(Path(config_path), 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_device(device_name: str):
    if not device_name:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            device = torch.device(device_name)
        except Exception as e:
            print(f'Explain error:\nNow device was set: {device_name} in base_config. {e}')
            sys.exit(1)
    print(f'Device: {device}')
    return device


def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


if __name__ =='__main__':
    #------------Argument parser------------
    parser = argparse.ArgumentParser(description='Train ISSM-SAR Model')
    parser.add_argument('--config_path', type=str, default= '/mnt/data1tb/vinh/ISSM-SAR/config/base_config.yaml', help='Path to the YAML config file')
    args = parser.parse_args()


    #-----------------Load config--------------
    config = load_config(args.config_path)
    data_cfg = config['data']
    train_cfg = config['train']
    model_cfg = config['model']
    # Train config
    ckpt_path = train_cfg['resume_path']
    kaggle_mode = train_cfg['kaggle']
    device_name = train_cfg['device']
    seed = train_cfg['seed']

    
    #---------Set seed-------------
    set_seed(seed)


    #----------Load device----------
    device = load_device(device_name)


    #-----------Load checkpoint-----------
    if ckpt_path:
        ckpt_path = Path(ckpt_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint path {ckpt_path} not found. End!')
        
        try:
            run_name = ckpt_path.parent.name
        except Exception as e:
            print(f'Can not extract run_name at {ckpt_path}: {e}')
            sys.exit(1)
    else:
        run_name = f'exp_{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    #------------Init SummaryWriter to log------------
    log_dir = os.path.join('runs', run_name)
    writer = SummaryWriter(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"TensorBoard logs will be saved to: {log_dir}")


    #-----------Dataset & Dataloader-----------
        
        #-----------Dataset------------
    train_set = MultiTempSARDataset(root_dir=data_cfg['root'], phase='train', transform=True)
    valid_set = MultiTempSARDataset(root_dir=data_cfg['root'], phase='val', transform=False)


        #-----------Dataloader---------
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=data_cfg['train_batch_size'],
        shuffle=True,
        num_workers=data_cfg['num_workers'],
        drop_last=True,
        worker_init_fn=worker_init_fn
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=data_cfg['test_batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        drop_last=False,
        worker_init_fn=worker_init_fn
    )


    #-------------Configure model--------------
    model = ISSM_SAR(model_cfg).to(device)
    optimizer = Adam(model.parameters(), lr=train_cfg['lr'], betas=tuple(train_cfg['betas']))


    #-------------Training--------------
    trainer = Trainer(model, optimizer, train_loader, valid_loader, device, config, writer, run_name, ckpt_path)
    trainer.run()





        

        
