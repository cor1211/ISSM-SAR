import torch
from src import ISSM_SAR, SarDataset, lratio_loss, l1_loss, psnr_torch, ssim_torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import yaml
import argparse
from trainer import Trainer

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ =='__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='Train ISSM-SAR Model')
    parser.add_argument('--config_path', type=str, default= '/mnt/data1tb/vinh/ISSM-SAR/config/base_config.yaml', help='Path to the YAML config file')
    parser.add_argument('--checkpoint_path', type = str, default=None, help='Path to checkpoint to resume training')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)
    data_cfg = config['data']
    train_cfg = config['train']
    model_cfg = config['model']

    # Check cuda available and use
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU {torch.cuda.get_device_name("cuda")}')
    else:
        device = torch.device('cpu')
        print('No GPU, using CPU instead')

    # run_name
    if args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            print('Checkpoint path not found. End!')
            exit()
        try:
            run_name = args.checkpoint_path.split('/')[-2]
        except Exception as e:
            print('Invalid check point path')
            exit()
    else:
        run_name = f'exp_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    
    # Init SummaryWriter to log
    log_dir = os.path.join('runs', run_name)
    writer = SummaryWriter(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"TensorBoard logs will be saved to: {log_dir}")


    # Transform
    transform = Compose(transforms = [
        ToTensor(),
        Normalize(mean = [0.5],
                std=[0.5])
    ])

    # Train, test set
    train_set= SarDataset(root = data_cfg['root'], train=True, transform=transform)
    valid_set= SarDataset(root = data_cfg['root'], train=False, transform=transform)

    # Train, test loader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=data_cfg['train_batch_size'],
        shuffle=True,
        num_workers=data_cfg['num_workers'],
        drop_last=True
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=data_cfg['test_batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        drop_last=True
    )

    # Configure model
    model = ISSM_SAR(model_cfg).to(device)
    optimizer = Adam(model.parameters(), lr=train_cfg['lr'])

    # Training
    trainer = Trainer(model, optimizer, train_loader, valid_loader, device, config, writer, run_name, args.checkpoint_path)
    trainer.run()





        

        
