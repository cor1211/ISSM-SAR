"""
PyTorch Lightning Training Script for ISSM-SAR with Multi-GPU Support

Usage:
    # Single GPU
    python train_lightning.py --config_path config/base_config.yaml --devices 1

    # Multi-GPU (2 GPUs with DDP)
    python train_lightning.py --config_path config/base_config.yaml --devices 2
"""
import argparse
import yaml
import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from datetime import datetime

from src.lightning_module import ISSM_SAR_Lightning
from src.data_module import SARDataModule


def load_config(config_path: str) -> dict:
    with open(Path(config_path), 'r') as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Train ISSM-SAR with PyTorch Lightning')
    parser.add_argument('--config_path', type=str, 
                        default='/mnt/data1tb/vinh/ISSM-SAR/config/base_config.yaml',
                        help='Path to the YAML config file')
    parser.add_argument('--devices', type=int, default=2,
                        help='Number of GPUs to use (default: 2)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (overrides config)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)
    train_cfg = config['train']
    
    # Set seed
    set_seed(train_cfg.get('seed', 42))

    # Determine run name
    resume_path = args.resume or train_cfg.get('resume_path')
    if resume_path and os.path.exists(resume_path):
        # Extract run name from checkpoint path
        run_name = Path(resume_path).parent.name
        print(f"Resuming from checkpoint: {resume_path}")
    else:
        run_name = f'lightning_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        resume_path = None
        print(f"Starting new run: {run_name}")

    # Setup logger
    logger = TensorBoardLogger(
        save_dir='runs',
        name=run_name,
        default_hp_metric=False
    )

    # Setup callbacks
    checkpoint_dir = os.path.join('checkpoints', run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        # Save best model by SSIM
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best-{epoch:02d}-{step:06d}',
            monitor='val/ssim',
            mode='max',
            save_top_k=1,
            save_last=True,
            verbose=True
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval='step'),
        # Rich progress bar
        RichProgressBar(refresh_rate=10)
    ]

    # Determine strategy based on devices
    if args.devices > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(
            find_unused_parameters=False,  # Set True if model has unused params
            static_graph=True,  # Optimization for static computation graphs
        )
        print(f"Using DDP strategy with {args.devices} GPUs")
    else:
        strategy = 'auto'
        print(f"Using single GPU")

    # Create trainer
    trainer = pl.Trainer(
        # Hardware
        accelerator='gpu',
        devices=args.devices,
        strategy=strategy,
        
        # Training config
        max_epochs=train_cfg['total_epochs'],
        precision='16-mixed' if train_cfg.get('use_amp', True) else 32,
        gradient_clip_val=train_cfg.get('grad_clip', 0.0),
        
        # Validation - val_check_interval as int means every N batches
        val_check_interval=int(train_cfg['val_step']),
        num_sanity_val_steps=2,  # Quick sanity check at start
        
        # Logging
        logger=logger,
        log_every_n_steps=10,
        
        # Callbacks
        callbacks=callbacks,
        
        # Performance
        enable_model_summary=True,
        enable_progress_bar=True,
        
        # Reproducibility
        deterministic=False,  # Set True for full reproducibility (slower)
    )

    # Create model and datamodule
    model = ISSM_SAR_Lightning(config)
    datamodule = SARDataModule(config)

    # Train
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=resume_path
    )

    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best model saved at: {checkpoint_dir}")
    print(f"TensorBoard logs at: runs/{run_name}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
