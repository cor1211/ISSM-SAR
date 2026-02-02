"""
PyTorch Lightning DataModule for ISSM-SAR
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import random
import numpy as np
import torch

from src.SAR_Dataset import MultiTempSARDataset


def worker_init_fn(worker_id):
    """Ensure reproducibility in data loading workers"""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class SARDataModule(pl.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.cfg_data = config['data']
        self.root_dir = self.cfg_data['root']
        self.train_batch_size = self.cfg_data['train_batch_size']
        self.test_batch_size = self.cfg_data['test_batch_size']
        self.num_workers = self.cfg_data['num_workers']
        
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """Called on every GPU separately - handles data setup"""
        if stage == 'fit' or stage is None:
            self.train_dataset = MultiTempSARDataset(
                root_dir=self.root_dir, 
                phase='train', 
                transform=True
            )
            self.val_dataset = MultiTempSARDataset(
                root_dir=self.root_dir, 
                phase='val', 
                transform=False
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,  # Lightning handles DistributedSampler automatically
            num_workers=self.num_workers,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
