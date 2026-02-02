"""
PyTorch Lightning Module for ISSM-SAR Multi-GPU Training
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid

from src.model import ISSM_SAR
from src.loss import lratio_loss


def denorm(x, mean=0.5, std=0.5):
    """Denormalize from [-1, 1] to [0, 1]"""
    return (x * std + mean).clamp(0, 1)


class ISSM_SAR_Lightning(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Config
        self.cfg_train = config['train']
        self.cfg_model = config['model']
        
        # Loss coefficients
        self.theta_1 = self.cfg_train['theta_1']
        self.theta_2 = self.cfg_train['theta_2']
        self.theta_3 = self.cfg_train.get('theta_3', 0.0)
        self.component_losses = self.cfg_train['component_losses']
        self.val_step = self.cfg_train['val_step']
        
        # Model
        self.model = ISSM_SAR(self.cfg_model)
        
        # Loss function
        self.criterion_L1 = nn.L1Loss()
        
        # Metrics (automatically synced across GPUs by torchmetrics)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        
        # Best metric tracking
        self.best_ssim = 0.0
        
        # Accumulate losses for logging every val_step
        self.accumulated_l1 = 0.0
        self.accumulated_ratio = 0.0
        self.accumulate_count = 0

    def forward(self, s1t1, s1t2):
        return self.model(s1t1, s1t2)

    def training_step(self, batch, batch_idx):
        # Get data from dict
        s1t1 = batch['T1']
        s1t2 = batch['T2']
        hr = batch['HR']
        
        # Forward
        s1sr_up, s1sr_down = self(s1t1, s1t2)
        
        # Compute losses (use same dtype as model for AMP compatibility)
        total_ratio_loss = torch.tensor(0.0, device=self.device, dtype=s1t1.dtype)
        total_l1_loss = torch.tensor(0.0, device=self.device, dtype=s1t1.dtype)
        
        # Component losses: calculate for intermediate SR outputs
        if self.component_losses:
            for idx in range(self.cfg_model['num_ifs']):
                if self.theta_1 != 0:
                    total_ratio_loss = total_ratio_loss + lratio_loss(s1sr_up[idx], hr) + lratio_loss(s1sr_down[idx], hr)
                if self.theta_2 != 0:
                    total_l1_loss = total_l1_loss + self.criterion_L1(s1sr_up[idx], hr) + self.criterion_L1(s1sr_down[idx], hr)
        
        # Final output loss
        if self.theta_1 != 0.0:
            total_ratio_loss = total_ratio_loss + lratio_loss(s1sr_up[-1], hr) + lratio_loss(s1sr_down[-1], hr)
        if self.theta_2 != 0.0:
            total_l1_loss = total_l1_loss + self.criterion_L1(s1sr_up[-1], hr) + self.criterion_L1(s1sr_down[-1], hr)
        
        loss = self.theta_1 * total_ratio_loss + self.theta_2 * total_l1_loss
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            self.log('train/nan_detected', 1.0, prog_bar=True)
            return None  # Skip this batch
        
        # Accumulate for periodic logging
        self.accumulated_l1 += total_l1_loss.item()
        self.accumulated_ratio += total_ratio_loss.item()
        self.accumulate_count += 1
        
        # Log every step (sync_dist=True for multi-GPU)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/l1_loss', total_l1_loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/ratio_loss', total_ratio_loss, on_step=True, on_epoch=False, sync_dist=True)
        
        # Log average every val_step
        if self.global_step > 0 and self.global_step % self.val_step == 0:
            avg_l1 = self.accumulated_l1 / max(self.accumulate_count, 1)
            avg_ratio = self.accumulated_ratio / max(self.accumulate_count, 1)
            self.log('train/avg_l1_loss', avg_l1, on_step=True, on_epoch=False)
            self.log('train/avg_ratio_loss', avg_ratio, on_step=True, on_epoch=False)
            # Reset
            self.accumulated_l1 = 0.0
            self.accumulated_ratio = 0.0
            self.accumulate_count = 0
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Get data
        s1t1 = batch['T1']
        s1t2 = batch['T2']
        hr = batch['HR']
        
        # Forward
        s1sr_up, s1sr_down = self(s1t1, s1t2)
        
        # Fusion output
        sr_fusion = 0.5 * s1sr_up[-1] + 0.5 * s1sr_down[-1]
        
        # Compute L1 loss
        l1_loss = self.criterion_L1(sr_fusion, hr)
        
        # Denormalize for metrics
        sr_denorm = denorm(sr_fusion)
        hr_denorm = denorm(hr)
        
        # Update metrics
        self.val_psnr.update(sr_denorm, hr_denorm)
        self.val_ssim.update(sr_denorm, hr_denorm)
        
        # Log L1
        self.log('val/l1_loss', l1_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log images (only from rank 0, first batch)
        if batch_idx == 0 and self.trainer.is_global_zero:
            n_imgs = min(8, sr_fusion.size(0))
            self.logger.experiment.add_image(
                'val/SR', make_grid(sr_denorm[:n_imgs], nrow=4), self.global_step
            )
            self.logger.experiment.add_image(
                'val/HR', make_grid(hr_denorm[:n_imgs], nrow=4), self.global_step
            )
            self.logger.experiment.add_image(
                'val/S1T1', make_grid(denorm(s1t1)[:n_imgs], nrow=4), self.global_step
            )
            self.logger.experiment.add_image(
                'val/S1T2', make_grid(denorm(s1t2)[:n_imgs], nrow=4), self.global_step
            )
        
        return {'l1_loss': l1_loss}

    def on_validation_epoch_end(self):
        # Compute final metrics
        psnr = self.val_psnr.compute()
        ssim = self.val_ssim.compute()
        
        # Log metrics
        self.log('val/psnr', psnr, prog_bar=True, sync_dist=True)
        self.log('val/ssim', ssim, prog_bar=True, sync_dist=True)
        
        # Track best SSIM (only on rank 0)
        if self.trainer.is_global_zero and ssim > self.best_ssim:
            self.best_ssim = ssim
            self.log('val/best_ssim', self.best_ssim, sync_dist=True)
        
        # Reset metrics
        self.val_psnr.reset()
        self.val_ssim.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg_train['lr'],
            betas=tuple(self.cfg_train['betas'])
        )
        return optimizer
