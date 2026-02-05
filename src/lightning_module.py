"""
PyTorch Lightning Module for ISSM-SAR Multi-GPU Training
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid

from src.model import ISSM_SAR
from src.loss import lratio_loss, gradient_loss, frequency_domain_loss, speckle_aware_loss


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
        self.theta_1 = self.cfg_train['theta_1']  # Ratio loss
        self.theta_2 = self.cfg_train['theta_2']  # L1 loss
        self.theta_3 = self.cfg_train.get('theta_3', 0.0)  # Gradient loss
        self.theta_4 = self.cfg_train.get('theta_4', 0.0)  # Frequency domain loss (anti-oversmoothing)
        self.theta_5 = self.cfg_train.get('theta_5', 0.0)  # Speckle-aware loss (SAR-specific)
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
        self.accumulated_grad = 0.0
        self.accumulated_freq = 0.0
        self.accumulated_speckle = 0.0
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
        total_grad_loss = torch.tensor(0.0, device=self.device, dtype=s1t1.dtype)
        total_freq_loss = torch.tensor(0.0, device=self.device, dtype=s1t1.dtype)
        total_speckle_loss = torch.tensor(0.0, device=self.device, dtype=s1t1.dtype)
        
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
        if self.theta_3 != 0.0:
            total_grad_loss = total_grad_loss + gradient_loss(s1sr_up[-1], hr) + gradient_loss(s1sr_down[-1], hr)
        
        # Anti-oversmoothing losses (only on final output for efficiency)
        if self.theta_4 != 0.0:
            total_freq_loss = frequency_domain_loss(s1sr_up[-1], hr) + frequency_domain_loss(s1sr_down[-1], hr)
        if self.theta_5 != 0.0:
            total_speckle_loss = speckle_aware_loss(s1sr_up[-1], hr) + speckle_aware_loss(s1sr_down[-1], hr)

        loss = (self.theta_1 * total_ratio_loss + 
                self.theta_2 * total_l1_loss + 
                self.theta_3 * total_grad_loss +
                self.theta_4 * total_freq_loss +
                self.theta_5 * total_speckle_loss)
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            self.log('Debug/Train/nan_detected', 1.0, prog_bar=True)
            return None  # Skip this batch
        
        # Accumulate for periodic logging
        self.accumulated_l1 += total_l1_loss.item()
        self.accumulated_ratio += total_ratio_loss.item()
        self.accumulated_grad += total_grad_loss.item()
        self.accumulated_freq += total_freq_loss.item()
        self.accumulated_speckle += total_speckle_loss.item()
        self.accumulate_count += 1
        
        # Log every step (sync_dist=False for on_step to avoid deadlock)
        # Using hierarchical tags: Loss/Train/, Metrics/, Image/ for better TensorBoard organization
        self.log('Loss/Train/total', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=False)
        self.log('Loss/Train/l1', total_l1_loss, on_step=True, on_epoch=False, sync_dist=False)
        self.log('Loss/Train/ratio', total_ratio_loss, on_step=True, on_epoch=False, sync_dist=False)
        self.log('Loss/Train/gradient', total_grad_loss, on_step=True, on_epoch=False, sync_dist=False)
        self.log('Loss/Train/frequency', total_freq_loss, on_step=True, on_epoch=False, sync_dist=False)
        self.log('Loss/Train/speckle', total_speckle_loss, on_step=True, on_epoch=False, sync_dist=False)
        
        # Log average every val_step
        if self.global_step > 0 and self.global_step % self.val_step == 0:
            avg_l1 = self.accumulated_l1 / max(self.accumulate_count, 1)
            avg_ratio = self.accumulated_ratio / max(self.accumulate_count, 1)
            avg_grad = self.accumulated_grad / max(self.accumulate_count, 1)
            avg_freq = self.accumulated_freq / max(self.accumulate_count, 1)
            avg_speckle = self.accumulated_speckle / max(self.accumulate_count, 1)
            self.log('Debug/Train/avg_l1', avg_l1, on_step=True, on_epoch=False)
            self.log('Debug/Train/avg_ratio', avg_ratio, on_step=True, on_epoch=False)
            self.log('Debug/Train/avg_gradient', avg_grad, on_step=True, on_epoch=False)
            self.log('Debug/Train/avg_frequency', avg_freq, on_step=True, on_epoch=False)
            self.log('Debug/Train/avg_speckle', avg_speckle, on_step=True, on_epoch=False)
            # Reset
            self.accumulated_l1 = 0.0
            self.accumulated_ratio = 0.0
            self.accumulated_grad = 0.0
            self.accumulated_freq = 0.0
            self.accumulated_speckle = 0.0
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
        self.log('Loss/Val/l1', l1_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log images (only from rank 0, first batch, skip during sanity check)
        # Check: not sanity checking, is rank 0, logger exists, first batch
        is_sanity_check = self.trainer.sanity_checking
        should_log_images = (
            batch_idx == 0 
            and self.trainer.is_global_zero 
            and self.logger is not None
            and not is_sanity_check
        )
        
        if should_log_images:
            try:
                n_imgs = min(8, sr_fusion.size(0))
                self.logger.experiment.add_image(
                    'Image/Val/SR', make_grid(sr_denorm[:n_imgs], nrow=4), self.global_step
                )
                self.logger.experiment.add_image(
                    'Image/Val/HR', make_grid(hr_denorm[:n_imgs], nrow=4), self.global_step
                )
                self.logger.experiment.add_image(
                    'Image/Val/S1T1', make_grid(denorm(s1t1)[:n_imgs], nrow=4), self.global_step
                )
                self.logger.experiment.add_image(
                    'Image/Val/S1T2', make_grid(denorm(s1t2)[:n_imgs], nrow=4), self.global_step
                )
            except Exception as e:
                # Silently skip if logging fails (e.g., during distributed edge cases)
                pass
        
        return {'l1_loss': l1_loss}

    def on_validation_epoch_end(self):
        # Compute final metrics (torchmetrics handles sync automatically)
        psnr = self.val_psnr.compute()
        ssim = self.val_ssim.compute()
        
        # Log metrics (sync_dist=True for validation epoch metrics)
        self.log('Metrics/Val/PSNR', psnr, prog_bar=True, sync_dist=True)
        self.log('Metrics/Val/SSIM', ssim, prog_bar=True, sync_dist=True)
        
        # Track best SSIM - compare on rank 0 only but don't use conditional logging
        # Use rank_zero_only decorator or just track locally
        if ssim > self.best_ssim:
            self.best_ssim = float(ssim)
        
        # Log best SSIM from all ranks (only rank 0 value matters)
        self.log('Metrics/Val/best_SSIM', self.best_ssim, sync_dist=False, rank_zero_only=True)
        
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
