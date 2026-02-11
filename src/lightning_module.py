"""
PyTorch Lightning Module for ISSM-SAR Multi-GPU Training with GAN Support
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.utils import make_grid

from src.model import ISSM_SAR
from src.loss import lratio_loss, gradient_loss, frequency_domain_loss, speckle_aware_loss
from src.discriminator import SARPatchDiscriminator
from src.gan_losses import (
    discriminator_loss_ragan, generator_loss_ragan,
    discriminator_loss_lsgan, generator_loss_lsgan,
    get_adversarial_weight
)
from src.perceptual_loss import VGGPerceptualLoss


def denorm(x, mean=0.5, std=0.5):
    """Denormalize from [-1, 1] to [0, 1]"""
    return (x * std + mean).clamp(0, 1)


class ISSM_SAR_Lightning(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Enable anomaly detection for debugging
        # torch.autograd.set_detect_anomaly(True)
        
        # Automatic optimization disabled for manual GAN training
        self.automatic_optimization = False
        
        # Config
        self.cfg_train = config['train']
        self.cfg_model = config['model']
        self.cfg_gan = config.get('gan', {})
        self.cfg_disc = config.get('discriminator', {})
        self.cfg_lightning = config.get('lightning', {})
        self.cfg_percep = config.get('perceptual', {})
        
        # Validation settings
        self.limit_val_batches = self.cfg_lightning.get('limit_val_batches')
        
        # GAN settings
        self.gan_enabled = self.cfg_gan.get('enabled', False)
        self.gan_loss_type = self.cfg_gan.get('loss_type', 'ragan')
        self.warmup_epochs = self.cfg_gan.get('warmup_epochs', 5)
        self.ramp_epochs = self.cfg_gan.get('ramp_epochs', 10)
        self.max_adv_weight = self.cfg_gan.get('max_weight', 0.02)
        self.label_smoothing = self.cfg_gan.get('label_smoothing', 0.0)
        self.d_lr_mult = self.cfg_gan.get('d_lr_mult', 4.0)
        
        # Loss coefficients
        self.theta_1 = self.cfg_train['theta_1']  # Ratio loss
        self.theta_2 = self.cfg_train['theta_2']  # L1 loss
        self.theta_3 = self.cfg_train.get('theta_3', 0.0)  # Gradient loss
        self.theta_4 = self.cfg_train.get('theta_4', 0.0)  # Frequency domain loss
        self.theta_5 = self.cfg_train.get('theta_5', 0.0)  # Speckle-aware loss
        self.theta_6 = self.cfg_train.get('theta_6', 0.0)  # Adversarial loss (via schedule)
        self.theta_7 = self.cfg_train.get('theta_7', 0.0)  # Perceptual loss
        self.theta_style = self.cfg_train.get('theta_style', 0.0)  # Style loss
        self.component_losses = self.cfg_train['component_losses']
        self.val_step = self.cfg_train['val_step']
        
        # Generator (ISSM_SAR model)
        self.model = ISSM_SAR(self.cfg_model)
        
        # Discriminator (only if GAN enabled)
        if self.gan_enabled:
            self.discriminator = SARPatchDiscriminator(
                in_channels=self.cfg_disc.get('in_channels', 1),
                ndf=self.cfg_disc.get('ndf', 64),
                use_spectral_norm=self.cfg_disc.get('use_spectral_norm', True)
            )
        else:
            self.discriminator = None
            
        # Perceptual Loss
        if self.theta_7 > 0:
            self.perceptual_loss = VGGPerceptualLoss(
                layer_weights=self.cfg_percep.get('layer_weights', {'34': 1.0}),
                use_input_norm=self.cfg_percep.get('use_input_norm', True)
            )
        else:
            self.perceptual_loss = None
        
        # Loss function
        self.criterion_L1 = nn.L1Loss()
        
        # Metrics
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True)
        
        # Best metric tracking
        self.best_ssim = 0.0
        self.best_lpips = float('inf')
        
        # Accumulate losses for logging
        self.accumulated_l1 = 0.0
        self.accumulated_ratio = 0.0
        self.accumulated_grad = 0.0
        self.accumulated_freq = 0.0
        self.accumulated_speckle = 0.0
        self.accumulated_adv = 0.0
        self.accumulated_d_loss = 0.0
        self.accumulated_percep = 0.0
        self.accumulated_style = 0.0
        self.accumulate_count = 0

    def forward(self, s1t1, s1t2):
        return self.model(s1t1, s1t2)
    
    def _compute_reconstruction_losses(self, sr_up, sr_down, hr, dtype):
        """Compute all reconstruction losses (non-adversarial)"""
        total_ratio_loss = torch.tensor(0.0, device=self.device, dtype=dtype)
        total_l1_loss = torch.tensor(0.0, device=self.device, dtype=dtype)
        total_grad_loss = torch.tensor(0.0, device=self.device, dtype=dtype)
        total_freq_loss = torch.tensor(0.0, device=self.device, dtype=dtype)
        total_speckle_loss = torch.tensor(0.0, device=self.device, dtype=dtype)
        
        # Component losses for intermediate outputs
        if self.component_losses:
            for idx in range(self.cfg_model['num_ifs']):
                if self.theta_1 != 0:
                    total_ratio_loss += lratio_loss(sr_up[idx], hr) + lratio_loss(sr_down[idx], hr)
                if self.theta_2 != 0:
                    total_l1_loss += self.criterion_L1(sr_up[idx], hr) + self.criterion_L1(sr_down[idx], hr)
                if self.theta_3 != 0:
                    total_grad_loss += gradient_loss(sr_up[idx], hr) + gradient_loss(sr_down[idx], hr)
                if self.theta_4 != 0:
                    total_freq_loss += frequency_domain_loss(sr_up[idx], hr) + frequency_domain_loss(sr_down[idx], hr)
                if self.theta_5 != 0:
                    total_speckle_loss += speckle_aware_loss(sr_up[idx], hr) + speckle_aware_loss(sr_down[idx], hr)
        
        # Final output losses
        if self.theta_1 != 0:
            total_ratio_loss += lratio_loss(sr_up[-1], hr) + lratio_loss(sr_down[-1], hr)
        if self.theta_2 != 0:
            total_l1_loss += self.criterion_L1(sr_up[-1], hr) + self.criterion_L1(sr_down[-1], hr)
        if self.theta_3 != 0:
            total_grad_loss += gradient_loss(sr_up[-1], hr) + gradient_loss(sr_down[-1], hr)
        if self.theta_4 != 0:
            total_freq_loss += frequency_domain_loss(sr_up[-1], hr) + frequency_domain_loss(sr_down[-1], hr)
        if self.theta_5 != 0:
            total_speckle_loss += speckle_aware_loss(sr_up[-1], hr) + speckle_aware_loss(sr_down[-1], hr)
        
        return {
            'ratio': total_ratio_loss,
            'l1': total_l1_loss,
            'grad': total_grad_loss,
            'freq': total_freq_loss,
            'speckle': total_speckle_loss
        }

    def on_load_checkpoint(self, checkpoint):
        """
        Handle backward compatibility for checkpoints that don't have LPIPS weights.
        This injects the current model's LPIPS weights into the checkpoint state_dict
        if they are missing, preventing strict loading errors.
        """
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        
        # Keys related to LPIPS
        # torchmetrics usually prefixes with val_lpips.net...
        lpips_keys = [k for k in model_state_dict.keys() if "val_lpips" in k]
        
        for k in lpips_keys:
            if k not in state_dict:
                # Inject current initialized value (pretrained)
                state_dict[k] = model_state_dict[k]
        
    def training_step(self, batch, batch_idx):
        # Get optimizers
        if self.gan_enabled:
            opt_g, opt_d = self.optimizers()
        else:
            opt_g = self.optimizers()
            opt_d = None
        
        # Get data
        s1t1 = batch['T1']
        s1t2 = batch['T2']
        hr = batch['HR']
        
        # Forward pass (generator)
        s1sr_up, s1sr_down = self(s1t1, s1t2)
        
        # Get fused SR output for discriminator
        sr_fusion = 0.5 * s1sr_up[-1] + 0.5 * s1sr_down[-1]
        
        # Calculate adversarial weight based on schedule
        current_epoch = self.current_epoch
        adv_weight = get_adversarial_weight(
            current_epoch, 
            self.warmup_epochs, 
            self.ramp_epochs, 
            self.max_adv_weight,
            min_weight=0.001  # Start with small weight immediately
        ) if self.gan_enabled else 0.0
        
        # Safety: clamp adv_weight between 0 and max_adv_weight * 10 (just in case)
        adv_weight = max(0.0, min(adv_weight, self.max_adv_weight * 10))
        
        # ==================== Train Discriminator ====================
        d_loss = torch.tensor(0.0, device=self.device, dtype=s1t1.dtype)
        
        if self.gan_enabled and adv_weight > 0 and opt_d is not None:
            # Freeze generator
            self.toggle_optimizer(opt_d)
            
            if self.gan_loss_type == 'ragan':
                d_loss = discriminator_loss_ragan(
                    self.discriminator, hr, sr_fusion.detach(), self.label_smoothing
                )
            else:  # lsgan
                d_loss = discriminator_loss_lsgan(
                    self.discriminator, hr, sr_fusion.detach()
                )
            
            # Backward and step for D
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()
            
            self.untoggle_optimizer(opt_d)
        
        # ==================== Train Generator ====================
        self.toggle_optimizer(opt_g)
        
        # Compute reconstruction losses
        recon_losses = self._compute_reconstruction_losses(
            s1sr_up, s1sr_down, hr, s1t1.dtype
        )
        for k, v in recon_losses.items():
            if torch.isnan(v) or torch.isinf(v):
                print(f"Warning: NaN/Inf in recon_loss {k}")
                recon_losses[k] = torch.tensor(0.0, device=self.device, dtype=s1t1.dtype)
        
        # Compute adversarial loss for generator
        g_adv_loss = torch.tensor(0.0, device=self.device, dtype=s1t1.dtype)
        
        if self.gan_enabled and adv_weight > 0:
            if self.gan_loss_type == 'ragan':
                g_adv_loss = generator_loss_ragan(self.discriminator, hr, sr_fusion)
            else:  # lsgan
                g_adv_loss = generator_loss_lsgan(self.discriminator, sr_fusion)
        
        # Compute perceptual and style loss
        l_percep = torch.tensor(0.0, device=self.device, dtype=s1t1.dtype)
        l_style = torch.tensor(0.0, device=self.device, dtype=s1t1.dtype)
        
        if (self.theta_7 > 0 or self.theta_style > 0) and self.perceptual_loss is not None:
            # VGG expects [0, 1] input for its internal normalization
            # Returns (content_loss, style_loss)
            l_percep, l_style = self.perceptual_loss(denorm(sr_fusion), denorm(hr))
            
            if torch.isnan(l_percep) or torch.isinf(l_percep):
                 print(f"Warning: NaN/Inf in l_percep")
                 l_percep = torch.tensor(0.0, device=self.device, dtype=s1t1.dtype)
                 
            if torch.isnan(l_style) or torch.isinf(l_style):
                 print(f"Warning: NaN/Inf in l_style")
                 l_style = torch.tensor(0.0, device=self.device, dtype=s1t1.dtype)
        
        # Add explicit L1 loss on the final fused output
        # This ensures the actual inference output is optimized for reconstruction
        l1_fusion = self.criterion_L1(sr_fusion, hr)
        
        # Total generator loss
        g_total_loss = (
            self.theta_1 * recon_losses['ratio'] +
            self.theta_2 * (recon_losses['l1'] + l1_fusion) +  # Add fusion L1 to branch L1s
            self.theta_3 * recon_losses['grad'] +
            self.theta_4 * recon_losses['freq'] +
            self.theta_5 * recon_losses['speckle'] +
            self.theta_7 * l_percep +
            self.theta_style * l_style +
            adv_weight * g_adv_loss
        )
        
        # Check for NaN
        if torch.isnan(g_total_loss) or torch.isinf(g_total_loss):
            self.log('Debug/Train/nan_detected', 1.0, prog_bar=True)
            self.untoggle_optimizer(opt_g)
            return None
        
        # Backward and step for G
        opt_g.zero_grad()
        self.manual_backward(g_total_loss)
        
        # Gradient clipping
        grad_clip = self.cfg_train.get('grad_clip', 0.0)
        if grad_clip > 0:
            self.clip_gradients(opt_g, gradient_clip_val=grad_clip, gradient_clip_algorithm="norm")
        
        opt_g.step()
        self.untoggle_optimizer(opt_g)
        
        # ==================== Logging ====================
        # Accumulate losses
        self.accumulated_l1 += recon_losses['l1'].item()
        self.accumulated_ratio += recon_losses['ratio'].item()
        self.accumulated_grad += recon_losses['grad'].item()
        self.accumulated_freq += recon_losses['freq'].item()
        self.accumulated_speckle += recon_losses['speckle'].item()
        self.accumulated_percep += l_percep.item()
        self.accumulated_style += l_style.item()
        self.accumulated_adv += g_adv_loss.item()
        self.accumulated_d_loss += d_loss.item()
        self.accumulate_count += 1
        
        # Log losses
        self.log('Loss/Train/total', g_total_loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=False)
        self.log('Loss/Train/l1', recon_losses['l1'], on_step=True, on_epoch=False, sync_dist=False)
        self.log('Loss/Train/ratio', recon_losses['ratio'], on_step=True, on_epoch=False, sync_dist=False)
        self.log('Loss/Train/gradient', recon_losses['grad'], on_step=True, on_epoch=False, sync_dist=False)
        self.log('Loss/Train/frequency', recon_losses['freq'], on_step=True, on_epoch=False, sync_dist=False)
        self.log('Loss/Train/speckle', recon_losses['speckle'], on_step=True, on_epoch=False, sync_dist=False)
        self.log('Loss/Train/perceptual', l_percep, on_step=True, on_epoch=False, sync_dist=False)
        self.log('Loss/Train/style', l_style, on_step=True, on_epoch=False, sync_dist=False)
        
        if self.gan_enabled:
            self.log('Loss/Train/G_adv', g_adv_loss, on_step=True, on_epoch=False, sync_dist=False)
            self.log('Loss/Train/D', d_loss, on_step=True, on_epoch=False, sync_dist=False)
            self.log('Loss/Train/adv_weight', adv_weight, on_step=True, on_epoch=False, sync_dist=False)
        
        # Log averages every val_step
        if self.global_step > 0 and self.global_step % self.val_step == 0:
            n = max(self.accumulate_count, 1)
            self.log('Debug/Train/avg_l1', self.accumulated_l1 / n, on_step=True, on_epoch=False)
            self.log('Debug/Train/avg_ratio', self.accumulated_ratio / n, on_step=True, on_epoch=False)
            self.log('Debug/Train/avg_gradient', self.accumulated_grad / n, on_step=True, on_epoch=False)
            self.log('Debug/Train/avg_frequency', self.accumulated_freq / n, on_step=True, on_epoch=False)
            self.log('Debug/Train/avg_speckle', self.accumulated_speckle / n, on_step=True, on_epoch=False)
            self.log('Debug/Train/avg_perceptual', self.accumulated_percep / n, on_step=True, on_epoch=False)
            self.log('Debug/Train/avg_style', self.accumulated_style / n, on_step=True, on_epoch=False)
            if self.gan_enabled:
                self.log('Debug/Train/avg_G_adv', self.accumulated_adv / n, on_step=True, on_epoch=False)
                self.log('Debug/Train/avg_D', self.accumulated_d_loss / n, on_step=True, on_epoch=False)
            
            # Reset accumulators
            self.accumulated_l1 = 0.0
            self.accumulated_ratio = 0.0
            self.accumulated_grad = 0.0
            self.accumulated_freq = 0.0
            self.accumulated_speckle = 0.0
            self.accumulated_percep = 0.0
            self.accumulated_style = 0.0
            self.accumulated_adv = 0.0
            self.accumulated_d_loss = 0.0
            self.accumulate_count = 0
        
        return g_total_loss

    def validation_step(self, batch, batch_idx):
        # Manually limit validation batches if configured
        if self.limit_val_batches is not None and batch_idx >= self.limit_val_batches:
            return

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
        # Update metrics
        self.val_psnr.update(sr_denorm, hr_denorm)
        self.val_ssim.update(sr_denorm, hr_denorm)
        
        # LPIPS expects 3-channel input. Repeat 1-channel SAR to 3 channels.
        sr_lpips = sr_denorm.repeat(1, 3, 1, 1)
        hr_lpips = hr_denorm.repeat(1, 3, 1, 1)
        
        self.val_lpips.update(sr_lpips, hr_lpips)
        
        # Log L1
        self.log('Loss/Val/l1', l1_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log images
        is_sanity_check = self.trainer.sanity_checking
        should_log_images = (
            batch_idx < 3  # Log first 3 batches to see variety
            and self.trainer.is_global_zero 
            and self.logger is not None
            and not is_sanity_check
        )
        
        if should_log_images:
            try:
                n_imgs = min(8, sr_fusion.size(0))
                tag_suffix = f"_batch{batch_idx}" if batch_idx > 0 else ""
                
                self.logger.experiment.add_image(
                    f'Image/Val/SR{tag_suffix}', make_grid(sr_denorm[:n_imgs], nrow=4), self.global_step
                )
                self.logger.experiment.add_image(
                    f'Image/Val/HR{tag_suffix}', make_grid(hr_denorm[:n_imgs], nrow=4), self.global_step
                )
                self.logger.experiment.add_image(
                    f'Image/Val/S1T1{tag_suffix}', make_grid(denorm(s1t1)[:n_imgs], nrow=4), self.global_step
                )
                self.logger.experiment.add_image(
                    f'Image/Val/S1T2{tag_suffix}', make_grid(denorm(s1t2)[:n_imgs], nrow=4), self.global_step
                )
            except Exception:
                pass
        
        return {'l1_loss': l1_loss}

    def on_validation_epoch_end(self):
        # Compute final metrics
        psnr = self.val_psnr.compute()
        ssim = self.val_ssim.compute()
        
        # Log metrics
        # Log metrics
        self.log('Metrics/Val/PSNR', psnr, prog_bar=True, sync_dist=True)
        self.log('Metrics/Val/SSIM', ssim, prog_bar=True, sync_dist=True)
        
        # Compute LPIPS
        lpips = self.val_lpips.compute()
        self.log('Metrics/Val/LPIPS', lpips, prog_bar=True, sync_dist=True)
        
        # Track best SSIM
        if ssim > self.best_ssim:
            self.best_ssim = float(ssim)
            
        # Track best LPIPS (lower is better)
        if lpips < self.best_lpips:
            self.best_lpips = float(lpips)
        
        self.log('Metrics/Val/best_SSIM', self.best_ssim, sync_dist=False, rank_zero_only=True)
        self.log('Metrics/Val/best_LPIPS', self.best_lpips, sync_dist=False, rank_zero_only=True)
        
        # Reset metrics
        self.val_psnr.reset()
        self.val_ssim.reset()
        self.val_lpips.reset()

    def configure_optimizers(self):
        # Generator optimizer
        opt_g = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg_train['lr'],
            betas=tuple(self.cfg_train['betas'])
        )
        
        if self.gan_enabled and self.discriminator is not None:
            # Discriminator optimizer (faster learning rate)
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.cfg_train['lr'] * self.d_lr_mult,
                betas=tuple(self.cfg_train['betas'])
            )
            return [opt_g, opt_d]
        
        return opt_g