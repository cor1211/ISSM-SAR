"""
GAN Loss Functions for ISSM-SAR

Implements Relativistic Average GAN (RaGAN) losses for stable training.
Based on ESRGAN paper: https://arxiv.org/abs/1809.00219
"""
import torch
import torch.nn.functional as F


def discriminator_loss_ragan(
    D,
    real: torch.Tensor,
    fake: torch.Tensor,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Relativistic Average Discriminator Loss
    
    D learns that real images are relatively more realistic than fakes.
    L_D = E[BCE(D(x_r) - E[D(x_f)], 1)] + E[BCE(D(x_f) - E[D(x_r)], 0)]
    
    Args:
        D: Discriminator network
        real: Real HR images
        fake: Generated SR images (detached from generator)
        label_smoothing: Optional label smoothing (0.0 = no smoothing)
        
    Returns:
        Discriminator loss
    """
    # Get predictions (logits, no sigmoid)
    pred_real = D(real)
    pred_fake = D(fake.detach())
    
    # Labels with optional smoothing
    real_label = 1.0 - label_smoothing
    fake_label = 0.0 + label_smoothing
    
    # Relativistic average
    # D(x_r) should be higher than average of D(x_f)
    loss_real = F.binary_cross_entropy_with_logits(
        pred_real - pred_fake.mean(dim=0, keepdim=True),
        torch.full_like(pred_real, real_label)
    )
    
    # D(x_f) should be lower than average of D(x_r)
    loss_fake = F.binary_cross_entropy_with_logits(
        pred_fake - pred_real.mean(dim=0, keepdim=True),
        torch.full_like(pred_fake, fake_label)
    )
    
    return (loss_real + loss_fake) / 2


def generator_loss_ragan(
    D,
    real: torch.Tensor,
    fake: torch.Tensor
) -> torch.Tensor:
    """
    Relativistic Average Generator Loss
    
    G tries to make fake images relatively more realistic than real.
    L_G = E[BCE(D(x_r) - E[D(x_f)], 0)] + E[BCE(D(x_f) - E[D(x_r)], 1)]
    
    Args:
        D: Discriminator network
        real: Real HR images (for reference, gradients not computed)
        fake: Generated SR images
        
    Returns:
        Generator adversarial loss
    """
    # Get predictions
    pred_fake = D(fake)
    
    # Detach real predictions - we don't want gradients flowing through real
    with torch.no_grad():
        pred_real = D(real)
    
    # Relativistic average - generator wants:
    # D(x_f) to be higher than E[D(x_r)]
    loss_fake = F.binary_cross_entropy_with_logits(
        pred_fake - pred_real.mean(dim=0, keepdim=True),
        torch.ones_like(pred_fake)
    )
    
    # D(x_r) to be lower than E[D(x_f)]
    loss_real = F.binary_cross_entropy_with_logits(
        pred_real - pred_fake.mean(dim=0, keepdim=True),
        torch.zeros_like(pred_real)
    )
    
    return (loss_real + loss_fake) / 2


def discriminator_loss_lsgan(
    D,
    real: torch.Tensor,
    fake: torch.Tensor
) -> torch.Tensor:
    """
    Least Squares GAN Discriminator Loss (alternative, more stable)
    
    L_D = 0.5 * E[(D(x_r) - 1)²] + 0.5 * E[D(x_f)²]
    
    Args:
        D: Discriminator network
        real: Real HR images
        fake: Generated SR images (detached)
        
    Returns:
        LSGAN discriminator loss
    """
    pred_real = D(real)
    pred_fake = D(fake.detach())
    
    loss_real = torch.mean((pred_real - 1.0) ** 2)
    loss_fake = torch.mean(pred_fake ** 2)
    
    return (loss_real + loss_fake) / 2


def generator_loss_lsgan(
    D,
    fake: torch.Tensor
) -> torch.Tensor:
    """
    Least Squares GAN Generator Loss
    
    L_G = 0.5 * E[(D(x_f) - 1)²]
    
    Args:
        D: Discriminator network
        fake: Generated SR images
        
    Returns:
        LSGAN generator loss
    """
    pred_fake = D(fake)
    return torch.mean((pred_fake - 1.0) ** 2)


def get_adversarial_weight(
    epoch: int,
    warmup_epochs: int = 5,
    ramp_epochs: int = 10,
    max_weight: float = 0.02,
    min_weight: float = 0.001
) -> float:
    """
    Calculate adversarial loss weight with warmup and ramp-up schedule
    
    Schedule:
        - Epochs 0 to warmup: weight = 0 (generator-only training)
        - Epochs warmup to warmup+ramp: linear increase from min_weight to max_weight
        - Epochs after: weight = max_weight
    
    Args:
        epoch: Current epoch
        warmup_epochs: Epochs with no adversarial loss
        ramp_epochs: Epochs to linearly increase weight (0 means immediate full weight)
        max_weight: Final adversarial weight
        min_weight: Starting weight when ramp begins (ensures GAN starts immediately)
        
    Returns:
        Current adversarial loss weight
    """
    if epoch < warmup_epochs:
        return 0.0
    elif ramp_epochs <= 0:
        # No ramp, immediate full weight
        return max_weight
    elif epoch < warmup_epochs + ramp_epochs:
        progress = (epoch - warmup_epochs) / ramp_epochs
        # Linear interpolation from min_weight to max_weight
        return min_weight + (max_weight - min_weight) * progress
    else:
        return max_weight


if __name__ == '__main__':
    # Test losses
    from discriminator import SARPatchDiscriminator
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = SARPatchDiscriminator().to(device)
    
    real = torch.randn(2, 1, 256, 256).to(device)
    fake = torch.randn(2, 1, 256, 256).to(device)
    
    # Test RaGAN
    d_loss = discriminator_loss_ragan(D, real, fake)
    g_loss = generator_loss_ragan(D, real, fake)
    print(f"RaGAN - D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")
    
    # Test LSGAN
    d_loss_ls = discriminator_loss_lsgan(D, real, fake)
    g_loss_ls = generator_loss_lsgan(D, fake)
    print(f"LSGAN - D loss: {d_loss_ls.item():.4f}, G loss: {g_loss_ls.item():.4f}")
    
    # Test schedule
    for epoch in [0, 3, 5, 10, 15, 20]:
        w = get_adversarial_weight(epoch)
        print(f"Epoch {epoch}: adv_weight = {w:.4f}")
