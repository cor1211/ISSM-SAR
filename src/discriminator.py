"""
PatchGAN Discriminator for ISSM-SAR Super Resolution

32×32 Patch discriminator with Spectral Normalization for training stability.
Based on pix2pix PatchGAN but optimized for SAR imagery.
"""
import torch
import torch.nn as nn


class SARPatchDiscriminator(nn.Module):
    """
    32×32 PatchGAN Discriminator with Spectral Normalization
    
    Architecture:
        Input: (B, 1, 256, 256) → Output: (B, 1, 30, 30)
        Each output pixel represents real/fake prediction for ~32×32 patch
    
    Receptive field calculation:
        Conv(4,2,1) → RF=4, stride=2
        Conv(4,2,1) → RF=10, stride=4  
        Conv(4,2,1) → RF=22, stride=8
        Conv(4,1,1) → RF=34 ≈ 32×32
    
    Args:
        in_channels: Input image channels (1 for SAR)
        ndf: Base discriminator feature channels
        use_spectral_norm: Whether to apply spectral normalization
    """
    
    def __init__(self, in_channels: int = 1, ndf: int = 64, use_spectral_norm: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.ndf = ndf
        
        # Helper to optionally wrap with spectral norm
        def maybe_spectral_norm(layer):
            if use_spectral_norm:
                return nn.utils.spectral_norm(layer)
            return layer
        
        self.layers = nn.Sequential(
            # Layer 1: 256 → 128, no normalization on first layer
            maybe_spectral_norm(
                nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 128 → 64
            maybe_spectral_norm(
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 64 → 32
            maybe_spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 32 → 31 (stride=1 for finer control)
            maybe_spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output Layer: 31 → 30, single channel output
            maybe_spectral_norm(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1, bias=False)
            )
            # No sigmoid - using logits for numerically stable loss
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with normal distribution"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            Patch predictions (B, 1, H', W') where each value is logit for real/fake
        """
        return self.layers(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for capturing both fine and coarse textures
    
    Uses 2 discriminators at different scales:
    - D1: Full resolution (256×256)
    - D2: Half resolution (128×128)
    
    This helps capture both local speckle and global structure.
    """
    
    def __init__(self, in_channels: int = 1, ndf: int = 64, use_spectral_norm: bool = True):
        super().__init__()
        
        # Full scale discriminator
        self.D1 = SARPatchDiscriminator(in_channels, ndf, use_spectral_norm)
        
        # Half scale discriminator
        self.D2 = SARPatchDiscriminator(in_channels, ndf, use_spectral_norm)
        
        # Downsampler for D2
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input image (B, C, H, W)
            
        Returns:
            Tuple of (D1_output, D2_output)
        """
        out1 = self.D1(x)
        out2 = self.D2(self.downsample(x))
        return out1, out2


if __name__ == '__main__':
    # Test discriminator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Single scale
    D = SARPatchDiscriminator(in_channels=1, ndf=64).to(device)
    x = torch.randn(2, 1, 256, 256).to(device)
    out = D(x)
    print(f"Single-scale D: Input {x.shape} → Output {out.shape}")
    
    # Multi scale
    D_multi = MultiScaleDiscriminator(in_channels=1, ndf=64).to(device)
    out1, out2 = D_multi(x)
    print(f"Multi-scale D: Input {x.shape} → D1 {out1.shape}, D2 {out2.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in D.parameters())
    print(f"Single-scale D parameters: {total_params:,}")
