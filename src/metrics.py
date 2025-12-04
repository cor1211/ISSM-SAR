
import torch
import torch.nn.functional as F
import numpy as np


def psnr_torch(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    img1, img2: Tensor [B, C, H, W] hoặc [C, H, W], giá trị [0, 1] hoặc [0, 255]
    """
    if img1.dtype != torch.float32:
        img1 = img1.float()
    if img2.dtype != torch.float32:
        img2 = img2.float()

    mse = torch.mean((img1 - img2) ** 2)
    print(mse)
    if mse == 0:
        return torch.tensor(float('inf'))
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr

def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Compute PSNR per image for inputs with shape [B, C, H, W] (or [C, H, W]).
    Returns a tensor of shape [B] with PSNR in dB.
    """
    # support single image input [C, H, W]
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    assert img1.shape == img2.shape and img1.dim() == 4, "Inputs must have shape [B, C, H, W]"

    if img1.dtype != torch.float32:
        img1 = img1.float()
    if img2.dtype != torch.float32:
        img2 = img2.float()

    # MSE per image in the batch
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
    print(f'New mse: {mse.shape}')
    # avoid log10(0); set PSNR to +inf where mse == 0
    zero_mask = mse == 0
    print('zero_mask: ', zero_mask)
    safe_mse = mse.clone()
    safe_mse[zero_mask] = 1e-10

    psnr = 10.0 * torch.log10((max_val ** 2) / safe_mse)
    if zero_mask.any():
        psnr[zero_mask] = float('inf')

    return torch.mean(psnr)


def ssim_torch(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> torch.Tensor:
    """
    img1, img2: Tensor [B, C, H, W] với giá trị [0, 1]
    """
    if img1.dtype != torch.float32:
        img1 = img1.float()
    if img2.dtype != torch.float32:
        img2 = img2.float()

    # Tạo kernel Gaussian 11x11
    sigma = 1.5
    coords = torch.arange(window_size).to(img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    window = window.expand(img1.size(1), 1, window_size, window_size)

    # Tính trung bình cục bộ
    mu1 = F.conv2d(img1, window, groups=img1.size(1), padding=window_size // 2)
    mu2 = F.conv2d(img2, window, groups=img2.size(1), padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=img1.size(1), padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=img2.size(1), padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=img1.size(1), padding=window_size // 2) - mu1_mu2

    # Hằng số ổn định
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

if __name__ == '__main__':
    img1 = torch.rand(4, 1, 256, 256)
    img2 = torch.rand(4, 1, 256, 256)
    print(psnr_torch(img1, img2))
    print(compute_psnr(img1, img2))