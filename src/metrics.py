
import torch
import torch.nn.functional as F

def psnr_torch(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    img1, img2: Tensor [B, C, H, W] hoặc [C, H, W], giá trị [0, 1] hoặc [0, 255]
    """
    if img1.dtype != torch.float32:
        img1 = img1.float()
    if img2.dtype != torch.float32:
        img2 = img2.float()

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr




def ssim_torch(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> torch.Tensor:
    """
    SSIM tính bằng thuần PyTorch, không cần torchmetrics
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
