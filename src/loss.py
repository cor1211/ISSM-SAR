import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# def lratio_loss(output: torch.Tensor,target: torch.Tensor,eps: float = 1e-6,) -> torch.Tensor:
#     """
#     Compute Ratio Loss (Radiometric Consistency).
#     WARNING: Inputs MUST be in range [0, 1] or positive domain. 
#     Do NOT pass tensors in range [-1, 1] directly.
#     """
#     sum_output = torch.sum(output, dim=(1, 2, 3)) #(B, )
#     # print(sum_output.shape)
#     sum_target = torch.sum(target, dim=(1, 2, 3)) #(B, )
#     ratios = sum_output/ (sum_target + eps)
#     loss_per_img = torch.abs(1-ratios)
#     # print(loss_per_img.shape)
#     loss = torch.mean(loss_per_img)
#     return loss

def lratio_loss(output: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Args:
        output: Raw output từ model (miền giá trị (-inf, +inf), mong muốn hội tụ về [-1, 1])
        target: Ground truth (đã norm về [-1, 1])
    """
    
    # 1. Denormalize giả lập: Mapping không gian [-1, 1] về [0, 1]
    # Công thức: x_01 = (x + 1) / 2
    output_01 = (output + 1) / 2
    target_01 = (target + 1) / 2

    # 2. Xử lý Unbounded (Quan trọng):
    # Vì output là raw, nó có thể ra -1.5 -> output_01 thành -0.25 (Vô lý với miền [0,1])
    # Cần clamp về 0 để đảm bảo tính chất vật lý (không có cường độ âm)
    output_01 = torch.clamp(output_01, min=0.0)
    target_01 = torch.clamp(target_01, min=0.0) # Target chuẩn thì ko cần, nhưng clamp cho an toàn

    # 3. Tính tổng cường độ (Sum Intensity)
    sum_output = torch.sum(output_01, dim=(1, 2, 3)) 
    sum_target = torch.sum(target_01, dim=(1, 2, 3)) 
    
    # 4. Tính Ratio
    # Thêm eps vào mẫu số để tránh chia cho 0
    ratios = sum_output / (sum_target + eps)
    
    # 5. Loss: Lệch khỏi 1
    loss_per_img = torch.abs(1 - ratios)
    
    return torch.mean(loss_per_img)


def l1_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(output, target)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Charbonnier Loss (Robust L1) — used by SwinIR, HAT, ESPCN.
    
    L = sqrt((pred - target)^2 + eps^2)
    
    Advantages over L1:
    - Smooth gradient near zero (L1 has discontinuous gradient at 0)
    - Better convergence for SAR where speckle creates many small residuals
    - More robust to outliers than L2
    """
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def gradient_loss(pred, target):
    # Tính đạo hàm theo phương ngang (x) và dọc (y)
    # pred shape: (B, C, H, W)
    
    # Gradient X: chênh lệch giữa cột sau và cột trước
    pred_dx = pred[:, :, :, :-1] - pred[:, :, :, 1:]
    target_dx = target[:, :, :, :-1] - target[:, :, :, 1:]
    
    # Gradient Y: chênh lệch giữa hàng dưới và hàng trên
    pred_dy = pred[:, :, :-1, :] - pred[:, :, 1:, :]
    target_dy = target[:, :, :-1, :] - target[:, :, 1:, :]
    
    # Tính L1 loss trên các gradient này
    loss_x = F.l1_loss(pred_dx, target_dx)
    loss_y = F.l1_loss(pred_dy, target_dy)
    
    return loss_x + loss_y


# ============== ANTI-OVERSMOOTHING LOSSES ==============

class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss (Jiang et al., ICCV 2021).
    
    Key insight: Neural networks have spectral bias toward low-frequency functions.
    FFL adaptively focuses on "hard frequencies" (those the model struggles with)
    by dynamically weighting the frequency spectrum based on current prediction errors.
    
    Superior to static frequency masking because:
    - Adaptive: weight matrix updates each iteration based on per-frequency error
    - alpha parameter controls focus intensity on hard frequencies
    - Proven to improve both PSNR/SSIM and perceptual quality
    
    Args:
        alpha: Scaling factor for spectrum weight matrix. Higher = more focus on hard frequencies.
        log_matrix: If True, adjust spectrum weight matrix by logarithm for stability.
        batch_matrix: If True, calculate weight matrix using batch-based statistics.
    """
    def __init__(self, alpha: float = 1.0, log_matrix: bool = False, batch_matrix: bool = False):
        super().__init__()
        self.alpha = alpha
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def _get_frequency_distance(self, pred_freq: torch.Tensor, target_freq: torch.Tensor) -> torch.Tensor:
        """Compute per-frequency L2 distance between pred and target spectra."""
        # pred_freq, target_freq: complex tensors [B, C, H, W]
        # Return: real tensor [B, C, H, W] — per-frequency squared error
        diff = pred_freq - target_freq
        # |a + bi|^2 = a^2 + b^2
        distance = torch.real(diff) ** 2 + torch.imag(diff) ** 2
        return distance

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted image [B, C, H, W]
            target: Ground truth image [B, C, H, W]
        Returns:
            Focal frequency loss scalar
        """
        # Compute 2D FFT
        pred_freq = torch.fft.fft2(pred, norm='ortho')
        target_freq = torch.fft.fft2(target, norm='ortho')

        # Per-frequency squared error
        freq_distance = self._get_frequency_distance(pred_freq, target_freq)  # [B, C, H, W]

        # Compute adaptive weight matrix
        # Normalize distance to [0, 1] range per sample for weighting
        if self.batch_matrix:
            # Use batch-level statistics
            weight = freq_distance.detach()  # [B, C, H, W]
            weight = weight / (weight.max() + 1e-8)
        else:
            # Use per-sample statistics
            B = freq_distance.shape[0]
            weight = freq_distance.detach()
            # Normalize per sample
            weight_flat = weight.view(B, -1)
            weight_max = weight_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            weight = weight / (weight_max + 1e-8)

        # Apply log scaling if configured (for numerical stability with large spectra)
        if self.log_matrix:
            weight = torch.log1p(weight)

        # Apply focal weighting: raise to power alpha
        # Higher alpha → more focus on hard (high-error) frequencies
        weight = weight ** self.alpha

        # Weighted frequency loss
        loss = (weight * freq_distance).mean()

        return loss


def frequency_domain_loss(pred: torch.Tensor, target: torch.Tensor, 
                          high_freq_weight: float = 1.0) -> torch.Tensor:
    """
    [LEGACY] Static frequency-domain loss — kept for backward compatibility.
    Consider using FocalFrequencyLoss instead for adaptive frequency weighting.
    """
    # Compute 2D FFT
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)
    
    # Shift zero-frequency component to center
    pred_fft_shifted = torch.fft.fftshift(pred_fft)
    target_fft_shifted = torch.fft.fftshift(target_fft)
    
    # Get magnitude spectrum (log scale for stability, add eps for numerical safety)
    pred_mag = torch.log1p(torch.abs(pred_fft_shifted) + 1e-8)
    target_mag = torch.log1p(torch.abs(target_fft_shifted) + 1e-8)
    
    # Overall magnitude loss
    mag_loss = F.l1_loss(pred_mag, target_mag)
    
    # Create high-frequency mask (circular mask, outer 70% of spectrum)
    B, C, H, W = pred.shape
    cy, cx = H // 2, W // 2
    y_coords = torch.arange(H, device=pred.device).view(-1, 1).expand(H, W)
    x_coords = torch.arange(W, device=pred.device).view(1, -1).expand(H, W)
    distance = torch.sqrt((y_coords - cy).float()**2 + (x_coords - cx).float()**2)
    
    # High-freq: everything beyond 30% of max radius
    max_radius = min(cy, cx)
    high_freq_mask = (distance > max_radius * 0.3).float()
    high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Weighted high-frequency loss
    pred_hf = pred_mag * high_freq_mask
    target_hf = target_mag * high_freq_mask
    hf_loss = F.l1_loss(pred_hf, target_hf)
    
    return mag_loss + high_freq_weight * hf_loss


def speckle_aware_loss(pred: torch.Tensor, target: torch.Tensor, 
                       window_size: int = 7) -> torch.Tensor:
    """
    Speckle-Aware Statistical Loss cho SAR imagery.
    
    SAR speckle có characteristics đặc trưng:
    - Equivalent Number of Looks (ENL) = mean² / variance
    - Coefficient of Variation (CV) = std / mean
    
    Loss này enforce pred phải match local statistics của target,
    đảm bảo speckle distribution được preserve.
    
    Args:
        pred: Predicted SR image [B, C, H, W]
        target: Ground truth HR image [B, C, H, W]
        window_size: Size of local window for statistics computation
    
    Returns:
        Combined speckle statistics loss
    """
    # Ensure positive values for statistics
    # IMPORTANT: Clamp AFTER denorm to handle unbounded model outputs
    pred_pos = torch.clamp((pred + 1) / 2, min=1e-6)
    target_pos = torch.clamp((target + 1) / 2, min=1e-6)
    
    # Create averaging kernel
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device)
    kernel = kernel / (window_size * window_size)
    
    # Compute local mean: E[X]
    pad = window_size // 2
    pred_mean = F.conv2d(pred_pos, kernel, padding=pad)
    target_mean = F.conv2d(target_pos, kernel, padding=pad)
    
    # Compute local variance: E[X²] - E[X]²
    pred_sq_mean = F.conv2d(pred_pos ** 2, kernel, padding=pad)
    target_sq_mean = F.conv2d(target_pos ** 2, kernel, padding=pad)
    
    pred_var = pred_sq_mean - pred_mean ** 2
    target_var = target_sq_mean - target_mean ** 2
    
    # Clamp variance to avoid negative values due to numerical issues
    pred_var = torch.clamp(pred_var, min=1e-6)
    target_var = torch.clamp(target_var, min=1e-6)
    
    # Compute Equivalent Number of Looks (ENL): mean² / variance
    # Higher ENL = more smoothed/averaged, lower = more speckly
    pred_enl = pred_mean ** 2 / pred_var
    target_enl = target_mean ** 2 / target_var
    
    # Clamp ENL to reasonable range to avoid outliers
    pred_enl = torch.clamp(pred_enl, max=100.0)
    target_enl = torch.clamp(target_enl, max=100.0)
    
    enl_loss = F.l1_loss(pred_enl, target_enl)
    
    # Compute Coefficient of Variation (CV): std / mean
    pred_cv = torch.sqrt(pred_var) / pred_mean
    target_cv = torch.sqrt(target_var) / target_mean
    
    cv_loss = F.l1_loss(pred_cv, target_cv)
    
    return enl_loss + cv_loss


if __name__ == '__main__':
    output = torch.rand(4, 1, 16, 16)
    target = torch.rand(4, 1, 16, 16)

    loss = lratio_loss(output, target)
    print(f"Ratio Loss: {loss}, shape: {loss.shape}")

    loss_l1 = l1_loss(output, target)
    print(f"L1 Loss: {loss_l1}")
    
    # Test new losses
    freq_loss = frequency_domain_loss(output, target)
    print(f"Frequency Domain Loss: {freq_loss}")
    
    speckle_loss = speckle_aware_loss(output, target)
    print(f"Speckle Aware Loss: {speckle_loss}")
    