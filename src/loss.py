import torch
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

def frequency_domain_loss(pred: torch.Tensor, target: torch.Tensor, 
                          high_freq_weight: float = 1.0) -> torch.Tensor:
    """
    Frequency-domain loss để preserve high-frequency components (texture, speckle).
    
    Cách hoạt động:
    - Chuyển pred và target sang frequency domain qua FFT
    - So sánh magnitude spectrum (chứa info về texture)
    - Emphasis vào high-frequency bands (outer region of spectrum)
    
    Args:
        pred: Predicted SR image [B, C, H, W]
        target: Ground truth HR image [B, C, H, W]
        high_freq_weight: Weight cho high-frequency component loss
    
    Returns:
        Combined frequency loss
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
    