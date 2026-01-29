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


if __name__ == '__main__':
    output = torch.rand(4, 1, 16, 16)
    target= torch.rand(4, 1, 16, 16)

    loss = lratio_loss(output, target)
    print(loss, loss.shape)

    loss_l1 = l1_loss(output, target)
    print(loss_l1)
    