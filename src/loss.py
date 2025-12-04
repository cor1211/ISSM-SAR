import torch
import numpy as np
import torch.nn.functional as F


def lratio_loss(output: torch.Tensor,target: torch.Tensor,eps: float = 1e-6,) -> torch.Tensor:

    sum_output = torch.sum(output, dim=(1, 2, 3)) #(B, )
    # print(sum_output.shape)
    sum_target = torch.sum(target, dim=(1, 2, 3)) #(B, )
    ratios = sum_output/ (sum_target + eps)
    loss_per_img = torch.abs(1-ratios)
    # print(loss_per_img.shape)
    loss = torch.mean(loss_per_img)
    return loss

def l1_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(output, target)


def gradient_loss(pred, target):
    # Tính đạo hàm theo phương ngang (x) và dọc (y)
    # pred shape: (B, C, H, W)
    
    # Gradient X: chênh lệch giữa cột sau và cột trước
    pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    
    # Gradient Y: chênh lệch giữa hàng dưới và hàng trên
    pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
    
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
    