import torch
import numpy as np
import torch.nn.functional as F


def lratio_loss(output: torch.Tensor,target: torch.Tensor,eps: float = 1e-6,) -> torch.Tensor:

    sum_output = torch.sum(output, dim=(1, 2, 3)) #(B, )
    # print(sum_output.shape)
    sum_target = torch.sum(target, dim=(1, 2, 3)) #(B, )
    ratios = sum_output/ (sum_target + eps)
    loss_per_img = torch.abs(1-ratios)
    loss = torch.mean(loss_per_img)
    return loss

def l1_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(output, target)

if __name__ == '__main__':
    output = torch.rand(4, 1, 16, 16)
    target= torch.rand(4, 1, 16, 16)

    loss = lratio_loss(output, target)
    print(loss, loss.shape)

    loss_l1 = l1_loss(output, target)
    print(loss_l1)
    