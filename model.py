import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import squeeze, transpose, unsqueeze
from pytorch_wavelets import DWTForward, DWTInverse
import math
from PIL import Image 
from torchvision.transforms import Resize,ToTensor, InterpolationMode
from torchvision.ops import DeformConv2d
# Khoi 3 CNN
class MultiLooks(nn.Module):
    def __init__(self, in_channel:int = 1):
        super().__init__()
        self.three_cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, padding=3//2, stride=1),
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=5, padding=5//2, stride=1),
            nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=7, padding=7//2, stride=1),
        )

    def forward(self, x):
        out_cnns = []
        for layer in self.three_cnn:
            out_cnns.append(layer(x))
        return torch.concat(out_cnns, dim=1)


# Khoi tach biet tan so cao/thap
class HighFrequencyExtractor(nn.Module):
    def __init__(self, wavelet = 'haar', mode = 'zero'):
        super().__init__()
        self.dwt = DWTForward(J = 1, wave= wavelet, mode= mode)
        self.idwt = DWTInverse(wave=wavelet, mode = mode)

    def forward(self, x):
        Yl, Yh = self.dwt(x)
        Yl_zeros = torch.zeros_like(Yl)

        high_freq_img = self.idwt((Yl_zeros, Yh))
        return high_freq_img

    
class ECA(nn.Module):
    def __init__(self,in_c:int, gamma:int =2, b : int = 1):
        super().__init__()

        t = int(abs((math.log(in_c, 2)+b)/gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.ac_func = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.ac_func(y)
        # print(y.shape)
        return x * y.expand_as(x)

# Main block
class PFE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.first_way = MultiLooks(in_channel=in_channels) # Out = [N, 112, H, W]

        self.second_way = nn.Sequential(
            # nn.Upsample(scale_factor=0.5, mode='bicubic', align_corners=False), # Out = [H//2, W//2]
            MultiLooks(in_channel=in_channels), # Out = [N, 112, H//2, W//2]
            # nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False), # Out = [H, W]

        )   
        self.third_way = nn.Sequential(
            HighFrequencyExtractor(), # Origin dimens
            MultiLooks(in_channel=in_channels) # Out = [N, 112, H, W]  
        )

        self.pfe = nn.Sequential(self.first_way, self.second_way, self.third_way)
        
    def forward(self, x):
        outs_way = []
        for idx, way in enumerate(self.pfe):
            # print(idx, way)
            if idx == 1:
                N, C, H, W = x.size()
                # print(1, N, C, H, W)    
                out_second_way = Resize(size = (H//2, W//2), interpolation=InterpolationMode.BICUBIC)(x)
                # print(2, x.size())
                out_second_way = way(out_second_way)
                # print(3, x.size())
                out_second_way = Resize(size = (H, W), interpolation=InterpolationMode.BICUBIC)(out_second_way)
                # print(4, x.size())
                outs_way.append(out_second_way)
            else:
                outs_way.append(way(x))

        after_relu =  nn.ReLU(True)(torch.concat(outs_way, dim=1))
        return ECA(in_c=after_relu.size()[1])(after_relu)

# Deconv2d Block
class DeConv2d(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.deconv2d = nn.ConvTranspose2d(in_channels=in_c, out_channels=1, kernel_size=3, padding=0, stride=2, output_padding=1)
    
    def forward(self, x):
        x = self.deconv2d(x)
        return x

class DeformConv2dBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_c, 2*3*3, 3, 1, 0)
        self.deform_conv = DeformConv2d(in_c, 1, 3, 1, 0)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        return x

class Deconv_DeformBlock(nn.Module):
    def __init__(self, in_c, kernel_size):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_c, out_channels=64, kernel_size=kernel_size, stride=1, padding=1)
        self.deformconv = DeformConv2d(in_channels=64, out_channels=in_c, kernel_size=kernel_size, padding=kernel_size//2)
        self.offset_conv = nn.Conv2d(in_channels=64, out_channels=2 * kernel_size*kernel_size, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x): # In = [N, 336, H, W]
        out_block = self.deconv(x)  # Out = [N, 64, H', W'] (H' = H, W' = W)
        out_block = nn.ReLU(True)(out_block) # Out = [N, 64, H', W']
        offset = self.offset_conv(out_block) # Out_offset = [N, 2*k*k, H', W']
        out_block = self.deformconv(out_block, offset) # Out = [N, in_c, H', W']
        out_block = nn.ReLU(True)(out_block) # Out = [N, in_c, H', W']
        return out_block            
            
class HFE(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.in_block = Deconv_DeformBlock(in_c=in_c, kernel_size=kernel_size)
        self.blocks = nn.ModuleList()
        for _ in range(3):
            self.blocks.append(Deconv_DeformBlock(in_c=in_c, kernel_size=kernel_size))
        
        self.offset_conv = nn.Conv2d(in_channels=in_c + in_c, out_channels=2 * kernel_size*kernel_size, kernel_size=kernel_size, padding=kernel_size//2)
        self.out_deformConv = DeformConv2d(in_channels=in_c + in_c, out_channels=out_c, kernel_size=kernel_size, padding=kernel_size//2) # Note: last block have input concat with in_module, so in_channels = 2*in_c

    def forward(self, x):
        in_module = x # In = [N, 336, H, W]
        x = self.in_block(x) # Out = [N, 336, H, W]
        in_previous_block = torch.zeros_like(x) # [N, 336, H, W]
        
        for block in self.blocks:
            in_cur_block = x + in_previous_block
            x = block(in_cur_block) # Out = [N, 336, H, W]
            in_previous_block = in_cur_block
        x_concat = torch.concat((x, in_module), dim=1)
        offset = self.offset_conv(x_concat)
        return self.out_deformConv(x_concat, offset)

class REC(nn.Module): # In = [N, out_HFE, H, W]
    def __init__(self, in_c, out_c):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_c, out_channels= 64, kernel_size=3, stride=2, padding=1, output_padding=1) # Out = [N, 64, 2H, 2W]
        self.conv = nn.Conv2d(in_channels=64, out_channels=out_c, kernel_size=3, padding=1) # Out = [N, 1, 2H, 2W]
    
    def forward(self, x):
        x = self.deconv(x) # Out = [N, 64, 2H, 2W]
        x = self.conv(x) # Out = [N, 1, 2H, 2W]
        return x
    

# class ISSM_SAR(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super().__init__()
        
         
if __name__ == '__main__':
    # input = torch.rand(16, 1, 256,256)
    # print(f"Input size: {input.shape}")
    # layer = MultiLooks(1)
    # pfe_block = PFE(in_channels=1)
    # out = pfe_block(input)
    # print(f'Output size: {out.shape}')

    # input1 = torch.rand(1, 1, 2, 2)
    # print(f"Input size: {input1.shape}")
    # deconv2d = DeConv2d(1)
    # out = deconv2d(input1)    
    # print(f'Output size: {out.shape}')
    # print(out)

    # input2 = torch.rand(1,1,16,16)
    # print(f"Input size: {input2.shape}")
    # deformconv2d = DeformConv2dBlock(1)
    # out = deformconv2d(input2)    
    # print(f'Output size: {out.shape}')
    # print(out)

    # input3 = torch.rand(1,336,256,256)
    # print(f"Input size: {input3.shape}")
    # trans_deform_block = Deconv_DeformBlock(in_c=336, out_c=336, kernel_size=3)
    # out = trans_deform_block(input3)    
    # print(f'Output size: {out[0].shape}, {out[1].shape}')

    input = torch.rand(4, 1, 16,16)
    print(f"Input size: {input.shape}")

    pfe = PFE(in_channels=1)
    hfe = HFE(in_c=336, out_c=32, kernel_size=3)
    rec = REC(in_c=32, out_c=1)

    
    new_size = (input.shape[2]*2, input.shape[3]*2)
    f1 = pfe(input) # Out = [N, 336, H, W]
    print(f"F1 size: {f1.shape}")
    h1 = hfe(f1) # Out = [N, 32, H, W]
    print(f"h1 size: {h1.shape}")   
    after_REC = rec(h1) # Out = [N, 1, 2H, 2W]
    print(f"after_REC size: {after_REC.shape}")

    umsampled_tensor = F.interpolate(input, scale_factor=2, align_corners=False, mode='bilinear')
    # Skip-connection
    after_sum = after_REC + umsampled_tensor
    print(after_sum.shape) # Out = [N, 1, 2H, 2W]

    # print(f'Output size: {out.shape}')