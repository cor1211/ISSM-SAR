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
        self.three_cnn = nn.ModuleList()
        self.three_cnn.extend([nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, padding=3//2, stride=1),
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=5, padding=5//2, stride=1),
            nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=7, padding=7//2, stride=1),]
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
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False), # Out = [H//2, W//2]
            MultiLooks(in_channel=in_channels), # Out = [N, 112, H//2, W//2]
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # Out = [H, W]

        )   
        self.third_way = nn.Sequential(
            HighFrequencyExtractor(), # Origin dimens
            MultiLooks(in_channel=in_channels) # Out = [N, 112, H, W]  
        )

        self.pfe = nn.Sequential(self.first_way, self.second_way, self.third_way)
        
    def forward(self, x):
        outs_way = []
        for way in self.pfe:
            # print(idx, way)
            outs_way.append(way(x))

        after_relu =  nn.ReLU(True)(torch.concat(outs_way, dim=1))
        return ECA(in_c=after_relu.size()[1])(after_relu)

# Deconv2d Block
# class DeConv2d(nn.Module):
#     def __init__(self, in_c):
#         super().__init__()
#         self.deconv2d = nn.ConvTranspose2d(in_channels=in_c, out_channels=1, kernel_size=3, padding=0, stride=2, output_padding=1)
    
#     def forward(self, x):
#         x = self.deconv2d(x)
#         return x

# class DeformConv2dBlock(nn.Module):
#     def __init__(self, in_c):
#         super().__init__()
#         self.offset_conv = nn.Conv2d(in_c, 2*3*3, 3, 1, 0)
#         self.deform_conv = DeformConv2d(in_c, 1, 3, 1, 0)

#     def forward(self, x):
#         offset = self.offset_conv(x)
#         x = self.deform_conv(x, offset)
#         return x

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
            
class HFE(nn.Module): # Fix out_c = in_c
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

#--------- IFS's Blocks ----------
class DeConvBlock(nn.Module): # Keep Channel constance, up-sample H, W follow scale_factor x2
    def __init__(self, in_c, out_c, kernel_size, padding, stride):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.PReLU(num_parameters=1, init=0.2),
            nn.BatchNorm2d(num_features=out_c),
        )

    def forward(self, x):
        x = self.deconv(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padding, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.PReLU(num_parameters=1, init=0.2),
            nn.BatchNorm2d(num_features=out_c),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# ----- Build IFS Block ------
class IFS(nn.Module):
    def __init__(self, in_c, out_c, num_groups):
        super().__init__()
        self.compress_in = ConvBlock(in_c=3*in_c, out_c = in_c, kernel_size=1, padding=0, stride=1)
        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        self.re_guide = ConvBlock(in_c=2*in_c, out_c=in_c, kernel_size=1, padding=0, stride=1)

        self.num_groups = num_groups
        for idx in range(self.num_groups):
            self.upBlocks.append(
                DeConvBlock(in_c=in_c, out_c=in_c, kernel_size=6, padding=2, stride=2)
            )

            self.downBlocks.append(
                ConvBlock(in_c=in_c, out_c=in_c, kernel_size=6, padding=2, stride=2)
            )

            if idx > 0: # Add conv 1x1
                self.uptranBlocks.append(
                    ConvBlock(in_c=(idx+1)*in_c, out_c=in_c, kernel_size=1, stride=1, padding=0)
                )

                self.downtranBlocks.append(
                    ConvBlock(in_c=(idx+1)*in_c, out_c=in_c, kernel_size=1, stride=1, padding=0)
                )
        
        self.compress_out = ConvBlock(in_c=num_groups*in_c, out_c=out_c, kernel_size=1, padding=0, stride=1)

    def forward(self, f_in, h_same_way, h_other_way):
        x = torch.concat((f_in, h_same_way, h_other_way), dim=1) # Out = [N, 3*in_c, H, W]
        x = self.compress_in(x) # Out = [N, in_c, H, W]

        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LR_F = torch.concat(lr_features, dim=1) # idx 0: Out = [N, in_c, H, W]
            if idx > 0:
                LR_F = self.uptranBlocks[idx - 1](LR_F) # Out = [N, in_c, H, W]
            HR_F = self.upBlocks[idx](LR_F) # idx 0: Out = [N, in_c, 2H, 2W]
            hr_features.append(HR_F)

            HR_F = torch.concat(hr_features, dim=1)
            if idx > 0:
                HR_F = self.downtranBlocks[idx - 1](HR_F)
            LR_F = self.downBlocks[idx](HR_F)

            if idx == int(self.num_groups//2):
                LR_F = torch.concat((LR_F, h_other_way), dim=1)
                LR_F = self.re_guide(LR_F)
            
            lr_features.append(LR_F)
        
        del hr_features
        output = torch.concat(lr_features[1:], dim=1) # Out = [N, 3in_c, H, W]
        output = self.compress_out(output) # Out = [N, in_c, H, W]
        return output

# class ISSM_SAR(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super().__init__()
        
         
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU {torch.cuda.get_device_name(device)}')
    else:
        device = torch.device('cpu')
        print(f'No GPU. Using CPU instead')

    # Test forward with lr = [4, 1, 256, 256] and hr=[4, 1, 512, 512]
 
    input = torch.rand(4, 1, 16, 16)
    print(f'Input size: {input.shape}')
    # PFE modules
    pfe_up = PFE(in_channels=1)
    pfe_down = PFE(in_channels=1)
    # HFE modules
    hfe_up = HFE(in_c=336, out_c=336, kernel_size=3)
    hfe_down = HFE(in_c=336, out_c=336, kernel_size=3)
    # FFB Modules
    ffb_up = []
    ffb_down = []
    for _ in range(3):
        ffb_up.append(IFS(in_c=336, out_c=336, num_groups=3))
        ffb_down.append(IFS(in_c=336, out_c=336, num_groups=3))
    # REC modules
    rec_up = REC(in_c=336, out_c=1)
    rec_down = REC(in_c=336, out_c=1)

    # To device
    # pfe_up = pfe_up.to(device)
    # pfe_down = pfe_down.to(device)
    # hfe_up = hfe_up.to(device)
    # hfe_down = hfe_down.to(device)
    # ffb_up = [ifs.to(device) for ifs in ffb_up]
    # ffb_down = [ifs.to(device) for ifs in ffb_down]
    # rec_up = rec_up.to(device)
    # rec_down = rec_down.to(device)
    
    # Forward
    input_upsample = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)
    print(f'Input upsample size: {input_upsample.shape}')

    f1= pfe_up(input)
    print(f'F1 size: {f1.shape}')
    f2 = pfe_down(input)
    print(f'F2 size: {f2.shape}')

    h1 = hfe_up(f1)
    print(f'H1 size: {h1.shape}')

    h2 = hfe_down(f2)
    print(f'H2 size: {h2.shape}')

    for idx in range(3):
        out_up = ffb_up[idx](f1, h1, h2)
        out_down = ffb_up[idx](f2, h2, h1)
        
        h1 = out_up
        h2 = out_down
    
    i_up = rec_up(out_up) + input_upsample
    i_down = rec_down(out_down) + input_upsample

    i_out = 0.5*i_up + 0.5*i_down
    print(f'I_Out size: {i_out.shape}')

    
