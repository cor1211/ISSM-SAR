import torch
import torch.nn as nn
from torch import squeeze, transpose, unsqueeze
from pytorch_wavelets import DWTForward, DWTInverse
import math
from PIL import Image 
from torchvision.transforms import Resize, InterpolationMode

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
class Deconv2d(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.deconv2d = nn.ConvTranspose2d(in_channels=in_c, out_channels=1, kernel_size=3, padding=5, stride=2, output_padding=0)
    
    def forward(self, x):
        x = self.deconv2d(x)
        return x

if __name__ == '__main__':
    # input = torch.rand(16, 1, 256,256)
    # print(f"Input size: {input.shape}")
    # layer = MultiLooks(1)
    # pfe_block = PFE(in_channels=1)
    # out = pfe_block(input)
    # print(f'Output size: {out.shape}')

    input1 = torch.rand(1, 1, 2, 2)
    print(f"Input size: {input1.shape}")
    deconv2d = Deconv2d(1)
    out = deconv2d(input1)    
    print(f'Output size: {out.shape}')
    print(out)
