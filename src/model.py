import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import squeeze, transpose, unsqueeze
from pytorch_wavelets import DWTForward, DWTInverse
import math
from PIL import Image 
from torchvision.transforms import Resize,ToTensor, InterpolationMode
from torchvision.ops import DeformConv2d
from torchinfo import summary
import yaml
import argparse

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Khoi 3 CNN
class MultiLooks(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.three_cnn = nn.ModuleList()
        self.three_cnn.extend(
            [       
                ConvBlock(in_c=in_channel, out_c= out_channel//3, kernel_size=3, padding=3//2, stride=1, act_type='relu'),
                ConvBlock(in_c=in_channel, out_c= out_channel//3, kernel_size=5, padding=5//2, stride=1, act_type='relu'),
                ConvBlock(in_c=in_channel, out_c= out_channel//3, kernel_size=7, padding=7//2, stride=1, act_type='relu'),
            ]
        )

    def forward(self, x):
        out_cnns = []
        for layer in self.three_cnn:
            out_cnns.append(layer(x))
        return torch.concat(out_cnns, dim=1)


# Khoi tach biet tan so cao/thap
class HighFrequencyExtractor(nn.Module):
    def __init__(self, wavelet = 'haar', mode = 'reflect'):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.first_way = MultiLooks(in_channel=in_channels, out_channel=int(out_channels//3)) # Out = [N, 112, H, W]

        self.second_way = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False), # Out = [H//2, W//2]
            MultiLooks(in_channel=in_channels,out_channel=int(out_channels//3)), # Out = [N, 112, H//2, W//2]
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # Out = [H, W]

        )   
        self.third_way = nn.Sequential(
            HighFrequencyExtractor(), # Origin dimens
            MultiLooks(in_channel=in_channels,out_channel=int(out_channels//3)) # Out = [N, 112, H, W]  
        )

        self.pfe = nn.ModuleList()
        self.pfe.extend([self.first_way, self.second_way, self.third_way])

        self.compress_out = ConvBlock(in_c=out_channels, out_c=out_channels, kernel_size=3, padding=1, stride=1, act_type='relu')
        self.eca = ECA(out_channels)

    def forward(self, x):
        outs_way = []
        for way in self.pfe:
            # print(idx, way)
            outs_way.append(way(x))

        # after_relu =  nn.ReLU(True)(torch.concat(outs_way, dim=1))

        return self.eca(self.compress_out(torch.concat(outs_way, dim=1)))
        # return self.eca(after_relu)
     

class DeFormBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padding, stride, act_type):
        super().__init__()
        self.deform = DeformConv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.offset_conv = nn.Conv2d(in_channels=in_c, out_channels=2*kernel_size*kernel_size, padding=padding, stride=stride, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(num_features=out_c)
        self.act = ''
        if act_type.lower() =='prelu':
            self.act = nn.PReLU(num_parameters=1, init=0.2)
        elif act_type.lower() == 'relu':
            self.act = nn.ReLU(True)
        
    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform(x, offset)
        x = self.bn(x)
        return self.act(x) if self.act else x
              

class DeFormBlockv2(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padding, stride, act_type):
        super().__init__()
        self.deform = DeformConv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.offset_mask_conv = nn.Conv2d(in_channels=in_c, out_channels=3*kernel_size*kernel_size, padding=padding, stride=stride, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(num_features=out_c)
        self.act = ''
        if act_type.lower() =='prelu':
            self.act = nn.PReLU(num_parameters=1, init=0.2)
        elif act_type.lower() == 'relu':
            self.act = nn.ReLU(True)
        
    def forward(self, x):
        out = self.offset_mask_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1) # Split offset and mask
        offset = torch.concat((o1, o2), dim=1) # Offset
        mask = torch.sigmoid(mask)
        x = self.deform(x, offset, mask)
        x = self.bn(x)
        return self.act(x) if self.act else x
      

class HFE(nn.Module): # Fix out_c = in_c
    def __init__(self, in_c, out_c, kernel_size, padding, stride, num_blocks):
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                DeConvBlock(in_c=in_c, out_c=in_c, kernel_size=kernel_size, padding=padding, stride=stride, act_type='relu'),
                # DeFormBlock(in_c=in_c, out_c=in_c, kernel_size=kernel_size, padding=padding, stride = stride, act_type='relu'),
                DeFormBlockv2(in_c=in_c, out_c=in_c, kernel_size=kernel_size, padding=padding, stride = stride, act_type='relu')
            ))

        self.down_channel_blocks = nn.ModuleList()
        for _ in range(num_blocks-2):
            self.down_channel_blocks.append(ConvBlock(in_c=2*in_c, out_c=in_c, kernel_size=1, stride=1, padding=0, act_type='relu'))

        self.last_deform = DeFormBlockv2(in_c=2*in_c, out_c=out_c, kernel_size=3, padding=1, stride=1, act_type='relu') # Not change H, W
        # self.last_deform = DeFormBlock(in_c=2*in_c, out_c=out_c, kernel_size=3, padding=1, stride=1, act_type='relu') # Not change H, W

    def forward(self, x):
        in_module = x # In = [N, 336, H, W]
        outs = []
        for idx in range(self.num_blocks):
            just_out = self.blocks[idx](x)
            outs.append(just_out)
            if idx > 0 and idx != self.num_blocks-1:
                x = torch.concat((just_out, outs[idx-1]), dim = 1) # Concat
                x = self.down_channel_blocks[idx-1](x) # Down_channel after concat
            else: 
                x = just_out

        x_concat = torch.concat((x, in_module), dim=1)
        return self.last_deform(x_concat)


class REC(nn.Module): # In = [N, out_HFE, H, W] | Out = [N, 1, 2H, 2W]
    def __init__(self, in_c, out_c):
        super().__init__()
        self.rec = nn.Sequential(
            DeConvBlock(in_c=in_c, out_c=in_c, kernel_size=6, padding=2, stride=2, act_type='prelu'),
            ConvBlock(in_c=in_c, out_c=out_c, kernel_size=3, padding=1, stride = 1, act_type='none')
        )
    def forward(self, x):
        x = self.rec(x)
        return x            


#--------- IFS's Blocks ----------
class DeConvBlock(nn.Module): # Keep Channel constance, up-sample H, W follow scale_factor x2
    def __init__(self, in_c, out_c, kernel_size, padding, stride, act_type='prelu'):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_c),
        )
        if act_type.lower() =='prelu':
            self.deconv.append(nn.PReLU(num_parameters=1, init=0.2))
        elif act_type.lower() == 'relu':
            self.deconv.append(nn.ReLU(True))

    def forward(self, x):
        x = self.deconv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padding, stride, act_type='prelu'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_c),
        )
        if act_type.lower() =='prelu':
            self.conv.append(nn.PReLU(num_parameters=1, init=0.2))
        elif act_type.lower() == 'relu':
            self.conv.append(nn.ReLU(True))

    def forward(self, x):
        x = self.conv(x)
        return x


# ----- Build IFS Block ------
class IFS(nn.Module):
    def __init__(self, in_c, out_c, num_groups, kernel_size, padding, stride):
        super().__init__()
        self.compress_in = ConvBlock(in_c=3*in_c, out_c = in_c, kernel_size=1, padding=0, stride=1, act_type='prelu')
        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        self.re_guide = ConvBlock(in_c=2*in_c, out_c=in_c, kernel_size=1, padding=0, stride=1, act_type='prelu')

        self.num_groups = num_groups
        for idx in range(self.num_groups):
            self.upBlocks.append(
                DeConvBlock(in_c=in_c, out_c=in_c, kernel_size=kernel_size, padding=padding, stride=stride, act_type='prelu')
            )

            self.downBlocks.append(
                ConvBlock(in_c=in_c, out_c=in_c, kernel_size=kernel_size, padding=padding, stride=stride, act_type='prelu')
                # DeFormBlockv2(in_c=in_c, out_c=in_c, kernel_size=kernel_size, padding=padding, stride= stride, act_type='prelu')
                # DeFormBlock(in_c=in_c, out_c=in_c, kernel_size=kernel_size, padding=padding, stride= stride, act_type='prelu')
            )

            if idx > 0: # Add conv 1x1
                self.uptranBlocks.append(
                    ConvBlock(in_c=(idx+1)*in_c, out_c=in_c, kernel_size=1, stride=1, padding=0, act_type='prelu')
                )

                self.downtranBlocks.append(
                    ConvBlock(in_c=(idx+1)*in_c, out_c=in_c, kernel_size=1, stride=1, padding=0, act_type='prelu')
                )
        
        self.compress_out = ConvBlock(in_c=num_groups*in_c, out_c=out_c, kernel_size=1, padding=0, stride=1, act_type='prelu')

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


class ISSM_SAR(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # Load config
        self.num_ifs = config['num_ifs']
        pfe_cfg = config['pfe']
        hfe_cfg = config['hfe']
        rec_cfg = config['rec']
        ifs_cfg = config['ifs']

        # PFE Modules
        self.pfe_up = PFE(in_channels=pfe_cfg['in_channels'], out_channels=pfe_cfg['out_channels'])
        self.pfe_down = PFE(in_channels=pfe_cfg['in_channels'], out_channels=pfe_cfg['out_channels'])

        # HFE Modules
        self.hfe_up = HFE(in_c=hfe_cfg['in_c'], out_c=hfe_cfg['out_c'], kernel_size=hfe_cfg['kernel_size'], padding=hfe_cfg['padding'], stride = hfe_cfg['stride'], num_blocks=hfe_cfg['num_blocks'])
        self.hfe_down = HFE(in_c=hfe_cfg['in_c'], out_c=hfe_cfg['out_c'], kernel_size=hfe_cfg['kernel_size'], padding=hfe_cfg['padding'], stride = hfe_cfg['stride'], num_blocks=hfe_cfg['num_blocks'])
        
        # FFB Modules
        self.ffb_up = nn.ModuleList()
        self.ffb_down = nn.ModuleList()

        # REC Modules
        self.rec_up = nn.ModuleList()
        self.rec_down = nn.ModuleList()

        for _ in range(self.num_ifs):
            self.ffb_up.append(IFS(in_c=ifs_cfg['in_c'], out_c=ifs_cfg['out_c'], num_groups=ifs_cfg['num_groups'], kernel_size=ifs_cfg['kernel_size'], padding=ifs_cfg['padding'], stride=ifs_cfg['stride']))
            self.ffb_down.append(IFS(in_c=ifs_cfg['in_c'], out_c=ifs_cfg['out_c'], num_groups=ifs_cfg['num_groups'], kernel_size=ifs_cfg['kernel_size'], padding=ifs_cfg['padding'], stride = ifs_cfg['stride']))
        
        for _ in range(self.num_ifs + 1):
            self.rec_up.append(REC(in_c=rec_cfg['in_c'], out_c=rec_cfg['out_c']))
            self.rec_down.append(REC(in_c=rec_cfg['in_c'], out_c=rec_cfg['out_c']))
    
    def forward(self, in_first_time, in_second_time):
        # Upsample 2 inputs
        in_ft_upsampled = F.interpolate(in_first_time,scale_factor=2, mode='bilinear', align_corners=False)
        in_se_upsampled = F.interpolate(in_second_time,scale_factor=2, mode='bilinear', align_corners=False)

        # Pass throught PFE
        f1 = self.pfe_up(in_first_time)
        f2 = self.pfe_down(in_second_time)

        # Pass throught HFE
        h1 = self.hfe_up(f1)
        h2 = self.hfe_down(f2)

        # Pass throught FFB (each IFSs)
        in_ifs_up = []
        in_ifs_down = []
        in_ifs_up.append(h1)
        in_ifs_down.append(h2)

        # Compute out_ifs
        for idx in range(self.num_ifs):
            out_ifs_up = self.ffb_up[idx](f1, in_ifs_up[idx], in_ifs_down[idx])
            out_ifs_down = self.ffb_down[idx](f2, in_ifs_down[idx], in_ifs_up[idx])

            in_ifs_up.append(out_ifs_up)
            in_ifs_down.append(out_ifs_down)

        # Compute sr (after REC and sum)
        sr_up = []
        sr_down = []
        for idx in range(self.num_ifs+1):
            sr_up.append(torch.clamp((self.rec_up[idx](in_ifs_up[idx]) + in_ft_upsampled), -1.0, 1.0))# Out (-1, 1)
            sr_down.append(torch.clamp((self.rec_down[idx](in_ifs_down[idx]) + in_se_upsampled), -1.0, 1.0))

        
        return sr_up, sr_down

        
         
if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='Train ISSM-SAR Model')
    parser.add_argument('--config_path', type=str, default= '/mnt/data1tb/vinh/ISSM-SAR/config/base_config.yaml', help='Path to the YAML config file')
    parser.add_argument('--checkpoint_path', type = str, default=None, help='Path to checkpoint to resume training')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)
    model_cfg = config['model']


    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU {torch.cuda.get_device_name(device)}')
    else:
        device = torch.device('cpu')
        print(f'No GPU. Using CPU instead')

    
    issm_sar = ISSM_SAR(config=model_cfg).to(device)
    input_first_time = torch.rand(1, 1, 128,128).to(device)
    input_second_time = torch.rand(1, 1, 128,128).to(device)

    sr_up, sr_down = issm_sar(input_first_time, input_second_time)
    
    for idx in range(4):
        print(sr_up[idx].shape, sr_down[idx].shape)
    
    summary(issm_sar, input_size=[(1,1, 128,128),(1,1,128,128)])
    
