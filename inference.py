import argparse
import yaml
import torch
from src import ISSM_SAR
from src import psnr_torch, ssim_torch
import os
from torch.optim import Adam
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image
import numpy as np
from pathlib import Path
import sys

def load_config(config_path: str) -> dict:
    try: 
        with open(Path(config_path), 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f'Invalid yaml inference config file: {e}')
        sys.exit(1)


def load_image2tensor(image_path: str, transform: Compose) -> torch.Tensor:
    try:
        image = Image.open(Path(image_path)).convert('L') # HWC [0, 255]
        # transform
        transformed_img = transform(image) # [C, H, W] [-1, 1]
        return transformed_img.unsqueeze(0) # [1, C, H, W] [-1, 1]
    except Exception as e:
        print(f'Can open image at {image_path}: {e}')
        sys.exit(1)
    

def load_device(device_name: str):
    if not device_name:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            device = torch.device(device_name)
        except Exception as e:
            print(f'Error explain:\nUse device: {device_name} by connfigure in yaml inference config\n{e}')
            sys.exit(1)
    print(f'{30 * '-'}Use: {device}{30 * '-'}')
    return device


if __name__ == '__main__':
    #----------Argument parser------------
    parser = argparse.ArgumentParser(description='Inference SR Sar')
    parser.add_argument('--config_path', type=str, default= '/mnt/data1tb/vinh/ISSM-SAR/config/inference_config.yaml', help='Path to the YAML config file')
    args = parser.parse_args()

    #---------Check config_path---------
    if not os.path.exists(Path(args.config_path)):
        raise FileNotFoundError(f'Not found config file at {Path(args.config_path)}')
    

    #-----------Load config----------
    config = load_config(Path(args.config_path))
    model_cfg = config['model']
    ckpt_path = config['ckpt_path']
    device_name = config['device_name']
    cfg_input = config['input']
    cfg_output = config['output']
    cfg_norm = config['normalize']

    #--------Check checkpoint_path--------
    if not os.path.exists(Path(ckpt_path)):
        raise FileNotFoundError(f'Not found checkpoint file at {Path(args.checkpoint_path)}')

    #--------Init Transform------------
    transform = Compose(    
        transforms=[
            ToTensor(),
            Normalize(mean=[cfg_norm['mean']], std=[cfg_norm['std']])
        ]
    )

    #-------Init device--------------
    device = load_device(device_name)


    #---------Init model------------
    model = ISSM_SAR(config=model_cfg).to(device)
    checkpoint = torch.load(Path(ckpt_path))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


#----------------MAIN PROCESS------------------
    
    #-------Get the path of inputs------
    s1t1_path = cfg_input['s1t1_path']
    print(s1t1_path)
    s1t2_path = cfg_input['s1t2_path']
    print(s1t2_path)
    
    #--------Transform to Tensor: 1CHW & [-1, 1]---------
    transformed_s1t1 = load_image2tensor(s1t1_path, transform).to(device)
    transformed_s1t2 = load_image2tensor(s1t2_path, transform).to(device)


    #---------------Infer SR------------------
    with torch.no_grad():
        sr_up, sr_down = model(transformed_s1t1, transformed_s1t2)
    

    #----------------Denorm sr to [0, 255]-------------------
    sr_fusion = 0.5 * sr_up[-1] + 0.5 * sr_down[-1]
    sr_fusion = sr_fusion.squeeze(0).squeeze(0)  # Remove batch and channel dims -> (H, W)
    sr_fusion = (sr_fusion * cfg_norm['std'] + cfg_norm['mean']).clamp(0, 1)
    output_image = (sr_fusion.cpu() * 255).clamp(0, 255).byte().numpy()  # Convert to uint8
    output_image = Image.fromarray(output_image)
    output_image.show()




    