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
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f'Invalid yaml config file: {e}')
        sys.exit(1)


def load_image_tensor(image_path: str, transform: Compose) -> torch.Tensor:
    try:
        image = Image.open(Path(image_path)).convert('L')
        # transform
        transformed_img = transform(image) # [C, H, W]
        return transformed_img.unsqueeze(0) # [1, C, H, W]
    except Exception as e:
        print(f'Can open image at {image_path}: {e}')
        sys.exit(1)
    
    


def load_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f'Using GPU {torch.cuda.get_device_name("cuda:0")}')
    else:
        device = torch.device('cpu')
        print('No GPU, using CPU instead')
    return device


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='Inference SR Sar')
    parser.add_argument('--image_path_1', type=str, default='/mnt/data1tb/vinh/ISSM-SAR/dataset/test/1_0_F_48_69_C_c_1_time_1.png')
    parser.add_argument('--config_path', type=str, default= '/mnt/data1tb/vinh/ISSM-SAR/config/base_config.yaml', help='Path to the YAML config file')
    parser.add_argument('--checkpoint_path', type=str, default='/mnt/data1tb/vinh/ISSM-SAR/checkpoints/exp_20251110-135418/best.pth')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='device to run')
    args = parser.parse_args()

    # Check config_path
    if not os.path.exists(Path(args.config_path)):
        raise FileNotFoundError(f'Not found config file at {Path(args.config_path)}')
    

    # Load config
    config = load_config(Path(args.config_path))
    data_cfg = config['data']
    train_cfg = config['train']
    model_cfg = config['model']

    # Check checkpoint_path
    if not os.path.exists(Path(args.checkpoint_path)):
        raise FileNotFoundError(f'Not found checkpoint file at {Path(args.checkpoint_path)}')


    # Check cuda avalable
    if args.device == 'cuda':
        device = torch.device('cuda:0')
        print(f'Using GPU {torch.cuda.get_device_name("cuda:0")}')
    elif args.device == 'cpu':
        device = torch.device('cpu')
        print('No GPU, using CPU instead')
    else:
        device = load_device()

# --------------- Process input ------------------
    # Init transform compose
    transform = Compose(
        transforms=[
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5])
        ]
    )

    # Transforming image to tensor [-1, 1]
    img_path_1 = args.image_path_1
    print(img_path_1)
    # Image.open(img_path_1).show()
    img_path_2 = img_path_1.replace('time_1', 'time_2')
    print(img_path_2)
    # Image.open(img_path_2).show()
    transformed_img_1 = load_image_tensor(img_path_1, transform).to(device)
    transformed_img_2 = load_image_tensor(img_path_2, transform).to(device)

    # Init model
    model = ISSM_SAR(config=model_cfg).to(device)
    checkpoint = torch.load(Path(args.checkpoint_path))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # -------------- Infer SR ------------------
    with torch.no_grad():
        sr_up, sr_down = model(transformed_img_1, transformed_img_2)
    
    # ---------------- Transforming sr to [0, 255] -------------------
    sr_fusion = 0.5 * sr_up[0] + 0.5 * sr_down[0]
    sr_fusion = sr_fusion.squeeze(0).squeeze(0)  # Remove batch and channel dims -> (H, W)
    sr_fusion = (sr_fusion * 0.5 + 0.5).clamp(0, 1)
    output_image = (sr_fusion.cpu() * 255).clamp(0, 255).byte().numpy()  # Convert to uint8
    output_image = Image.fromarray(output_image)
    output_image.show()


    