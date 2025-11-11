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

def load_config(config_path: str) -> dict:
    try: 
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except:
        print('Invalid yaml config file')
        exit()

def load_image_tensor(image_path: str, transform: Compose) -> torch.Tensor:
    try:
        image = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f'Error: Not found image at {image_path}')
        exit()
    
    # transform
    transformed_img = transform(image) # [C, H, W]
    return transformed_img.unsqueeze(0) # [1, C, H, W]

def load_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU {torch.cuda.get_device_name("cuda")}')
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
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Check config_path
    if not os.path.exists(args.config_path):
        print('Config path not exist. End!')
        exit()

    # Load config
    config = load_config(args.config_path)
    data_cfg = config['data']
    train_cfg = config['train']
    model_cfg = config['model']

    # Check checkpoint_path
    if not os.path.exists(args.checkpoint_path):
        print('Checkpoint path not exist. End!')
        exit()

    # Check cuda avalable
    if args.device == 'cuda':
        device = torch.device('cuda')
        print(f'Using GPU {torch.cuda.get_device_name("cuda")}')
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
            Normalize(mean=0.5, std=0.5)
        ]
    )

    # Transforming image to tensor [-1, 1]
    img_path_1 = args.image_path_1
    img_path_2 = img_path_1.replace('time_1', 'time_2')
    transformed_img_1 = load_image_tensor(img_path_1, transform).to(device)
    transformed_img_2 = load_image_tensor(img_path_1, transform).to(device)

    # Init model
    model = ISSM_SAR(config=model_cfg).to(device)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Infer SR
    with torch.no_grad():
        sr_up, sr_down = model(transformed_img_1, transformed_img_2)
    
    # Transforming sr to [0, 255]
    sr_fusion = 0.5 * sr_up[-1] + 0.5 * sr_down[-1]
    sr_fusion = sr_fusion.squeeze(0).squeeze(0)  # Remove batch and channel dims -> (H, W)
    sr_fusion = (sr_fusion * 0.5 + 0.5).clamp(0, 1)
    output_image = (sr_fusion.cpu() * 255).clamp(0, 255).byte().numpy()  # Convert to uint8
    output_image = Image.fromarray(output_image)
    output_image.show()
    # print(sr_fusion.shape)

    