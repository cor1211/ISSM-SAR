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
from tqdm import tqdm

def load_config(config_path: str) -> dict:
    try: 
        with open(Path(config_path), 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f'Invalid yaml inference config file: {e}')
        sys.exit(1)


def load_image2tensor(image_path: str, transform: Compose) -> torch.Tensor:
    p = Path(image_path)
    if p.suffix == '.npy':
        try:
            img_np = np.load(image_path).astype(np.float32)
            # Ensure shape [C, H, W]
            if img_np.ndim == 2:
                img_np = img_np[np.newaxis, ...]
            elif img_np.ndim == 3 and img_np.shape[2] <= 4: # HWC -> CHW
                img_np = img_np.transpose(2, 0, 1)
            return torch.from_numpy(img_np).unsqueeze(0) # [1, C, H, W]
        except Exception as e:
            print(f'Error loading npy file at {image_path}: {e}')
            sys.exit(1)

    try:
        image = Image.open(p).convert('L') # HWC [0, 255]
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
    print(f"{30 * '-'}Use: {device}{30 * '-'}")
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
    device_name = config.get('device', 'cuda')
    cfg_input = config['input']
    cfg_output = config['output']
    cfg_norm = config['normalize']
    limit = config.get('limit', None)

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
    checkpoint = torch.load(Path(ckpt_path), map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        # Support PyTorch Lightning checkpoints
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present (from LightningModule wrapper)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            # Ignore keys that don't start with 'model.' (discriminator, metrics, etc.)
        state_dict = new_state_dict
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()


#----------------MAIN PROCESS------------------
    
    #-------Get the path of inputs------
    s1t1_path = Path(cfg_input['s1t1_path'])
    s1t2_path = Path(cfg_input['s1t2_path'])
    save_path = cfg_output.get('save_path')

    print(f"S1T1 Input: {s1t1_path}")
    print(f"S1T2 Input: {s1t2_path}")

    # Check if inputs are directories (Batch Inference)
    if s1t1_path.is_dir() and s1t2_path.is_dir():
        print(f"{30*'-'} Batch Inference Mode {30*'-'}")
        
        if not save_path:
            print("Error: 'save_path' must be specified in config for batch inference.")
            sys.exit(1)
        
        os.makedirs(save_path, exist_ok=True)
        
        # Get image files
        valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.npy'}
        files = sorted([f.name for f in s1t1_path.iterdir() if f.suffix.lower() in valid_exts])
        
        if limit:
            files = files[:int(limit)]
            print(f"Limit set to {limit}. Processing {len(files)} images.")
        else:
            print(f"Found {len(files)} images.")

        for filename in tqdm(files, desc="Inferring"):
            p_t1 = s1t1_path / filename
            p_t2 = s1t2_path / filename
            
            if not p_t2.exists():
                print(f"Skipping {filename}: Companion file not found in {s1t2_path}")
                continue
                
            # Load
            try:
                t_s1t1 = load_image2tensor(p_t1, transform).to(device)
                t_s1t2 = load_image2tensor(p_t2, transform).to(device)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

            # Infer
            with torch.no_grad():
                sr_up, sr_down = model(t_s1t1, t_s1t2)
                sr_fusion = 0.5 * sr_up[-1] + 0.5 * sr_down[-1]
            
            # Post-process
            sr_fusion = sr_fusion.squeeze(0).squeeze(0) # (H, W)
            sr_fusion = (sr_fusion * cfg_norm['std'] + cfg_norm['mean']).clamp(0, 1)
            output_image = (sr_fusion.cpu() * 255).clamp(0, 255).byte().numpy()
            
            # Save
            save_file = Path(save_path) / filename
            if save_file.suffix == '.npy':
                save_file = save_file.with_suffix('.png')
            Image.fromarray(output_image).save(save_file)
            
        print(f"Inference completed. Results saved to {save_path}")

    # Single Image Inference
    elif s1t1_path.is_file() and s1t2_path.is_file():
        print(f"{30*'-'} Single Image Inference Mode {30*'-'}")
        
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
        
        if save_path:
            save_file = Path(save_path)
            # If save_path is a directory, append filename
            if save_file.suffix == '': 
                os.makedirs(save_file, exist_ok=True)
                save_file = save_file / s1t1_path.name
            else:
                os.makedirs(save_file.parent, exist_ok=True)
            
            if save_file.suffix == '.npy':
                save_file = save_file.with_suffix('.png')
                
            output_image.save(save_file)
            print(f"Saved result to {save_file}")
        else:
            output_image.show()
    
    else:
        print("Error: Invalid input paths. Both must be either directories or files.")
        sys.exit(1)




    