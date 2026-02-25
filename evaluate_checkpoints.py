import argparse
import yaml
import torch
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image

# Import metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity

# Fallbacks for No-Reference metrics (MUSIQ, CLIP-IQA)
try:
    from pyiqa import create_metric
    PYIQA_AVAILABLE = True
except ImportError:
    PYIQA_AVAILABLE = False
    print("WARNING: 'pyiqa' is not installed. MUSIQ and CLIPIQA will be disabled.")
    print("Install via: pip install pyiqa")

# Project imports
from src import ISSM_SAR

def load_config(config_path: str) -> dict:
    with open(Path(config_path), 'r') as file:
        return yaml.safe_load(file)

def load_image2tensor(image_path: Path, transform: Compose) -> torch.Tensor:
    valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    if image_path.suffix.lower() == '.npy':
        img_np = np.load(image_path).astype(np.float32)
        if img_np.ndim == 2:
            img_np = img_np[np.newaxis, ...]
        elif img_np.ndim == 3 and img_np.shape[2] <= 4:
            img_np = img_np.transpose(2, 0, 1)
        return torch.from_numpy(img_np).unsqueeze(0)
    elif image_path.suffix.lower() in valid_exts:
        image = Image.open(image_path).convert('L')
        transformed_img = transform(image)
        return transformed_img.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported file format: {image_path}")

def denorm_to_01(tensor, mean, std):
    """Denormalize tensor from [-1, 1] to [0, 1] range."""
    return (tensor * std + mean).clamp(0, 1)

def get_checkpoints_in_dir(ckpt_dir):
    """Find all .ckpt files recursively in a directory."""
    checkpoints = []
    path = Path(ckpt_dir)
    if not path.exists():
        return checkpoints
    for ext in ['*.ckpt', '*.pth', '*.pt']:
        checkpoints.extend(list(path.rglob(ext)))
    return sorted(checkpoints)

def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoints for ISSM-SAR (Current Branch)')
    parser.add_argument('--config_path', type=str, default='config/evaluate_config.yaml', help='Path to evaluate config')
    args = parser.parse_args()

    config = load_config(args.config_path)
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset paths
    ds_cfg = config['dataset']
    s1t1_path = Path(ds_cfg['s1t1_path'])
    s1t2_path = Path(ds_cfg['s1t2_path'])
    hr_path = Path(ds_cfg['hr_path'])
    limit = ds_cfg.get('limit')
    
    mean = ds_cfg['normalize']['mean']
    std = ds_cfg['normalize']['std']
    
    transform = Compose([
        ToTensor(),
        Normalize(mean=[mean], std=[std])
    ])

    # Get valid files
    valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.npy'}
    all_files = sorted([f.name for f in s1t1_path.iterdir() if f.suffix.lower() in valid_exts])
    if limit is not None:
        all_files = all_files[:int(limit)]
    print(f"Evaluating on {len(all_files)} images...")

    # Initialize Metrics
    print("Initializing metrics...")
    metrics = {
        'psnr': PeakSignalNoiseRatio(data_range=1.0).to(device),
        'ssim': StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
        'lpips': LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
    }
    
    if PYIQA_AVAILABLE:
        metrics['musiq'] = create_metric('musiq').to(device)
        metrics['clipiqa'] = create_metric('clipiqa').to(device)

    # Load Model Architecture (Once)
    print("\nLoading model architecture...")
    model_cfg = config['model_config']
    model = ISSM_SAR(config=model_cfg).to(device)
    
    # Find Checkpoints
    ckpt_dir = config['checkpoint_dir']
    run_name = config.get('run_name', 'Branch_Unknown')
    checkpoints = get_checkpoints_in_dir(ckpt_dir)
    
    if not checkpoints:
        print(f"Error: No checkpoints found in {ckpt_dir}")
        sys.exit(1)
        
    print(f"\n{'='*50}")
    print(f"Run Name: {run_name}")
    print(f"Found {len(checkpoints)} checkpoints in {ckpt_dir}")
    print(f"{'='*50}\n")

    results = []

    # Evaluate each checkpoint
    for ckpt_path in checkpoints:
        print(f"\nEvaluating: {ckpt_path.name}")
        
        # Load weights
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
            clean_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict, strict=False)
            model.eval()
        except Exception as e:
            print(f"❌ Error loading checkpoint {ckpt_path.name}: {e}")
            continue

        # Reset metrics
        for m in metrics.values():
            if hasattr(m, 'reset'):
                m.reset()
        
        musiq_scores = []
        clipiqa_scores = []

        # Inference loop
        for filename in tqdm(all_files, desc="Inferring"):
            try:
                t_s1t1 = load_image2tensor(s1t1_path / filename, transform).to(device)
                t_s1t2 = load_image2tensor(s1t2_path / filename, transform).to(device)
                t_hr = load_image2tensor(hr_path / filename, transform).to(device)
                
                with torch.no_grad():
                    sr_up, sr_down = model(t_s1t1, t_s1t2)
                    sr = 0.5 * sr_up[-1] + 0.5 * sr_down[-1]
                
                sr_01 = denorm_to_01(sr, mean, std)
                hr_01 = denorm_to_01(t_hr, mean, std)

                metrics['psnr'].update(sr_01, hr_01)
                metrics['ssim'].update(sr_01, hr_01)
                
                sr_rgb = sr_01.repeat(1, 3, 1, 1)
                hr_rgb = hr_01.repeat(1, 3, 1, 1)
                metrics['lpips'].update(sr_rgb, hr_rgb)
                
                if PYIQA_AVAILABLE:
                    with torch.no_grad():
                        musiq_scores.append(metrics['musiq'](sr_rgb).item())
                        clipiqa_scores.append(metrics['clipiqa'](sr_rgb).item())

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        # Aggregate scores
        ckpt_results = {
            'Run_Name': run_name,
            'Checkpoint': ckpt_path.name,
            'PSNR': metrics['psnr'].compute().item(),
            'SSIM': metrics['ssim'].compute().item(),
            'LPIPS': metrics['lpips'].compute().item()
        }
        if PYIQA_AVAILABLE:
            ckpt_results['MUSIQ'] = np.mean(musiq_scores) if musiq_scores else 0.0
            ckpt_results['CLIP-IQA'] = np.mean(clipiqa_scores) if clipiqa_scores else 0.0
            
        print(" | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in ckpt_results.items()]))
        results.append(ckpt_results)

    # Export
    if results:
        df = pd.DataFrame(results)
        output_csv = config.get('output_csv', 'evaluation_results.csv')
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Evaluation complete. Results exported to: {output_csv}")
    else:
        print("\n⚠️ No results were collected.")

if __name__ == '__main__':
    main()
