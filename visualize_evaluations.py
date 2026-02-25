import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def find_dir(base, experiment, pattern):
    dirs = list((base / experiment).glob(pattern))
    if not dirs:
        raise ValueError(f"Directory not found matching pattern: {experiment} / {pattern}")
    return dirs[0]

def main():
    input_base = Path("/mnt/data1tb/vinh/ISSM-SAR/dataset/fine-tune_splited/val")
    eval_base = Path("/mnt/data1tb/vinh/ISSM-SAR/evaluation_outputs")
    
    s1t1_dir = input_base / "S1T1"
    s1t2_dir = input_base / "S1T2"
    hr_dir = input_base / "S1HR"
    
    # Locate the output folders dynamically
    try:
        v1_ssim_dir = find_dir(eval_base, "NN_Conv Experiment (Ver 1)", "*ssim*/*/*.ckpt")
        v1_lpips_dir = find_dir(eval_base, "NN_Conv Experiment (Ver 1)", "*lpips*/*/*.ckpt")
        v2_ssim_dir = find_dir(eval_base, "Pixel Shuffle Experiment (Ver 2)", "*ssim*/*/*.ckpt")
        v2_lpips_dir = find_dir(eval_base, "Pixel Shuffle Experiment (Ver 2)", "*lpips*/*/*.ckpt")
    except Exception as e:
        print(f"Error locating output directories: {e}")
        return

    # Use Ver_1-SSIM as the source of truth for filenames to avoid processing missing ones
    filenames = [f.name for f in v1_ssim_dir.glob("*.png")]
    
    if not filenames:
        print("No output PNG files found to visualize.")
        return

    out_viz_dir = Path("/mnt/data1tb/vinh/ISSM-SAR/visualizations")
    out_viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(filenames)} images to visualize.")

    for fn in tqdm(filenames, desc="Generating Visualizations"):
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        fig.suptitle(f"Phiên hiệu: {fn}", fontsize=18, fontweight='bold', y=0.98)

        # Helper method for plotting PNGs
        def plot_img(ax, img_path, title):
            if img_path.exists():
                ax.imshow(Image.open(img_path), cmap='gray')
            else:
                ax.text(0.5, 0.5, "Image Not Found", ha='center', va='center', fontsize=12, color='red')
            ax.set_title(title, fontsize=14, pad=10)
            ax.axis('off')

        # Helper method for plotting NPYs
        def plot_npy(ax, dir_path, fn, title):
            npy_fn = Path(fn).with_suffix('.npy').name
            npy_path = dir_path / npy_fn
            if npy_path.exists():
                img = np.load(npy_path)
                # Ensure img is 2D and range is [0, 1] instead of [-1, 1]
                img = np.squeeze(img)
                img = np.clip((img + 1.0) / 2.0, 0.0, 1.0)
                ax.imshow(img, cmap='gray')
            else:
                ax.text(0.5, 0.5, "Image Not Found", ha='center', va='center', fontsize=12, color='red')
            ax.set_title(title, fontsize=14, pad=10)
            ax.axis('off')

        # Row 1: S1T1, S1T2, Ground Truth
        plot_npy(axes[0, 0], s1t1_dir, fn, "S1T1")
        plot_npy(axes[0, 1], s1t2_dir, fn, "S1T2")
        plot_npy(axes[0, 2], hr_dir, fn, "Ground Truth (GT)")
        axes[0, 3].axis('off') # Keep column 4 empty on row 1

        # Row 2: Ver_1-SSIM, Ver_1-LPIPS, Ver_2-SSIM, Ver_2-LPIPS
        plot_img(axes[1, 0], v1_ssim_dir / fn, "Ver_1-SSIM")
        plot_img(axes[1, 1], v1_lpips_dir / fn, "Ver_1-LPIPS")
        plot_img(axes[1, 2], v2_ssim_dir / fn, "Ver_2-SSIM")
        plot_img(axes[1, 3], v2_lpips_dir / fn, "Ver_2-LPIPS")

        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.1, wspace=0.1)
        
        # Save output visual
        plt.savefig(out_viz_dir / fn, bbox_inches='tight', dpi=150)
        plt.close(fig)

    print(f"\n✅ Visualization complete. All grids saved to: {out_viz_dir}")

if __name__ == '__main__':
    main()
