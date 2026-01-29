"""
Module for cropping and saving raw SAR data as .npy files.
- Clip values to [-25, 5] dB range
- Normalize to [-1, 1] range
- Save as .npy with filename format: row_col_originalfilename.npy
"""

import os
import numpy as np
import rasterio
from tqdm import tqdm
import yaml
from collections import OrderedDict


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(f):
    """Load yaml file or string."""
    if os.path.isfile(f):
        with open(f, 'r') as file:
            return yaml.load(file, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])


def load_black_list(black_list_path) -> list:
    """Load blacklist of file names from a txt file."""
    black_list = []
    try:
        with open(black_list_path, 'r') as file:
            for line in file.readlines():
                black_list.append(os.path.basename(line.strip()).split('.')[0])
        return black_list
    except FileNotFoundError as e:
        print(f'{e}: Black list path isnt valid')
        return []


def load_whitelist(whitelist_path) -> list:
    """Load whitelist of file names from a txt file."""
    whitelist = []
    try:
        with open(whitelist_path, 'r') as file:
            for line in file.readlines():
                line = line.strip()
                if line:
                    whitelist.append(os.path.basename(line).split('.')[0])
        return whitelist
    except FileNotFoundError as e:
        print(f'{e}: Whitelist path isnt valid')
        return []


def in_black_list(filename, black_list):
    return filename in black_list


def in_whitelist(filename, whitelist):
    return filename in whitelist


def clip_and_normalize_to_minus1_1(data, v_min=-25, v_max=5):
    """
    Clip data to [v_min, v_max] then normalize to [-1, 1] range.
    
    Args:
        data: Input numpy array (SAR data in dB)
        v_min: Minimum clip value (default: -25 dB)
        v_max: Maximum clip value (default: 5 dB)
    
    Returns:
        Normalized numpy array in range [-1, 1] as float32
    """
    data_clipped = np.clip(data, v_min, v_max)
    # Normalize to [0, 1] first
    data_norm = (data_clipped - v_min) / (v_max - v_min)
    # Convert to [-1, 1]
    data_normalized = data_norm * 2 - 1
    return data_normalized.astype(np.float32)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def split_and_save_raw_npy(image_path, output_dir, 
                           tile_size=128,
                           split_factor=4,
                           remove_A=False,
                           sar_vv=True,
                           sar_vh=False,
                           v_min=-25,
                           v_max=5,
                           save_fmt='npy'):
    """
    Split image into tiles, clip to [v_min, v_max], normalize to [-1, 1], 
    and save as .npy or .npz files.
    
    Args:
        image_path: Path to input TIFF file
        output_dir: Output directory for saved files
        tile_size: Size of each tile (default: 128)
        split_factor: Overlap factor for splitting (default: 4)
        remove_A: Whether to remove alpha channel (default: False)
        sar_vv: Use only VV band (default: True)
        sar_vh: Use only VH band (default: False)
        v_min: Minimum clip value (default: -25 dB)
        v_max: Maximum clip value (default: 5 dB)
        save_fmt: Save format - 'npy' or 'npz' (default: 'npy')
    """
    
    def process_and_save_raw(tile, row, col):
        # Clip and normalize to [-1, 1]
        tile_normalized = clip_and_normalize_to_minus1_1(tile, v_min, v_max)
        
        # Generate filename: row_col_originalfilename.npy
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ext = '.npy' if save_fmt.lower() == 'npy' else '.npz'
        tile_path = os.path.join(output_dir, f"{row}_{col}_{base_name}{ext}")
        
        if save_fmt.lower() == 'npy':
            np.save(tile_path, tile_normalized)
        else:
            np.savez_compressed(tile_path, data=tile_normalized)

    with rasterio.open(image_path) as src:
        image = src.read()
        width, height = src.width, src.height
        array = np.array(image)

        if array.shape[1] < tile_size or array.shape[2] < tile_size:
            print(f'Shape of image must not smaller than {tile_size}. NEXT!')
            return 0
    
    image = np.transpose(image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    
    if remove_A and image.shape[2] > 3:
        image = image[:, :, 0:3]

    # Select channel based on sar_vv or sar_vh
    if sar_vv:
        image = image[:, :, 0:1]  # VV only, keep dims
    elif sar_vh:
        image = image[:, :, 1:2]  # VH only, keep dims

    num_rows = height // int(tile_size / split_factor)
    num_cols = width // int(tile_size / split_factor)
    
    tile_count = 0
    
    # Process main grid
    for row in range(num_rows):
        for col in range(num_cols):
            start_row = row * (tile_size // split_factor)
            end_row = row * (tile_size // split_factor) + tile_size
            start_col = col * (tile_size // split_factor)
            end_col = col * (tile_size // split_factor) + tile_size

            if end_row <= height and end_col <= width:
                tile = image[start_row:end_row, start_col:end_col, :]
                process_and_save_raw(tile, row, col)
                tile_count += 1

    # Handle remaining rows (bottom edge)
    if height % tile_size != 0:
        for col in range(num_cols):
            start_row = height - tile_size
            start_col = col * (tile_size // split_factor)
            end_col = col * (tile_size // split_factor) + tile_size

            if end_col <= width:
                tile = image[start_row:, start_col:end_col, :]
                process_and_save_raw(tile, row, col)
                tile_count += 1

    # Handle remaining columns (right edge)
    if width % tile_size != 0:
        for row in range(num_rows):
            start_col = width - tile_size
            start_row = row * (tile_size // split_factor)
            end_row = row * (tile_size // split_factor) + tile_size

            if end_row <= height:
                tile = image[start_row:end_row, start_col:, :]
                process_and_save_raw(tile, row, col)
                tile_count += 1

    # Handle corner (bottom-right)
    if height % tile_size != 0 and width % tile_size != 0:
        start_row = height - tile_size
        start_col = width - tile_size
        tile = image[start_row:, start_col:, :]
        process_and_save_raw(tile, row, col)
        tile_count += 1
    
    return tile_count


def main(config_path):
    """
    Main function to process all TIFF files based on config.
    
    Args:
        config_path: Path to YAML config file
    """
    opt = yaml_load(config_path)
    
    chip_size = opt.get('chip_size', 128)
    data_dir = opt['data_dir']
    split_dir = opt.get('split_dir_raw', opt['split_dir'] + '_raw')  # Default: add _raw suffix
    split_factor = opt.get('split_factor', 4)
    remove_A = opt.get('remove_A', False)
    sar_vv = opt.get('sar_vv', True)
    sar_vh = opt.get('sar_vh', False)
    v_min = opt.get('v_min', -25)
    v_max = opt.get('v_max', 5)
    save_fmt = opt.get('raw_save_fmt', 'npy')
    black_list_path = opt.get('black_list_path', '')
    whitelist_path = opt.get('white_list_path', '')

    # Load blacklist and whitelist
    black_list = load_black_list(black_list_path) if black_list_path else []
    if black_list:
        print(f'Loaded {len(black_list)} items in blacklist')
    
    whitelist = None
    if whitelist_path and os.path.exists(whitelist_path):
        whitelist = load_whitelist(whitelist_path)
        print(f'Loaded {len(whitelist)} items in whitelist from {whitelist_path}')
    else:
        print('No whitelist provided - will process all files (except blacklisted)')
    
    count_in_black_list = 0
    count_not_in_whitelist = 0
    count_processed = 0
    total_tiles = 0
    
    # List TIFF files
    tiff_files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
    print(f'Found {len(tiff_files)} .tif files in {data_dir}')
    
    ensure_dir(split_dir)
    
    for tiff_file in tqdm(tiff_files, desc=f"Splitting to {chip_size}x{chip_size} raw tiles"):
        filename_without_ext = tiff_file.split('.')[0]
        
        # Check blacklist
        if black_list and in_black_list(filename_without_ext, black_list):
            print(f'SKIP - Blacklisted: {tiff_file}')
            count_in_black_list += 1
            continue
        
        # Check whitelist
        if whitelist is not None:
            if not in_whitelist(filename_without_ext, whitelist):
                print(f'SKIP - Not in whitelist: {tiff_file}')
                count_not_in_whitelist += 1
                continue
        
        tiff_path = os.path.join(data_dir, tiff_file)
        
        print(f'Processing: {tiff_file}')
        tiles = split_and_save_raw_npy(
            tiff_path, split_dir, chip_size, split_factor,
            remove_A, sar_vv, sar_vh, v_min, v_max, save_fmt
        )
        total_tiles += tiles
        count_processed += 1
    
    print('=' * 50)
    print('Processing Summary:')
    print(f'Total files found: {len(tiff_files)}')
    print(f'Files processed: {count_processed}')
    print(f'Files in blacklist (skipped): {count_in_black_list}')
    if whitelist is not None:
        print(f'Files not in whitelist (skipped): {count_not_in_whitelist}')
    print(f'Total output tiles: {total_tiles}')
    print(f'Output directory: {split_dir}')
    print('=' * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Crop and save raw SAR data as .npy files')
    parser.add_argument('-opt', '--config', type=str, 
                        default='/mnt/data1tb/vinh/ISSM-SAR/utils/config_crop_raw.yaml',
                        help='Path to the config file')
    args = parser.parse_args()
    
    main(args.config)
