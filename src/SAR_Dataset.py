import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from torchvision.transforms import ToPILImage

def denorm(tensor: torch.Tensor)-> torch.Tensor:
    """
    Denorm tensor from [-1, 1] to [0, 1]
    """
    return (tensor * 0.5 + 0.5).clamp(0, 1)


class MultiTempSARDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=True):
        """
        Args:
            root_dir (str): The path contains sub folders (S1T1, S1T2, S1HR).
            phase (str): 'train' or 'val'.
            transform (bool): Apply Augmentation ? (for only train phase).
        """
        self.phase = phase
        root_dir = os.path.join(root_dir, self.phase)
        self.transform = transform
        
        # Init the path to dir
        self.dir_t1 = os.path.join(root_dir, 'S1T1') # Input1
        self.dir_t2 = os.path.join(root_dir, 'S1T2') # Input 2
        self.dir_hr = os.path.join(root_dir, 'S1HR') # Target
        
        # Get the list of file names
        self.filenames = [x for x in sorted(os.listdir(self.dir_t1)) if x.endswith('.npy')]
        
        # Check the number of dataset
        num_t1 = len(self.filenames)
        num_t2 = len(os.listdir(self.dir_t2))
        num_hr = len(os.listdir(self.dir_hr))
        
        assert num_t1 == num_t2 == num_hr, \
            f"Mismatch number of files! T1: {num_t1}, T2: {num_t2}, HR: {num_hr}"
        
        print(f"[{phase.upper()}] Dataset initialized with {num_t1} samples.")

    def __len__(self):
        return len(self.filenames)

    def _sync_transform(self, t1, t2, hr):
        """
        Hàm thực hiện Augmentation đồng bộ cho cả 3 ảnh.
        Input là numpy array [C, H, W] hoặc [H, W]
        """
        # Random Horizontal Flip
        if random.random() < 0.5: # Return random floating number between 0 and 1
            t1 = np.flip(t1, axis=-1) # Flip W dimension
            t2 = np.flip(t2, axis=-1)
            hr = np.flip(hr, axis=-1)
            
        # Random Vertical Flip
        if random.random() < 0.5: # Return random floating number between 0 and 1
            t1 = np.flip(t1, axis=-2) # Flip H dimension
            t2 = np.flip(t2, axis=-2)
            hr = np.flip(hr, axis=-2)
            
        # Random Rotate 90 (0, 90, 180, 270)
        k = random.randint(0, 3)
        if k != 0: 
            t1 = np.rot90(t1, k, axes=(-2, -1))
            t2 = np.rot90(t2, k, axes=(-2, -1))
            hr = np.rot90(hr, k, axes=(-2, -1))
      
        return t1.copy(), t2.copy(), hr.copy() # .copy() to fix error negative stride numpy when convert to torch

    def __getitem__(self, index):
        filename = self.filenames[index]
        
        # 1. Load Data (Use mmap_mode='r' if large file)
        path_t1 = os.path.join(self.dir_t1, filename)
        path_t2 = os.path.join(self.dir_t2, filename)
        path_hr = os.path.join(self.dir_hr, filename)
        
        # Load raw npy (by default [-1, 1])
        img_t1 = np.load(path_t1).astype(np.float32)
        img_t2 = np.load(path_t2).astype(np.float32)
        img_hr = np.load(path_hr).astype(np.float32)
        
        # Check for NaN/Inf values and replace with 0
        for name, arr in [('T1', img_t1), ('T2', img_t2), ('HR', img_hr)]:
            if np.isnan(arr).any() or np.isinf(arr).any():
                # print(f" Warning: NaN/Inf found in {name} of {filename}, replacing with 0")
                np.nan_to_num(arr, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 2. Process Shape: For sure format [C, H, W]
        # If save raw is [H, W] -> unsqueeze C
        if img_t1.ndim == 2:
            img_t1 = img_t1[np.newaxis, ...] # (1, H, W)
        elif img_t1.ndim == 3 and img_t1.shape[2] <= 4: # If's [H, W, C]
            img_t1 = img_t1.transpose(2, 0, 1) # -> [C, H, W]
            
        if img_t2.ndim == 2: img_t2 = img_t2[np.newaxis, ...]
        elif img_t2.ndim == 3 and img_t2.shape[2] <= 4: img_t2 = img_t2.transpose(2, 0, 1)
            
        if img_hr.ndim == 2: img_hr = img_hr[np.newaxis, ...]
        elif img_hr.ndim == 3 and img_hr.shape[2] <= 4: img_hr = img_hr.transpose(2, 0, 1)

        # 3. Augmentation (Only for training phase)
        if self.phase == 'train' and self.transform:
            img_t1, img_t2, img_hr = self._sync_transform(img_t1, img_t2, img_hr)

        # 4. To Tensor
        t1_tensor = torch.from_numpy(img_t1)
        t2_tensor = torch.from_numpy(img_t2)
        hr_tensor = torch.from_numpy(img_hr)

        return {'T1': t1_tensor, 'T2': t2_tensor, 'HR': hr_tensor, 'filename': filename}



if __name__ == '__main__':
    idx = 0
    root_dir = "/mnt/data1tb/vinh/ISSM-SAR/dataset/fine-tune_splited"
    toPil = ToPILImage()

    train_set = MultiTempSARDataset(root_dir, phase='train', transform=True)
    result_dict = train_set[idx]
    t1_tensor = result_dict['T1']
    t2_tensor = result_dict['T2']
    hr_tensor = result_dict['HR']

    t1_denormed = denorm(t1_tensor)
    t2_denormed = denorm(t2_tensor)
    hr_denormed = denorm(hr_tensor)

    t1_pil = toPil(t1_denormed)
    t2_pil = toPil(t2_denormed)
    hr_pil = toPil(hr_denormed)

    t1_pil.show()
    t2_pil.show()
    hr_pil.show()
    
