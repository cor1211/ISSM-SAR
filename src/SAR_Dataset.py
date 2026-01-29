import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class MultiTempSARDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=True):
        """
        Args:
            root_dir (str): The path contains sub folders (S1T1, S1T2, S1HR).
            phase (str): 'train' or 'val'.
            transform (bool): Apply Augmentation ? (for only train phase).
        """
        self.phase = phase
        self.transform = transform
        
        # Định nghĩa đường dẫn
        self.dir_t1 = os.path.join(root_dir, 'S1T1')
        self.dir_t2 = os.path.join(root_dir, 'S1T2')
        self.dir_hr = os.path.join(root_dir, 'S1HR') # Target
        
        # Lấy danh sách file (Dựa vào folder S1T1 làm chuẩn)
        # Chỉ lấy file .npy
        self.filenames = [x for x in sorted(os.listdir(self.dir_t1)) if x.endswith('.npy')]
        
        # [Sanity Check] Kiểm tra nhanh xem số lượng file có khớp không
        # (Bước này quan trọng để tránh lỗi runtime khi training lâu)
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
        if random.random() < 0.5:
            t1 = np.flip(t1, axis=-1) # Flip chiều W
            t2 = np.flip(t2, axis=-1)
            hr = np.flip(hr, axis=-1)
            
        # Random Vertical Flip
        if random.random() < 0.5:
            t1 = np.flip(t1, axis=-2) # Flip chiều H
            t2 = np.flip(t2, axis=-2)
            hr = np.flip(hr, axis=-2)
            
        # Random Rotate 90 (0, 90, 180, 270)
        k = random.randint(0, 3)
        if k > 0:
            t1 = np.rot90(t1, k, axes=(-2, -1))
            t2 = np.rot90(t2, k, axes=(-2, -1))
            hr = np.rot90(hr, k, axes=(-2, -1))
            
        return t1.copy(), t2.copy(), hr.copy() # .copy() để fix lỗi negative stride của numpy khi convert sang torch

    def __getitem__(self, index):
        filename = self.filenames[index]
        
        # 1. Load Data (Dùng mmap_mode='r' nếu file quá lớn, nhưng patch nhỏ thì load thẳng cho nhanh)
        path_t1 = os.path.join(self.dir_t1, filename)
        path_t2 = os.path.join(self.dir_t2, filename)
        path_hr = os.path.join(self.dir_hr, filename)
        
        # Load raw npy (đã normalize -1, 1 từ khâu prepare data trước đó)
        img_t1 = np.load(path_t1).astype(np.float32)
        img_t2 = np.load(path_t2).astype(np.float32)
        img_hr = np.load(path_hr).astype(np.float32)
        
        # 2. Xử lý Shape: Đảm bảo format là [C, H, W]
        # Nếu save raw là [H, W] thì phải unsqueeze
        if img_t1.ndim == 2:
            img_t1 = img_t1[np.newaxis, ...] # (1, H, W)
        elif img_t1.ndim == 3 and img_t1.shape[2] <= 4: # Nếu đang là [H, W, C]
            img_t1 = img_t1.transpose(2, 0, 1) # -> [C, H, W]
            
        if img_t2.ndim == 2: img_t2 = img_t2[np.newaxis, ...]
        elif img_t2.ndim == 3 and img_t2.shape[2] <= 4: img_t2 = img_t2.transpose(2, 0, 1)
            
        if img_hr.ndim == 2: img_hr = img_hr[np.newaxis, ...]
        elif img_hr.ndim == 3 and img_hr.shape[2] <= 4: img_hr = img_hr.transpose(2, 0, 1)

        # 3. Augmentation (Chỉ dùng cho training)
        if self.phase == 'train' and self.transform:
            img_t1, img_t2, img_hr = self._sync_transform(img_t1, img_t2, img_hr)

        # 4. To Tensor
        # PyTorch nhận float32 tensor
        t1_tensor = torch.from_numpy(img_t1)
        t2_tensor = torch.from_numpy(img_t2)
        hr_tensor = torch.from_numpy(img_hr)

        return {'T1': t1_tensor, 'T2': t2_tensor, 'HR': hr_tensor, 'filename': filename}