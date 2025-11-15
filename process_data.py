import PIL
from PIL import Image
import os
import numpy as np


def pil_rgb_to_bt709(img_pil):
    """
    img_pil: ảnh PIL RGB
    return: ảnh PIL grayscale theo chuẩn BT.709
    """
    img = np.array(img_pil).astype("float32")

    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    gray = 0.2126 * R + 0.7152 * G + 0.0722 * B

    # Clip & convert to uint8
    gray = np.clip(gray, 0, 255).astype("uint8")

    return Image.fromarray(gray, mode='L')


def change_to_gray(folder_path, mode = 'L'):
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        # Open image
        img = Image.open(image_path)

        if mode == 'L':
            img_gray = img.convert('L') # convert to gray
        elif mode == 'bt709':
            img_gray = pil_rgb_to_bt709(img)

        img_gray.save(image_path) # Write (ghi de`)
        print(f'Convert {image_path} to Gray')
    

def generate_lr(folderpath, shape: int = 5, scale_factor: int = 2): # Need to have hr_gray first
    for image_name in os.listdir(folderpath):
        if 'time_1' not in image_name and 'time_2' not in image_name: #hr
            image_path = os.path.join(folderpath, image_name)
            hr = (Image.open(image_path))
            H, W = hr.size
            # Generate LR_noise time 1, tim 2
            for idx in range(2):
                hr_noise = np.multiply(hr, np.random.gamma(shape = shape, scale=1/shape, size=hr.size))
                hr_noise = np.clip(hr_noise, 0, 255).astype(np.uint8)
                if scale_factor != 1:
                    lr = Image.fromarray(hr_noise).resize((H//scale_factor, W//scale_factor), resample=Image.Resampling.BILINEAR)
                    lr_path = image_path.split('.')[0]+f'_time_{idx+1}.png'
                    lr.save(lr_path)
                    print(f'Gen LR {lr_path}') 
                else:
                    lr_path = image_path.split('.')[0]+f'_time_{idx+1}.png'
                    lr = Image.fromarray(hr_noise)
                    lr.save(lr_path)
                    print(f'Gen LR {lr_path}') 
                
if __name__ == '__main__':
    # change_to_gray(r'/mnt/data1tb/vinh/ISSM-SAR/CloudRemovalData/CloudRemovalData_Region2/s2 (1)', mode='bt709')
    generate_lr(r'/mnt/data1tb/vinh/ISSM-SAR/CloudRemovalData/CloudRemovalData_Region2/s2 (1)', shape=5, scale_factor=1)