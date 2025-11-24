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


def change_to_gray(folder_path, save_folder, mode = 'L', stop: int = 0):
    count = 0
    os.makedirs(save_folder, exist_ok=True)
    if stop == 0:
        stop = 9999999999999999999999 # Full dataset
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        # Open image
        img = Image.open(image_path)

        if mode == 'L':
            img_gray = img.convert('L') # convert to gray
        elif mode == 'bt709':
            img_gray = pil_rgb_to_bt709(img)

        save_path = os.path.join(save_folder, image_name)
        img_gray.save(save_path) # Write (ghi de`)
        print(f'Convert {image_path} → {save_path} to Gray')
        count+=1
        if count >= stop:
            print(f'\n-------------Early stop with quantity: {count}---------------\n')
            break


def generate_lr(folderpath, save_folder, shape: int = 5, scale_factor: int = 2): # Need to have hr_gray first
    for image_name in os.listdir(folderpath):
        if 'time_1' not in image_name and 'time_2' not in image_name: #hr
            image_path = os.path.join(folderpath, image_name)
            hr = (Image.open(image_path))
            W, H = hr.size
            # Generate LR_noise time 1, tim 2
            for idx in range(2):
                hr_noise = np.multiply(hr, np.random.gamma(shape = shape, scale=1/shape, size=hr.size))
                hr_noise = np.clip(hr_noise, 0, 255).astype(np.uint8)
                if scale_factor != 1:
                    lr = Image.fromarray(hr_noise).resize((H//scale_factor, W//scale_factor), resample=Image.Resampling.BILINEAR)
                else:
                    lr = Image.fromarray(hr_noise)

                lr_name =  image_name.split('.')[0]+f'_time_{idx+1}.png'
                lr_path = os.path.join(save_folder, lr_name)
                lr.save(lr_path)
                print(f'Gen LR {lr_path}')
                  
                
if __name__ == '__main__': # train: 96108
    # change_to_gray(folder_path=r'/mnt/hdd12tb/code/vinhlc/SGDM_x8_340manh_2022_2023/train/hr_256', save_folder=r'/mnt/hdd12tb/code/vinhlc/issm_sar/dataset/train_1_shape_20', mode = 'L', stop = 150000)
    generate_lr(folderpath=r'/mnt/hdd12tb/code/vinhlc/issm_sar/dataset/train_1_shape_20', save_folder=r'/mnt/hdd12tb/code/vinhlc/issm_sar/dataset/train_1_shape_20', shape=20, scale_factor=2)