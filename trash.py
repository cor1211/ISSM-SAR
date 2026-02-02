import torch
import torch.nn
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.transforms import ToPILImage
import numpy as np

img = Image.open(r'/mnt/data1tb/vinh/ISSM-SAR/dataset/shape20/0_6_C_48_35_D_d_3_time_1.png').convert('L')
img.show()
img_1 = img.convert('L')
img_1.show()
img = np.asarray(img)
img_1 = np.asarray(img_1)
print(np.all(img == img_1))

