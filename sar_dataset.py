from torch.utils.data import Dataset
import PIL
from torchvision.transforms import ToTensor, Compose, RandomAffine, Normalize, ToPILImage
from PIL import Image
import os

class SarDataset(Dataset):
    def __init__(self, root: str, train:bool = True, transform= None):
        self.transform = transform
        root = os.path.join(root, 'train' if train else 'test')
        self.lr_time_1 = []
        self.lr_time_2 = []
        self.hr = []

        for image_name in os.listdir(root): # Train folder or test folder
            if 'time_1' not in image_name and 'time_2' not in image_name: # is HR
                image_path = os.path.join(root, image_name)
                self.hr.append(image_path)
                # Append time_1, time_2 corresponding with hr
                self.lr_time_1.append(os.path.join(root, image_name.split('.')[0]+'_time_1.png'))
                self.lr_time_2.append(os.path.join(root, image_name.split('.')[0]+'_time_2.png'))
    
    def __len__(self):
        return len(self.hr)
    
    def __getitem__(self, idx):
        hr = Image.open(self.hr[idx]).convert('L')
        path = self.hr[idx]
        lr_time_1 = Image.open(self.lr_time_1[idx]).convert('L')
        lr_time_2 = Image.open(self.lr_time_2[idx]).convert('L')

        if self.transform:
            lr_time_1 = self.transform(lr_time_1)
            lr_time_2 = self.transform(lr_time_2)
            hr= self.transform(hr)

        return path, (lr_time_1, lr_time_2), hr
        

if __name__ == '__main__':
    transform  = Compose(
        transforms=[
            ToTensor(),
            Normalize(mean=[0.5],
                       std=[0.5])
        ]
    )

    dataset = SarDataset(root=r'/mnt/data1tb/vinh/ISSM-SAR/dataset', train=True, transform=transform)
    print(len(dataset))
    path_origin, (lr_1, lr_2), hr = dataset[0]
    print(lr_1.shape, hr.shape)
    # convert tensors back to PIL images (undo Normalize) and show
    to_pil = ToPILImage()
    def unnormalize(t):
        return (t * 0.5) + 0.5

    print(path_origin)
    pil_lr_1 = to_pil(unnormalize(lr_1))
    pil_lr_2 = to_pil(unnormalize(lr_2))
    pil_hr = to_pil(unnormalize(hr))

    pil_lr_1.show()
    pil_lr_2.show()
    pil_hr.show()


