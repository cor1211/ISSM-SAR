import torch
from model import ISSM_SAR
from loss import lratio_loss, l1_loss
from sar_dataset import SarDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
from torch.optim import Adam
from metrics import psnr_torch, ssim_torch
# from torchmetrics.image import PeakSignalNoiseRatio as PSNR
# from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


if __name__ =='__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU {torch.cuda.get_device_name("cuda")}')
    else:
        print('No GPU, using CPU instead')

    # Transform
    transform = Compose(transforms = [
        ToTensor(),
        Normalize(mean = [0.5],
                std=[0.5])
    ])

    # Train, test set
    train_set= SarDataset(root = r'/mnt/data1tb/vinh/ISSM-SAR/dataset', train=True, transform=transform)
    test_set= SarDataset(root = r'/mnt/data1tb/vinh/ISSM-SAR/dataset', train=False, transform=transform)

    # Train, test loader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        drop_last=True
    )
    num_iter_train = len(train_loader)
    num_iter_test = len(test_loader)

    # Configure model
    model = ISSM_SAR(in_channel=1, out_channel=1, num_ifs=3).to(device)
    optimizer = Adam(model.parameters(), lr=1e-6)

    # Metrics validation
    # psnr_metric = PSNR(data_range=1.0).to(device)
    # ssim_metric = SSIM(data_range=1.0).to(device)

    for epoch in range(5):
        torch.cuda.empty_cache()
        model.train()
        loss_show = 0.0
        tqdm_train_loader = tqdm(train_loader)

        for iter, ((lr1, lr2), hr) in enumerate(tqdm_train_loader):
            lr1, lr2 = lr1.to(device), lr2.to(device)
            hr = hr.to(device)
            sr_up, sr_down = model(lr1, lr2)

            # Compute loss
            l1_total = (lratio_loss(sr_up[-1], hr) + lratio_loss(sr_down[-1], hr)) + (l1_loss(sr_up[-1],hr) + l1_loss(sr_down[-1], hr))
            l2_total = 0
            for  idx in range(3):
                l2_total+= (lratio_loss(sr_up[idx], hr) + lratio_loss(sr_down[idx], hr)) + (l1_loss(sr_up[idx], hr) + l1_loss(sr_down[idx], hr))
            loss = l1_total + l2_total
            loss_show+=loss.item()
            optimizer.zero_grad() # Zero gradients
            loss.backward() # Compute gradients
            optimizer.step() # Update weights
            tqdm_train_loader.set_description(f'Epoch [{epoch+1}/{5}], Iter [{iter+1}/{num_iter_train}], Loss: {loss.item():.5f}')
        
        print(f"Epoch [{epoch+1}/{5}], Loss_avg: {(loss_show/num_iter_train):.5f}") # Print average loss of epoch
        
        # Valid
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        psnr_value = 0
        ssim_value = 0
        with torch.no_grad():
            for iter, ((lr1, lr2), hr) in enumerate(tqdm_test_loader):
                lr1, lr2, hr = lr1.to(device), lr2.to(device), hr.to(device)
                sr_up, sr_down = model(lr1, lr2)
                sr_fusion = 0.5*sr_up[-1] + 0.5*sr_down[-1]
                # Normalize from [-1, 1] -> [0, 1]
                hr = ((hr*0.5)+0.5).clamp(0,1)
                sr_fusion = ((sr_fusion*0.5)+0.5).clamp(0,1)

                # Calculate metrics
                psnr_value+= psnr_torch(sr_fusion, hr).item()
                ssim_value+= ssim_torch(sr_fusion, hr).item()

            print(f'Epoch [{epoch+1}/{5}]\nPSNR: {psnr_value/num_iter_test:.3f} dB\n SSIM: {ssim_value/num_iter_test:.3f}')







        

        
