import torch
from src import ISSM_SAR, SarDataset, lratio_loss, l1_loss, psnr_torch, ssim_torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
# from torchmetrics.image import PeakSignalNoiseRatio as PSNR
# from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import yaml
import argparse

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ =='__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='Train ISSM-SAR Model')
    parser.add_argument('--config_path', type=str, default= '/mnt/data1tb/vinh/ISSM-SAR/config/base_config.yaml', help='Path to the YAML config file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)
    data_cfg = config['data']
    train_cfg = config['train']
    model_cfg = config['model']

    # Check cuda available and use
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU {torch.cuda.get_device_name("cuda")}')
    else:
        device = torch.device('cpu')
        print('No GPU, using CPU instead')


    # Transform
    transform = Compose(transforms = [
        ToTensor(),
        Normalize(mean = [0.5],
                std=[0.5])
    ])

    # Init writer log
    log_dir = f"runs/exp_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir)

    # Train, test set
    train_set= SarDataset(root = data_cfg['root'], train=True, transform=transform)
    test_set= SarDataset(root = data_cfg['root'], train=False, transform=transform)

    # Train, test loader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=data_cfg['train_batch_size'],
        shuffle=True,
        num_workers=data_cfg['num_workers'],
        drop_last=True
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=data_cfg['test_batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        drop_last=True
    )
    num_iter_train = len(train_loader)
    num_iter_test = len(test_loader)

    # Configure model
    model = ISSM_SAR(model_cfg).to(device)
    optimizer = Adam(model.parameters(), lr=train_cfg['lr'])

    for epoch in range(train_cfg['epochs']):
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
            tqdm_train_loader.set_description(f'Epoch [{epoch+1}/{train_cfg["epochs"]}], Iter [{iter+1}/{num_iter_train}], Loss: {loss.item():.5f}')
        
        print(f"Epoch [{epoch+1}/{train_cfg['epochs']}], Loss_avg: {(loss_show/num_iter_train):.5f}") # Print average loss of epoch
        
        # Valid
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        psnr_all = 0
        ssim_all = 0
        with torch.no_grad():
            for iter, ((lr1, lr2), hr) in enumerate(tqdm_test_loader):
                lr1, lr2, hr = lr1.to(device), lr2.to(device), hr.to(device)
                sr_up, sr_down = model(lr1, lr2)
                sr_fusion = 0.5*sr_up[-1] + 0.5*sr_down[-1] # sr = 0.5*sr_up + 0.5*sr_down
                # Normalize from [-1, 1] -> [0, 1]
                hr = ((hr*0.5)+0.5).clamp(0,1)
                sr_fusion = ((sr_fusion*0.5)+0.5).clamp(0,1)

                # Calculate metrics
                psnr_iter= psnr_torch(sr_fusion, hr).item()
                ssim_iter= ssim_torch(sr_fusion, hr).item()
                psnr_all += psnr_iter
                ssim_all += ssim_iter
            print(f'Epoch [{epoch+1}/{train_cfg["epochs"]}]\nPSNR: {psnr_all/num_iter_test:.3f} dB\n SSIM: {ssim_all/num_iter_test:.3f}')
        
        # logging
        writer.add_scalar(tag='Loss/train', scalar_value=loss_show/num_iter_train, global_step=epoch)
        writer.add_scalar(tag='Metrics/PSNR', scalar_value=psnr_all/num_iter_test, global_step=epoch)
        writer.add_scalar(tag='Metrics/SSIM', scalar_value=ssim_all/num_iter_test, global_step=epoch)
    
    writer.close() # Close







        

        
