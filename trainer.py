import os
import torch
from datetime import datetime
from tqdm import tqdm
from src import l1_loss, lratio_loss, psnr_torch, ssim_torch

class Trainer():
    def __init__(self, model, optimizer, train_loader, valid_loader, device, config, writer, run_name, resume_path = None):
        # Model
        self.model = model
        self.optimizer = optimizer
        self.device = device

        # Loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_iter_train = len(self.train_loader)
        self.num_iter_valid = len(self.valid_loader)

        # config
        self.config = config
        self.train_cfg = self.config['train']
        self.model_cfg = self.config['model']

        # log, checkpoint
        self.writer = writer
        self.resume_path =resume_path
        self.run_name = run_name
        self.checkpoint_dir = os.path.join('checkpoints', self.run_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.start_epoch = 1
        self.best_psnr = 0.0
        self.total_epochs = self.train_cfg['epochs']
        self.end_epoch = self.start_epoch + self.total_epochs

        if self.resume_path:
            self._load_checkpoint(resume_path)


    def _load_checkpoint(self, resume_path):
        try:
            # Load checkpoint
            checkpoint = torch.load(resume_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.end_epoch = self.start_epoch + self.total_epochs
            self.best_psnr = checkpoint['best_psnr']
        
        except Exception as e:
            print(f'Error loading checkpoint {e}. Double check resume path')
            exit()
    
    def _save_checkpoint(self, epoch:int, is_best: bool):
        checkpoint_data = {
            'epoch':epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        # Save last checkpoint
        last_save_path = os.path.join(self.checkpoint_dir, 'last.pth')
        torch.save(checkpoint_data, last_save_path)

        # Save best checkpoint
        if is_best:
            best_save_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint_data, best_save_path)
            print(f"Epoch {epoch}: New best model saved to {best_save_path}")
    

    def _train_epoch(self, epoch: int):
        """
        Process an training epoch
        """
        self.model.train()
        loss_show = 0.0
        tqdm_train_loader = tqdm(self.train_loader, desc=f'Epoch [{epoch}/{self.end_epoch-1}] Train')

        for iter, ((lr1, lr2), hr) in enumerate(tqdm_train_loader):
            lr1, lr2 = lr1.to(self.device), lr2.to(self.device)
            hr = hr.to(self.device)
            # forward pass
            sr_up, sr_down = self.model(lr1, lr2)
            
            # compute loss
            l1_total = (lratio_loss(sr_up[-1], hr) + lratio_loss(sr_down[-1], hr)) + (l1_loss(sr_up[-1], hr) + l1_loss(sr_down[-1], hr))
            l2_total = 0
            for idx in range(self.model_cfg['num_ifs']):
                l2_total += (lratio_loss(sr_up[idx], hr) + lratio_loss(sr_down[idx], hr)) + (l1_loss(sr_up[idx], hr) + l1_loss(sr_down[idx], hr))
            loss = l1_total + l2_total
            loss_show += loss.item()
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Update tqdm
            tqdm_train_loader.set_postfix(Loss=f'{loss.item():.5f}')
        
        avg_loss = loss_show/self.num_iter_train
        print(f"Epoch [{epoch}/{self.end_epoch-1}], Average Train Loss: {avg_loss:.5f}")
        return avg_loss


    def _validate_epoch(self, epoch: int):
        """
        Process an validation epoch
        """
        self.model.eval()
        psnr_sum = 0
        ssim_sum = 0
        tqdm_valid_loader = tqdm(self.valid_loader, desc=f'Epoch [{epoch}/{self.end_epoch-1}] Valid')
        with torch.no_grad():
            for iter, ((lr1, lr2), hr) in enumerate(tqdm_valid_loader):
                lr1, lr2 = lr1.to(self.device), lr2.to(self.device)
                hr = hr.to(self.device)
                # forward pass
                sr_up, sr_down = self.model(lr1, lr2)
                sr_fusion = 0.5 * sr_up[-1] + 0.5 * sr_down[-1]
                
                # Normalize from [-1, 1] -> [0, 1]
                hr = (hr * 0.5 + 0.5).clamp(0, 1)
                sr_fusion = (sr_fusion * 0.5 + 0.5).clamp(0, 1)

                # calculate metrics
                psnr_batch = psnr_torch(sr_fusion, hr).item()
                ssim_batch = ssim_torch(sr_fusion, hr).item()
                psnr_sum += psnr_batch
                ssim_sum += ssim_batch

            avg_psnr = psnr_sum/self.num_iter_valid
            avg_ssim = ssim_sum/self.num_iter_valid
            print(f'Epoch [{epoch}/{self.end_epoch-1}] Valid\nPSNR: {avg_psnr:.3f} dB\nSSIM: {avg_ssim:.3f}')
        
        
        return avg_psnr, avg_ssim
    

    def run(self):
        if not self.resume_path:
            print(f"""--------------------
                \nStarting new run: {self.run_name}
                """)
        else:
            print(f"""------------------
                  \nResuming run '{self.run_name}' from epoch {self.start_epoch}. Best PSNR: {self.best_psnr:.3f}
                """)
        
        for epoch in range(self.start_epoch, self.end_epoch):
            torch.cuda.empty_cache()
            
            # training
            avg_loss = self._train_epoch(epoch)
            
            # validate
            avg_psnr, avg_ssim = self._validate_epoch(epoch)
            
            # logging
            self.writer.add_scalar(tag='Loss/Train', scalar_value=avg_loss, global_step=epoch)
            self.writer.add_scalar(tag='Metrics/PSNR', scalar_value=avg_psnr, global_step=epoch)
            self.writer.add_scalar(tag='Metrics/SSIM', scalar_value=avg_ssim, global_step=epoch)

            # Save checkpoint
            is_best = avg_psnr > self.best_psnr
            if is_best:
                self.best_psnr = avg_psnr
            self._save_checkpoint(epoch, is_best)
        
        self.writer.close()
        print("Training finished.")
            

            



        

