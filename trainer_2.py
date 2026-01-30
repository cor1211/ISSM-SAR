import os
import torch
import sys
from torch.nn import L1Loss
from torch.cuda.amp import autocast, GradScaler
from ignite.metrics import PSNR, SSIM, Loss
from ignite.engine import Engine
from pathlib import Path
from tqdm import tqdm
from src import l1_loss, lratio_loss, psnr_torch, ssim_torch, gradient_loss
from torchvision.utils import make_grid
from itertools import islice

def denorm(x, mean=0.5, std=0.5):
    return (x * std + mean).clamp(0, 1)


class Trainer():
    def __init__(self, model, optim, train_loader, valid_loader, device, config, writer, run_name, resume_path = None):
        # Config
        self.config = config
        self.cfg_train = self.config['train']
        self.cfg_model = self.config['model']
        self.cfg_data = self.config['data']
        self.val_step = self.cfg_train['val_step']
        self.total_epochs = self.cfg_train['total_epochs']

        # Model
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optim

        # The Coef of losses
        self.component_losses = self.cfg_train['component_losses']
        self.theta_1 = self.cfg_train['theta_1']
        self.theta_2 = self.cfg_train['theta_2']
        self.theta_3 = self.cfg_train['theta_3']

        # Loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_iter_train = len(self.train_loader)
        self.num_iter_valid = len(self.valid_loader)
        # self.mean = self.cfg_data['mean']
        # self.std = self.cfg_data['std']
        
        # Compute the total of steps
        self.total_steps = self.total_epochs * self.num_iter_train

        # Log & Checkpoint
        self.writer = writer
        self.resume_path = resume_path
        self.run_name = run_name
        self.checkpoint_path = os.path.join('checkpoints', self.run_name)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.current_step = 0
        self.best_ssim = 0.0

        # AMP (Automatic Mixed Precision)
        self.use_amp = self.cfg_train.get('use_amp', False)
        self.grad_clip = self.cfg_train.get('grad_clip', 0.0)
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print(f"AMP Training enabled with GradScaler")
        if self.grad_clip > 0:
            print(f"Gradient clipping enabled with max_norm={self.grad_clip}")

        # Losses
        self.criterion_L1 = L1Loss()
        self.loss_l1 = 0.0
        self.loss_ratio = 0.0
        
        # Initialize Valid Engine
        def eval_step(engine, batch):
            self.model.eval()
            with torch.no_grad():
                # Get data from dict
                s1t1 = batch['T1'].to(self.device)
                s1t2 = batch['T2'].to(self.device)
                hr = batch['HR'].to(self.device)
                # Forward
                s1sr_up, s1sr_down = self.model(s1t1, s1t2)
            
            fusion_sr = 0.5 * s1sr_up[-1] + 0.5 * s1sr_down[-1]

            return fusion_sr, hr, s1t1, s1t2
        
        # Init Metrics of valid set
        self.evaluator = Engine(eval_step)
        Loss(self.criterion_L1, output_transform=lambda x: (x[0], x[1])).attach(self.evaluator, 'l1')
        PSNR(data_range=1.0, output_transform=lambda x: (denorm(x[0]), denorm(x[1]))).attach(self.evaluator, 'psnr')
        SSIM(data_range=1.0, output_transform=lambda x: (denorm(x[0]), denorm(x[1]))).attach(self.evaluator, 'ssim')

        # Load checkpoint if resuming
        if self.resume_path:
            self._load_checkpoint(self.resume_path)
    

    def _load_checkpoint(self, resume_path):
        try:
            checkpoint = torch.load(Path(resume_path), map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_step = checkpoint.get('step', 0)
            self.best_ssim = checkpoint.get('best_ssim', 0.0)
            # Load scaler state if available
            if 'scaler_state_dict' in checkpoint and self.use_amp:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"Resumed from step {self.current_step}. Best SSIM so far: {self.best_ssim}")
        except Exception as e:
            print(f'Error loading checkpoint {e}. Double check resume path')
            sys.exit(1)


    def _save_checkpoint(self ,step: int, is_best: bool):
        checkpoint_data = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'best_ssim': self.best_ssim,
            'config': self.config
        }
        # Save last checkpoint
        lastckpt_save_path = os.path.join(self.checkpoint_path, 'last.pth')
        torch.save(checkpoint_data, lastckpt_save_path)
        print(f'\n{20 * "-"}\nStep {step}: New last model saved to {lastckpt_save_path}\n{20 * "-"}')

        # Save best checkpoint
        if is_best:
            bestckpt_save_path = os.path.join(self.checkpoint_path, 'best.pth')
            torch.save(checkpoint_data, bestckpt_save_path)
            print(f'\n{20 * "-"}\nStep {step}: New best model saved to {bestckpt_save_path}\n{20 * "-"}')

    
    def _validate_step(self, current_step):
        """
        Process an validation each val_step
        """
        #------------Compute loss average each val_step (1000 steps/iters)-----------
        loss_l1_avg = self.loss_l1 / self.val_step
        loss_ratio_avg = self.loss_ratio / self.val_step

        print(f"""Step [{current_step}/{self.total_steps}]
{20 * '-'}
Average Train L1 Loss: {loss_l1_avg:.3f}
Average Train Ratio Loss: {loss_ratio_avg:.3f}
{20 * '-'}""")
        print(f'Start Validating...')


        #-------------Compute metrics of valid set-------------
        DEBUG_VALID_LOADER = 10
        valid_loader = (islice(self.valid_loader, DEBUG_VALID_LOADER) 
            if DEBUG_VALID_LOADER else self.valid_loader)
         
        self.evaluator.run(tqdm(valid_loader, desc = "Validating", leave=False))
        l1_avg_valid_set = self.evaluator.state.metrics['l1']
        psnr_avg_valid_set = self.evaluator.state.metrics['psnr']
        ssim_avg_valid_set = self.evaluator.state.metrics['ssim']

        print(f"""{30*'-'}
L1_val: {l1_avg_valid_set:.3f}\nPSNR: {psnr_avg_valid_set:.3f}db\nSSIM: {ssim_avg_valid_set:.3f}
{30*'-'}""")
        
        #------Log at tensorboard------
        self.writer.add_scalar(tag = 'L1 Loss/Train_Step', scalar_value = loss_l1_avg, global_step = current_step)
        self.writer.add_scalar(tag = 'Ratio Loss/Train_Step', scalar_value = loss_ratio_avg, global_step = current_step)
        self.writer.add_scalar(tag='Metrics/L1', scalar_value = l1_avg_valid_set, global_step = current_step)
        self.writer.add_scalar(tag='Metrics/PSNR', scalar_value = psnr_avg_valid_set, global_step = current_step)
        self.writer.add_scalar(tag='Metrics/SSIM', scalar_value = ssim_avg_valid_set, global_step = current_step)

        # Log images
        sr, hr, s1t1, s1t2 = self.evaluator.state.output
        n_imgs = min(8, sr.size(0))
        
        self.writer.add_image('Images/SR', make_grid((denorm(sr))[:n_imgs], nrow=4), current_step)
        self.writer.add_image('Images/HR', make_grid((denorm(hr))[:n_imgs], nrow=4), current_step)
        self.writer.add_image('Images/S1T1', make_grid((denorm(s1t1))[:n_imgs], nrow=4), current_step)
        self.writer.add_image('Images/S1T2', make_grid((denorm(s1t2))[:n_imgs], nrow=4), current_step)

        # Check is best ssim and save checkpoint
        is_best_ssim = False
        if ssim_avg_valid_set > self.best_ssim:
            is_best_ssim = True
            self.best_ssim = ssim_avg_valid_set
            print(f'New best SSIM: {self.best_ssim:.4f}')

        self._save_checkpoint(self.current_step, is_best_ssim)

        # Reset losses show
        self.loss_l1 = 0.0
        self.loss_ratio = 0.0



    def run(self):
        if not self.resume_path:
            print(f'\n{20*"-"}Starting new run: {self.run_name}{20*"-"}')
        else:
            print(f'\n{20*"-"}Resuming run "{self.run_name}" from step: {self.current_step}')
        
        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.total_steps, initial=self.current_step, desc='Training')

        #-------------MAIN TRAINING------------=
        while self.current_step < self.total_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Get data from dict
            s1t1 = batch['T1'].to(self.device)
            s1t2 = batch['T2'].to(self.device)
            hr = batch['HR'].to(self.device)

            #-------------Start Training-------------
            
            # Set train mode: use dropout and batchnorm if has
            self.model.train()
            
            # Forward with AMP autocast
            with autocast(enabled=self.use_amp):
                s1sr_up, s1sr_down = self.model(s1t1, s1t2)
                # sr_fusion = 0.5 * s1sr_up[-1] + 0.5 * s1sr_down[-1]

                #------Compute losses each iter--------
                total_ratio_loss_iter = torch.tensor(0.0, device=self.device)
                total_l1_loss_iter = torch.tensor(0.0, device=self.device)
                # Note: s1sr_up and s1sr_down are lists of SR outputs (after REC)
                # s1sr_up[i] has shape [B, 1, 2H, 2W] - same as hr
                # total_ratio_loss_iter = torch.zeros(1, device=self.device, dtype=torch.float32)
                # total_l1_loss_iter = torch.zeros(1, device=self.device, dtype=torch.float32)

                # Component losses: calculate for intermediate SR outputs (indices 0 to num_ifs-1)
                if self.component_losses:
                    for idx in range(self.cfg_model['num_ifs']):
                        if self.theta_1 != 0:
                            total_ratio_loss_iter = total_ratio_loss_iter + lratio_loss(s1sr_up[idx], hr) + lratio_loss(s1sr_down[idx], hr)
                        if self.theta_2 != 0:
                            total_l1_loss_iter = total_l1_loss_iter + self.criterion_L1(s1sr_up[idx], hr) + self.criterion_L1(s1sr_down[idx], hr)

                # Final output loss (last element, index = num_ifs)
                if self.theta_1 != 0.0:
                    total_ratio_loss_iter = total_ratio_loss_iter + lratio_loss(s1sr_up[-1], hr) + lratio_loss(s1sr_down[-1], hr)
                if self.theta_2 != 0.0:
                    total_l1_loss_iter = total_l1_loss_iter + self.criterion_L1(s1sr_up[-1], hr) + self.criterion_L1(s1sr_down[-1], hr)

                loss_sum_iter = self.theta_1 * total_ratio_loss_iter + self.theta_2 * total_l1_loss_iter
            
            # Check for NaN
            if torch.isnan(loss_sum_iter) or torch.isinf(loss_sum_iter):
                print(f"\n NaN/Inf detected at step {self.current_step}!")
                print(f"   L1 loss: {total_l1_loss_iter.item()}, Ratio loss: {total_ratio_loss_iter.item()}")
                print(f"   SR output range: [{s1sr_up[-1].min().item():.3f}, {s1sr_up[-1].max().item():.3f}]")
                print(f"   HR range: [{hr.min().item():.3f}, {hr.max().item():.3f}]")
                print(f"   Skipping this batch...")
                self.current_step += 1
                pbar.update(1)
                continue
            
            #------------Update weights with AMP-------------
            self.optimizer.zero_grad()
            self.scaler.scale(loss_sum_iter).backward()  # Scale loss for mixed precision and compute gradients
            
            # Gradient clipping (unscale first, then clip)
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            
            self.scaler.step(self.optimizer)  # Update weights
            self.scaler.update()  # Update scaler

            #----------Ended 1 iter train---------

            #---------Log and Update----------
            self.loss_l1 += total_l1_loss_iter.item()
            self.loss_ratio += total_ratio_loss_iter.item()

            pbar.set_postfix({
                'L1_Loss': f'{total_l1_loss_iter.item():.3f}',
                'Rat_Loss': f'{total_ratio_loss_iter.item():.3f}'
            })

            self.current_step += 1
            pbar.update(1)



            #-----------Validate------------
            if self.current_step % self.val_step == 0:
                self._validate_step(self.current_step)
        
        
        self.writer.close()
        print('Training finished')

