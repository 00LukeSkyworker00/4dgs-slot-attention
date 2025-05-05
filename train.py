import os
import shutil
import glob
import argparse

from dataset import *
from model import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch
import torchvision
from omegaconf import OmegaConf

import torch.multiprocessing as mp
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, test_set, train_len, val_len, cfg, device):
        self.writer = SummaryWriter(os.path.join(cfg.output.dir, 'logs'))

        self.img = test_set['gt_imgs'].permute(2,0,1)
        self.gs = torch.cat(test_set['gs'], dim=-1).to(device).unsqueeze(0)
        self.pe = test_set['gs'][0].to(device).unsqueeze(0)
        self.Ks = test_set['Ks'].unsqueeze(0)
        self.w2cs = test_set['w2cs'].unsqueeze(0)

        self.renderer = Renderer(tuple(cfg.dataset.resolution), requires_grad=False)

        self.total_loss = 0
        self.p_loss = 0
        self.c_loss = 0

        self.ari = 0

        self.best = float('inf')

        self.train_len = train_len
        self.val_len = val_len

        self.save_env(cfg)

    def record_loss(self, total_loss:torch.Tensor, p_loss:torch.Tensor, c_loss:torch.Tensor):
        self.total_loss += total_loss.item()
        self.p_loss += p_loss.item()
        self.c_loss += c_loss.item()

    def record_ari(self, gt:torch.Tensor, pred:torch.Tensor):
        pass

    def plt_loss(self, epoch:int, start_time, mode='Train') -> bool:
        assert mode in ['Train', 'Val']
        if mode == 'Train':
            len = self.train_len
        else:
            len = self.val_len            

        self.total_loss /= len
        self.p_loss /= len
        self.c_loss /= len

        isBest = False
        if mode == 'Val' and self.best > self.total_loss:
            self.best = self.total_loss
            isBest = True

        self.writer.add_scalars(f'{mode} Loss', {
            'total': self.total_loss,
            'position': self.p_loss,
            'color': self.c_loss,
        }, epoch)

        print ("{} Loss: {}, Time: {}".format(mode, self.total_loss,
                    datetime.timedelta(seconds=time.time() - start_time)))
        
        self.total_loss = 0
        self.p_loss = 0
        self.c_loss = 0

        return isBest



    def plt_render(self, model, epoch:int):
        gs_out, gs_slot, gs_mask, _ = model(self.gs, self.pe)
        recon_combined, recon_slots = render_gs(self.renderer, gs_out, gs_slot, gs_mask, self.Ks, self.w2cs)
        
        recon_combined = recon_combined[0].permute(2,0,1)
        recon_slots = recon_slots.permute(0,3,1,2)

        result = torchvision.utils.make_grid(torch.stack([self.img,recon_combined],dim=0))
        self.writer.add_image('result', result, epoch)

        recon_slots = torchvision.utils.make_grid(recon_slots, nrow=5)
        self.writer.add_image('slots_recon', recon_slots, epoch)

        del recon_combined, recon_slots, gs_out, gs_slot, gs_mask

    def save_env(cfg):
        # Copy Python scripts to the output directory
        script_folder = os.path.dirname(os.path.abspath(__file__))
        python_files = glob.glob(os.path.join(script_folder, '*.py'))
        for file in python_files:
            shutil.copy(file, cfg.output.dir)

        # Save the configuration file to the output directory
        with open(os.path.join(cfg.output.dir, 'config.yaml'), 'w') as f:
            OmegaConf.save(cfg, f)


def Trainer(rank, world_size, cfg):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Set device for each process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Initialize process group for DDP
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

    model = SlotAttentionAutoEncoder(cfg.dataset, cfg.cnn, cfg.attention)
    model = model.to(device)

    # Wrap model in DDP
    model = DDP(model, device_ids=[rank])
    # model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Loss function
    mse_loss = nn.MSELoss()

    # Define optimizer
    params = [{'params': model.parameters()}]
    optimizer = optim.Adam(params, lr=cfg.training.lr)   

    # Create DataLoader for train set
    train_list = []    
    for i in range(cfg.dataset.train_idx[0], cfg.dataset.train_idx[1] + 1):
        path = os.path.join(cfg.dataset.dir,f'movi_a_{i:04}_anoMask')
        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            continue
        train_set = ShapeOfMotion(path, cfg.dataset)
        train_list.append(train_set)
    train_set = ConcatDataset(train_list)
    print(f"Number of scene in train set: {len(train_set)}")

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank,
        shuffle=True, seed=cfg.training.seed)
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.training.batch_size, num_workers=0,
        sampler=train_sampler, collate_fn=collate_fn_padd
        )   
    
    # Create DataLoader for val set
    val_list = []    
    for i in range(cfg.dataset.val_idx[0], cfg.dataset.val_idx[1] + 1):
        path = os.path.join(cfg.dataset.dir,f'movi_a_{i:04}_anoMask')
        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            continue
        val_set = ShapeOfMotion(path, cfg.dataset)
        val_list.append(val_set)
    val_set = ConcatDataset(val_list)
    print(f"Number of scene in val set: {len(val_set)}")
    
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank,
        seed=cfg.training.seed)
    val_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.training.batch_size, num_workers=0,
        sampler=val_sampler, collate_fn=collate_fn_padd
        )
    
    del train_list, val_list
    
    # Setup tensorboard and save environment
    if rank == 0:
        logger = Logger(val_set[0], len(train_set), len(val_set), cfg, device)

    # Resume from last checkpoint if exist
    start_epoch = 0
    checkpoint_path = os.path.join(cfg.output.dir,'checkpoints', 'last.ckpt')
    if os.path.exists(checkpoint_path):
        print(f"Rank {rank}: Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resume from {start_epoch} epoch!")
    i = start_epoch * len(train_dataloader)  # Resume step count
    
    start = time.time()

    try:
        for epoch in range(start_epoch, cfg.training.epochs + 1):  # Resume from the saved epoch
            print (f"Start epoch {epoch} of {cfg.training.epochs}")
            """
            Train Loop
            """
            print('Train model...')
            model.train()
            for sample in tqdm(train_dataloader):
                i += 1

                if i < cfg.training.warmup:
                    learning_rate = cfg.training.lr * (i / cfg.training.warmup)
                else:
                    learning_rate = cfg.training.lr

                learning_rate = learning_rate * (cfg.training.decay_rate ** (
                    i / cfg.training.decay_steps))                
                learning_rate *= world_size ** 0.5                
                optimizer.param_groups[0]['lr'] = learning_rate

                # Get data from set
                # imgs = sample['gt_imgs'].to(device)
                gs = sample['gs'].to(device)
                pad_mask = sample['mask'].to(device)
                pe = sample['pe'].to(device)

                # Forward pass
                gs_recon, _, _, loss = model(gs, pe, pad_mask=pad_mask)

                # Loss calculation
                pos_loss = mse_loss(gs_recon[:,:,:3][pad_mask], gs[:,:,:3][pad_mask])
                color_loss = mse_loss(gs_recon[:,:,11:14][pad_mask], gs[:,:,11:14][pad_mask])
                # rots_loss = mse_loss(gs_recon[:,:,3:7], gs[:,:,3:7])
                # scales_loss = mse_loss(gs_recon[:,:,7:10], gs[:,:,7:10])
                # opacities_loss = mse_loss(gs_recon[:,:,10:11], gs[:,:,10:11])

                # loss += mse_loss(gs_recon, gs)
                loss += pos_loss + color_loss

                logger.record_loss(loss,pos_loss,color_loss)

                # Backward pass and optimizer step
                optimizer.zero_grad()
                loss.backward()                
                optimizer.step()
            
                del gs_recon, loss, pos_loss, color_loss

            if rank == 0:
                logger.plt_loss(epoch,start,mode='Train')

            """
            Val Loop
            """
            if not epoch % cfg.output.save_interval: 
                print('Validate model...')
                with torch.no_grad():
                    model.eval()
                    for sample in tqdm(val_dataloader):
                        # Get data from set
                        # imgs = sample['gt_imgs'].to(device)
                        gs = sample['gs'].to(device)
                        pad_mask = sample['mask'].to(device)
                        pe = sample['pe'].to(device)

                        # Forward pass
                        gs_recon, _, _, loss = model(gs, pe, pad_mask=pad_mask)

                        # Loss calculation
                        pos_loss = mse_loss(gs_recon[:,:,:3][pad_mask], gs[:,:,:3][pad_mask])
                        color_loss = mse_loss(gs_recon[:,:,11:14][pad_mask], gs[:,:,11:14][pad_mask])

                        # rots_loss = mse_loss(gs_recon[:,:,3:7], gs[:,:,3:7])
                        # scales_loss = mse_loss(gs_recon[:,:,7:10], gs[:,:,7:10])
                        # opacities_loss = mse_loss(gs_recon[:,:,10:11], gs[:,:,10:11])

                        # loss += mse_loss(gs_recon, gs)
                        loss += pos_loss + color_loss

                        logger.record_loss(loss,pos_loss,color_loss)

                        del gs_recon, loss, pos_loss, color_loss

                if rank == 0:
                    isBest = logger.plt_loss(epoch,start,mode='Val')

                    logger.plt_render(model, epoch)

                    torch.save({
                        'epoch': epoch,  # Save the current epoch
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(cfg.output.dir,'checkpoints', f'{epoch}.ckpt'))

                    torch.save({
                        'epoch': epoch,  # Save the last epoch for resuming
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
                    
                    if isBest:
                        torch.save({
                            'epoch': epoch,  # Save the Best val model
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, os.path.join(cfg.output.dir,'checkpoints', 'best.ckpt'))

                    print (f"Save checkpoint at epoch {epoch}.")

    except KeyboardInterrupt:
        print(f"Process {rank} interrupted.")
    finally:
        destroy_process_group()
        print(f"Process {rank} cleaned up.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument
    parser.add_argument('--data_dir', default='./data', type=str, help='Where to find the dataset.')
    parser.add_argument('--output_dir', default='./tmp', type=str, help='Where to save model.')
    parser.add_argument('--cfg', default='./configs/default.yaml', type=str, help='Where to load cfg.')
    parser.add_argument('--seed', default=0, type=int, help='Set random seed for reproducibility.')
    args = parser.parse_args()

    # Load configuration file
    cfg = OmegaConf.load(args.cfg)
    cfg.dataset.dir = args.data_dir
    cfg.output.dir = args.output_dir
    cfg.training.seed = args.seed

    # Create output directory
    os.makedirs(cfg.output.dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output.dir,'checkpoints'), exist_ok=True)
    print(f"Output Directory: {cfg.output.dir}")

    # Set random seed for reproducibility
    torch.manual_seed(cfg.training.seed)

    world_size = torch.cuda.device_count()
    mp.spawn(Trainer, args=(world_size, cfg), nprocs=world_size, join=True)

        
if __name__ == '__main__':
    main()