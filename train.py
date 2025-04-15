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
from omegaconf import OmegaConf

import torch.multiprocessing as mp
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Trainer(rank, world_size, cfg):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Set device for each process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Initialize process group for DDP
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

    train_list = []
    for i in range(cfg.dataset.start_idx, cfg.dataset.end_idx+1):
        path = os.path.join(cfg.dataset.dir,f'movi_a_{i:04}_anoMask')
        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            continue
        train_set = ShapeOfMotion(path, cfg.dataset)
        train_list.append(train_set)
        # print(train_set[0]['fg_gs'].shape)
    train_set = ConcatDataset(train_list)
    # train_set = ShapeOfMotion(opt.data_dir)
    print(f"Number of scene in concat dataset: {len(train_set)}")

    # model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iters, train_set[0]['all_gs'].size(-1))
    model = SlotAttentionAutoEncoder(cfg.dataset, cfg.cnn, cfg.attention)
    # model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])
    model = model.to(device)

    # Wrap model in DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Loss function
    criterion = nn.MSELoss()

    # Define optimizer
    params = [{'params': model.parameters()}]
    optimizer = optim.Adam(params, lr=cfg.training.lr)

    # Check for existing checkpoint
    start_epoch = 0
    checkpoint_path = os.path.join(cfg.output.dir,'checkpoints', 'last.ckpt')

    if os.path.exists(checkpoint_path):
        print(f"Rank {rank}: Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resume from {start_epoch} epoch!")

    # Create DataLoader with DistributedSampler
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank,
        shuffle=True, seed=cfg.training.seed)
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers,
        sampler=train_sampler, collate_fn=collate_fn_padd
        )
    
    # Setup tensorboard and save environment
    if rank == 0:
        writer = SummaryWriter(os.path.join(cfg.output.dir, 'logs'))
        save_env(cfg)

    i = start_epoch * len(train_dataloader)  # Resume step count

    try:
        for epoch in range(start_epoch, cfg.training.epochs + 1):  # Resume from the saved epoch
            start = time.time()            
            model.train()
            total_loss = 0

            for sample in tqdm(train_dataloader):
                i += 1

                if i < cfg.training.warmup:
                    learning_rate = cfg.training.lr * (i / cfg.training.warmup)
                else:
                    learning_rate = cfg.training.lr

                learning_rate = learning_rate * (cfg.training.decay_rate ** (
                    i / cfg.training.decay_steps))
                
                learning_rate *= world_size ** 0.5  # Scale by number of GPUs

                optimizer.param_groups[0]['lr'] = learning_rate
                
                # print(sample['gt_imgs'].shape)

                # Get inputs and lengths
                gt_imgs = sample['gt_imgs'].to(device)

                # if cfg.attention.use_all_gs:   
                gs = sample['gs']
                mask = sample['mask']
                pos_embed = sample['gs_pos']
                
                Ks = sample['Ks']
                w2cs = sample['w2cs']

                gs = gs.to(device)
                mask = mask.to(device)
                pos_embed = pos_embed.to(device)

                # Forward pass through model
                recon_combined = model(gs, pos_embed, Ks=Ks, w2cs= w2cs, mask=mask)
                
                # Loss calculation
                loss = criterion(recon_combined, gt_imgs)
                # print(loss.item())
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss /= len(train_dataloader)

            if rank == 0:   # Print and save only from rank 0
                print ("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
                    datetime.timedelta(seconds=time.time() - start)))
                
                writer.add_scalar('Loss/train', total_loss, epoch)

                if not epoch % cfg.output.save_interval:

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

                    print ("Save checkpoint at Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
                    datetime.timedelta(seconds=time.time() - start)))

    except KeyboardInterrupt:
        print(f"Process {rank} interrupted.")
    finally:
        destroy_process_group()
        print(f"Process {rank} cleaned up.")

def save_env(cfg):
    # Copy Python scripts to the output directory
    script_folder = os.path.dirname(os.path.abspath(__file__))
    python_files = glob.glob(os.path.join(script_folder, '*.py'))
    for file in python_files:
        shutil.copy(file, cfg.output.dir)

    # Save the configuration file to the output directory
    with open(os.path.join(cfg.output.dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)
        
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
    
    # # Index for output directory
    # index = 0
    # output_dir = cfg.output.dir + f'__{index:02}'
    # while os.path.exists(output_dir):
    #     index += 1
    #     output_dir = cfg.output.dir + f'__{index:02}'
    # cfg.output.dir = output_dir

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