# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: slot4dgs
#     language: python
#     name: python3
# ---

# %%
# # %cd /home/skyworker/result/4DGS_SlotAttention/slot_4dgs/movi_a_test_04_21_gs_color_compete_RawColor
# %cd /home/skyworker/result/4DGS_SlotAttention/slot_4dgs/movi_a_test_05_10_temporal_test_full_pe_600set
# !ls checkpoints/

# %%
import matplotlib.pyplot as plt
from PIL import Image as Image

import sys
sys.path.append("/home/skyworker/workspace/4dgs-slot-attention")
from renderer import Renderer
from utils import Logger

import torch
from omegaconf import OmegaConf
from model import SlotAttentionAutoEncoder
from dataset import ShapeOfMotion, collate_fn_padd
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Get cuda device
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

# Load YAML.
cfg = OmegaConf.load('config.yaml')

# Get CFG
seed = cfg.training.seed
batch_size = cfg.training.batch_size
num_slots = cfg.attention.num_slots
num_iters = cfg.attention.num_iters
resolution = tuple(cfg.dataset.resolution)

# %%
# Set target properties.
test_seq = 'movi_a_0001_anoMask'
cfg.attention.num_slots = 10
target_frame = 10

scene_path = os.path.join(cfg.dataset.dir, test_seq)
dataset = ShapeOfMotion(scene_path, cfg.dataset)
sample = dataset[0]
batch = collate_fn_padd([sample])

logger = Logger(batch,cfg,device)

# %%
from renderer import Renderer

# Load ckpt
# best_tag = ''
# ckpt = torch.load(os.path.join('checkpoints',f'best.ckpt'), map_location=device)
# print(f'Best : ', ckpt['epoch'])

best_tag = 'arifg'
ckpt = torch.load(os.path.join('checkpoints',f'best_{best_tag}.ckpt'), map_location=device)
print(f'Best {best_tag}: ', ckpt['epoch'])

# Load model.
model = SlotAttentionAutoEncoder(cfg.dataset, cfg.cnn, cfg.attention, batch['gs'].shape[-1])
model = model.to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

renderer = Renderer(resolution, 24, requires_grad=False)

image = batch['gt_render'].to(device)
gs = batch['gs'].to(device)
pad_mask = batch['mask'].to(device)
pe = batch['pe'].to(device)
Ks = batch['Ks'].to(device)
w2cs = batch['w2cs'].to(device)
ano = batch['ano']

with torch.no_grad():
    gs_recon, gs_slot, gs_mask, loss = model(gs, pe, pad_mask=pad_mask,isInference=True)


# %%
from matplotlib import animation
from IPython.display import HTML

def plt_vid(vid, arifg, dpi=72, tag='recon'):
    assert tag in ['recon', 'slot']

    fig = plt.figure(figsize=(vid.shape[-2]/dpi,vid.shape[-3]/dpi), dpi=dpi)
    im = plt.imshow(vid[0], interpolation='none')

    plt.axis('off')  # turn off axis ticks and lines
    plt.grid(False)  # disable grid
    plt.tight_layout(pad=0)  # remove padding

    def update(frame):
        im.set_data(vid[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(vid), interval=100)  # 100ms/frame
    ani.save(f'render/{test_seq}_Recon_anim_arifg{arifg}_{best_tag}_{tag}.gif', writer='pillow', fps=10)

    plt.close()  # Prevents duplicate plot
    return HTML(ani.to_jshtml())  # or ani.to_html5_video()


# %%
logger.record_ari(gs_recon,gs_mask,Ks,w2cs,ano)
ari = f'{(logger.ari/logger.ari_sample_len):.5f}'
arifg = f'{(logger.ari_fg/logger.ari_sample_len):.5f}'
print('ARI',ari)
print('ARI-FG',arifg)

# %%
# from sklearn.metrics import adjusted_rand_score
# render_mask = renderer.make_vid_slot(gs[0], gs_mask[0], Ks[0], w2cs[0], mask_as_color=True)
# pred = render_mask.argmax(dim=1).cpu().numpy()
# print(pred.shape)
# plt.imshow(pred[target_frame], cmap='inferno',alpha=fg_mask[target_frame].astype(float))
# pred_fg = pred[target_frame][fg_mask[target_frame]].flatten()
# anos_fg = ano[target_frame][fg_mask[target_frame]].squeeze(-1).flatten()
# arifg = adjusted_rand_score(anos_fg,pred_fg)
# print(arifg)

# %%
..

# %%
result, result_slots = logger.render_preview(gs_recon, gs_slot, gs_mask)
result[[2,3]] = result[[3,2]]
result[[4,5]] = result[[5,4]]
vid = logger.make_vid_grid(result,3,2).cpu()
plt_vid(vid.permute(0,2,3,1),arifg)

# %%
vid = logger.make_vid_grid(result_slots,4,5).cpu()
plt_vid(vid.permute(0,2,3,1),arifg,tag='slot')

# %%
# from gsplat.distributed import cli
# from simple_viewer import viewer
# from argparse import Namespace
# slot_num = 1
# out_slot = gs_slot[0][slot_num]
# out_slot = out_slot[:,:14]
# out_slot[:,10] = gs_slot[0,slot_num,:,14]
# args = Namespace(
#     port=8080,
#     gs = out_slot.detach().cpu()
# )

# viewer(args)

# %%
# !tensorboard --logdir=logs
