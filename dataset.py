import os
import random
import json
import numpy as np
import imageio
from typing import Literal, cast
import roma

from transforms import *
from renderer import *

import torch
# from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class Normalize(nn.Module):
    def __init__(self, dim=-1, p=2):
        super(Normalize, self).__init__()
        self.dim = dim
        self.p = p
    
    def forward(self, x):
        return F.normalize(x, dim=self.dim, p=self.p)

class ShapeOfMotion(Dataset):
    def __init__(self, data_dir, data_cfg, transform=None):
        self.data_dir = data_dir
        self.ckpt = torch.load(f"{data_dir}/checkpoints/last.ckpt") # If RAM OOM, could try dynamic load.
        self.img_dir = f"{data_dir}/images/"
        self.img_ext = os.path.splitext(os.listdir(self.img_dir)[0])[1]
        self.frame_names = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.img_dir))]
        self.imgs: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.renderer = Renderer(tuple(data_cfg.resolution), requires_grad=True)
        self.transform = transform
        self.feature_mask = [data_cfg.use_xyz,
                             data_cfg.use_rots,
                             data_cfg.use_scale,
                             data_cfg.use_opacity,
                             data_cfg.use_color,
                             data_cfg.use_motion]
        self.quat_activation = Normalize(dim=-1, p=2)
        self.color_activation = torch.sigmoid
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.motion_coef_activation = nn.Softmax(dim=-1)
        self.num_frame = len(self.frame_names)

    def __len__(self):
        return 1
    
    def get_image(self, index) -> torch.Tensor:
        if self.imgs[index] is None:
            self.imgs[index] = self.load_image(index)
        img = cast(torch.Tensor, self.imgs[index])
        return img
        
    def get_fg_4dgs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means, quats, scales, opacities, colors = self.load_3dgs('fg')

        means_4d = []
        quats_4d = []
        for ts in range(self.num_frame):
            transfms = self.get_transforms(torch.tensor([ts]))  # (G, B, 3, 4)
            means_ts = torch.einsum(
                "pnij,pj->pni",
                transfms,
                F.pad(means, (0, 1), value=1.0),
            ) # (G, B, 3)
            quats_ts = roma.quat_xyzw_to_wxyz(
                (
                    roma.quat_product(
                        roma.rotmat_to_unitquat(transfms[..., :3, :3]),
                        roma.quat_wxyz_to_xyzw(quats[:, None]),
                    )
                )
            )
            quats_ts = F.normalize(quats_ts, p=2, dim=-1) # (G, B, 4)
            
            means_4d.append(means_ts[:, 0])
            quats_4d.append(quats_ts[:, 0])

        means = torch.cat(means_4d, dim=-1)
        quats = torch.cat(quats_4d, dim=-1)
         
        return means, quats, scales, opacities, colors
    
    def get_all_4dgs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means,quats,scales,opacities,color = self.load_3dgs('bg')
        means = means.repeat(1,self.num_frame)
        quats = quats.repeat(1,self.num_frame)
        bg_gs = means,quats,scales,opacities,color
        fg_gs = self.get_fg_4dgs()
        return tuple(torch.cat([a, b]) for a, b in zip(bg_gs, fg_gs))

    def get_transforms(self, ts: torch.Tensor| None = None) -> torch.Tensor:
        transls, rots, coefs = self.load_motion_base()  # (K, B, 3), (K, B, 6), (G, K)
        transfms = compute_transforms(transls, rots, ts, coefs)  # (G, B, 3, 4)
        return transfms
    
    def get_zero_transform(self, size: int):
        transfms = np.eye(4)
        transfms = transfms[:-1].flatten()  # (12,)
        transfms = np.repeat(transfms[np.newaxis, :], size, axis=0) # (G, B, 3, 4)
        return transfms
    
    def load_image(self, index) -> torch.Tensor:
        path = f"{self.img_dir}/{self.frame_names[index]}{self.img_ext}"
        return torch.from_numpy(imageio.imread(path)).float() / 255.0
    
    def load_3dgs(self, set='fg') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert set in ['fg', 'bg']

        # Load params
        means = self.ckpt["model"][f"{set}.params.means"]
        quats = self.ckpt["model"][f"{set}.params.quats"]
        scales = self.ckpt["model"][f"{set}.params.scales"]
        opacities = self.ckpt["model"][f"{set}.params.opacities"]
        colors = self.ckpt["model"][f"{set}.params.colors"]
        
        # Process throgh activations
        quats = self.quat_activation(quats)
        scales = self.scale_activation(scales)
        opacities = self.opacity_activation(opacities)[:, None]
        colors = self.color_activation(colors)
        colors = torch.nan_to_num(colors, nan=1e-6)

        return means, quats, scales, opacities, colors

    def load_motion_base(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transls = self.ckpt["model"]["motion_bases.params.transls"]
        rots = self.ckpt["model"]["motion_bases.params.rots"]
        coefs = self.ckpt["model"]["fg.params.motion_coefs"]
        coefs = self.motion_coef_activation(coefs)
        return transls, rots, coefs
    
    def __getitem__(self, index: int):
        all_gs = self.get_all_4dgs()
        Ks = self.ckpt["model"]["Ks"][index].float()
        w2cs = self.ckpt["model"]["w2cs"][index]
        data = {
            # "gt_imgs": self.get_image(index),
            "gt_imgs": self.renderer.rasterize_gs(all_gs[0][:,0:3], all_gs[1][:,0:4], all_gs[-3], all_gs[-2], all_gs[-1], Ks, w2cs),
            # "fg_gs": self.get_fg_3dgs(torch.tensor([index])),
            "all_gs": all_gs,
            "feature_mask": self.feature_mask,
            "Ks": Ks,
            "w2cs": w2cs
        }
        return data
    

def collate_fn_padd(batch):
    
    gt_imgs = torch.stack([t['gt_imgs'] for t in batch])
    Ks = torch.stack([t['Ks'] for t in batch])
    w2cs = torch.stack([t['w2cs'] for t in batch])

    # Extract all_gs
    all_gs = []
    all_gs_pos = []
    all_mask = []
    for t in batch:
        selected = [k for k, m in zip(t['all_gs'], t['feature_mask']) if m]
        gs = torch.cat(selected, dim=-1)
        all_gs.append(gs)
        all_gs_pos.append(t['all_gs'][0])
        all_mask.append(torch.ones_like(gs, dtype=torch.float32))

    # Pad sequences along the first dimension (G)
    # batch_fg = torch.nn.utils.rnn.pad_sequence(fg_gs, batch_first=True, padding_value=0.0)
    all_gs = torch.nn.utils.rnn.pad_sequence(all_gs, batch_first=True, padding_value=0.0)
    all_gs_pos = torch.nn.utils.rnn.pad_sequence(all_gs_pos, batch_first=True, padding_value=0.0)

    # # # Compute mask (True for valid values, False for padding)
    # fg_mask = (batch_fg != 0).any(dim=-1)
    all_mask = torch.nn.utils.rnn.pad_sequence(all_mask, batch_first=True, padding_value=0.0)
    all_mask = all_mask.any(dim=-1)

    out = {
        "gt_imgs": gt_imgs,
        # "fg_gs": batch_fg,
        "all_gs": all_gs,
        "all_mask": all_mask,
        "all_gs_pos": all_gs_pos,
        "Ks": Ks,
        "w2cs": w2cs,
    }
    return out