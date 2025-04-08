import os
import random
import json
import numpy as np
import imageio
from typing import Literal, cast
import roma

from transforms import *

import torch
# from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class ShapeOfMotion(Dataset):
    def __init__(self, data_dir, data_cfg, transform=None):        
        self.data_dir = data_dir
        self.ckpt = torch.load(f"{data_dir}/checkpoints/last.ckpt") # If RAM OOM, could try dynamic load.
        self.img_dir = f"{data_dir}/images/"
        self.img_ext = os.path.splitext(os.listdir(self.img_dir)[0])[1]
        self.frame_names = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.img_dir))]
        self.imgs: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.transform = transform
        self.use_xyz = bool(data_cfg.use_xyz)
        self.use_rots = bool(data_cfg.use_rots)
        self.use_scale = bool(data_cfg.use_scale)
        self.use_opacity = bool(data_cfg.use_opacity)
        self.use_color = bool(data_cfg.use_color)
        self.use_motion = bool(data_cfg.use_motion)

    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    def __len__(self):
        return len(self.frame_names)
    
    def get_image(self, index) -> torch.Tensor:
        if self.imgs[index] is None:
            self.imgs[index] = self.load_image(index)
        img = cast(torch.Tensor, self.imgs[index])
        return img
        
    def get_fg_3dgs(self, ts: torch.Tensor) -> torch.Tensor:
        means, quats, scales, opacities, colors = self.load_3dgs('fg')

        if ts is not None:
            transfms = self.get_transforms(ts)  # (G, B, 3, 4)
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
            means_ts = means_ts[:, 0]
            quats_ts = quats_ts[:, 0]
        else:
            means_ts = means
            quats_ts = quats
         
        result = []
        if self.use_xyz:
            result.append(means_ts)
        if self.use_rots:
            result.append(quats_ts)
        if self.use_scale:
            result.append(scales)
        if self.use_opacity:
            result.append(opacities)
        if self.use_color:
            result.append(colors)
        if self.use_motion:
            pass

        return torch.cat(result, dim=1)
    
    def get_all_3dgs(self, ts: torch.Tensor) -> torch.Tensor:
        means, quats, scales, opacities, colors = self.load_3dgs('bg')
        bg_cat = []
        if self.use_xyz:
            bg_cat.append(means)
        if self.use_rots:
            bg_cat.append(quats)
        if self.use_scale:
            bg_cat.append(scales)
        if self.use_opacity:
            bg_cat.append(opacities)
        if self.use_color:
            bg_cat.append(colors)
        if self.use_motion:
            pass
        bg_cat = torch.cat(bg_cat, dim=1)
        fg_cat = self.get_fg_3dgs(ts)
        return torch.cat((bg_cat, fg_cat), dim=0)

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
        means = self.ckpt["model"][f"{set}.params.means"]
        quats = self.ckpt["model"][f"{set}.params.quats"]
        scales = self.ckpt["model"][f"{set}.params.scales"]
        opacities = self.ckpt["model"][f"{set}.params.opacities"][:, None]
        # print('opacities loaded', opacities.shape)
        colors = self.ckpt["model"][f"{set}.params.colors"]
        colors = torch.nan_to_num(colors, posinf=5, neginf=-5)
        return means, quats, scales, opacities, colors
    
    def load_3dgs_norm(self, set='fg') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_3dgs = []
        for tensor in self.load_3dgs(set):
            norm_3dgs.append(self.min_max_norm(tensor))
        return tuple(norm_3dgs)

    def load_motion_base(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transls = self.ckpt["model"]["motion_bases.params.transls"]
        rots = self.ckpt["model"]["motion_bases.params.rots"]
        coefs = self.ckpt["model"]["fg.params.motion_coefs"]
        return transls, rots, coefs
    
    # def min_max_norm(self, tensor: torch.Tensor) -> torch.Tensor:
    #     min_val = tensor.min()
    #     max_val = tensor.max()
    #     return (tensor - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
    
    def __getitem__(self, index: int):

        bg_means = self.ckpt["model"][f"bg.params.means"]
        fg_means = self.ckpt["model"][f"fg.params.means"]

        data = {
            # (H, W, 3).
            "gt_imgs": self.get_image(index),
            # (G, 14).
            # "fg_gs": self.get_fg_3dgs(torch.tensor([index])), 
            # (G, 14).
            "all_gs": self.get_all_3dgs(torch.tensor([index])),
            # (G, 3)
            "all_gs_pos": torch.cat((bg_means, fg_means), dim=0)
        }
        return data
    

def collate_fn_padd(batch):
    
    gt_imgs = [torch.tensor(t['gt_imgs'], dtype=torch.float32) for t in batch]  # Keep gt_imgs as is (no padding)
    gt_imgs = torch.stack(gt_imgs)
    

    # Extract fg_gs, all_gs
    # fg_gs = [torch.tensor(t['fg_gs'], dtype=torch.float32) for t in batch]
    all_gs = [torch.tensor(t['all_gs'], dtype=torch.float32) for t in batch]
    all_gs_pos = [torch.tensor(t['all_gs_pos'], dtype=torch.float32) for t in batch]

    # Pad sequences along the first dimension (G)
    # batch_fg = torch.nn.utils.rnn.pad_sequence(fg_gs, batch_first=True, padding_value=0.0)
    all_gs = torch.nn.utils.rnn.pad_sequence(all_gs, batch_first=True, padding_value=0.0)
    all_gs_pos = torch.nn.utils.rnn.pad_sequence(all_gs_pos, batch_first=True, padding_value=0.0)

    # # # Compute mask (True for valid values, False for padding)
    # fg_mask = (batch_fg != 0).any(dim=-1)
    all_mask = [torch.ones_like(t['all_gs'], dtype=torch.float32) for t in batch]
    all_mask = torch.nn.utils.rnn.pad_sequence(all_mask, batch_first=True, padding_value=0.0)
    all_mask = all_mask.any(dim=-1)

    out = {
        "gt_imgs": gt_imgs,
        # "fg_gs": batch_fg,
        "all_gs": all_gs,
        "all_mask": all_mask,
        "all_gs_pos": all_gs_pos,
    }
    return out