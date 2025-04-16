import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization_2dgs
from scipy.spatial.transform import Rotation as R

class Renderer():
    def __init__(self, resolution: tuple[int,int], requires_grad=False):
        self.resolution = resolution
        self.requires_grad = requires_grad
        pass

    def rasterize_gs(self, means, quats, scales, opacities, colors, Ks, w2cs, w=None, h=None) -> torch.Tensor:
        device=w2cs.device
        bg_color = torch.full((1, 3), 0.0, device=device).float()
        Ks = Ks.unsqueeze(0)
        w2cs = w2cs.unsqueeze(0)

        if w is None:
            w = self.resolution[0]
        if h is None:
            h = self.resolution[1]

        (
            render_colors,
            alphas,
            render_normals,
            surf_normals,
            _,
            _,
            info
            ) = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities.squeeze(-1),
            colors=colors,
            viewmats=w2cs,
            backgrounds=None,
            Ks=Ks,
            width=w,
            height=h,
            packed=False,
            render_mode="RGB")

        # render_colors, alphas, info = rasterization(means,quats,scales,opacities,colors,w2c,Ks,W,H)

        img = torch.cat([render_colors,alphas], dim=-1)

        if self.requires_grad:
            img = render_colors[0]
        else:
            img = (render_colors[0]* 255.0).to(torch.uint8)

        return img
        
    def rotate_cam(self, w2cs_init: torch.Tensor, rot_vec: tuple[int,int,int]) -> torch.Tensor:
        device=w2cs_init.device

        # 30 degrees in radians
        rot_x = math.radians(rot_vec[0])
        rot_y = math.radians(rot_vec[1])
        rot_z = math.radians(rot_vec[2])

        # Rotation matrix around Z axis
        Rx = torch.tensor(R.from_euler('x', rot_x).as_matrix(), device=device)
        Ry = torch.tensor(R.from_euler('y', rot_y).as_matrix(), device=device)
        Rz = torch.tensor(R.from_euler('z', rot_z).as_matrix(), device=device)

        return w2cs_init @ Rz @ Ry @ Rx
    
    def random_cam_view(self, w2cs_init: torch.Tensor, min: int, max: int) -> torch.Tensor:
        rot_vec = (0, 0, random.randint(-min,max))
        return self.rotate_cam(w2cs_init, rot_vec)
