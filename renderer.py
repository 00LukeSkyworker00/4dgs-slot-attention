import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization_2dgs

class Renderer():
    def __init__(self):
        pass

    def rasterize(self, gs):
        frame = 0

        Ks = ckpt["model"]["Ks"][frame].unsqueeze(0).float()
        w2cs = ckpt["model"]["w2cs"][frame].unsqueeze(0)
        device=w2cs.device

        # 30 degrees in radians
        angle = math.radians(-30)

        # Rotation matrix around Z axis
        Rz = torch.tensor([
            [math.cos(angle), -math.sin(angle), 0, 0],
            [math.sin(angle),  math.cos(angle), 0, 0],
            [0,               0,               1, 0],
            [0,               0,               0, 1],
        ], device=device)

        w2cs = w2cs @ Rz

        bg_color = torch.full((1, 3), 0.0, device=device).float()

        W, H = (128,128)
        x = torch.randn((100, 3), device=device)

        (render_colors,alphas,render_normals,surf_normals,_,_,info) = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=w2cs,
            backgrounds=bg_color,
            Ks=Ks,
            width=W,
            height=H,
            packed=False,
            render_mode="RGB")

        # render_colors, alphas, info = rasterization(means,quats,scales,opacities,colors,w2c,Ks,W,H)

        img = (render_colors[0]* 255.0).to(torch.uint8)
