import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization_2dgs
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

class Renderer():
    def __init__(self, resolution: tuple[int,int], frame_num, requires_grad=False):
        self.resolution = resolution
        self.requires_grad = requires_grad
        self.frame_num = frame_num
        pass

    def rasterize_4dgs(self, gs_4d:tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Ks, w2cs, w=None, h=None, alpha=False) -> torch.Tensor:
        frame = self.frame_num
        scales = gs_4d[2]
        opacities = gs_4d[3]
        colors = gs_4d[4]

        rendered_vid = []
        for i in range(frame):
            means = gs_4d[0][:,3*i:3*i+3]
            quats = gs_4d[1][:,4*i:4*i+4]
            render = self.rasterize_gs((means,quats,scales,opacities,colors), Ks, w2cs,w,h,alpha)
            rendered_vid.append(render)
        rendered_vid = torch.stack(rendered_vid)
        return rendered_vid       

    def split_4dgs(self, gs_4d:torch.Tensor):
        assert len(gs_4d.shape) == 2
        G,_ = gs_4d.shape
        means = gs_4d[:,:3*self.frame_num].reshape(G, self.frame_num, 3)
        quats = gs_4d[:,3*self.frame_num:-7].reshape(G, self.frame_num, 4)
        others = gs_4d[:,-7:].unsqueeze(1).expand(-1, self.frame_num, -1)
        gs = torch.cat([means,quats,others],dim=-1).permute(1,0,2)
        return gs

    def rasterize_gs(self, gs:tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Ks, w2cs, w=None, h=None, alpha=False) -> torch.Tensor:
        device=w2cs.device
        
        if not self.requires_grad:
            for attri in gs:
                attri = attri.detach()
                attri.requires_grad_(False)

        if len(w2cs.shape) == 2:
            Ks = Ks.unsqueeze(0)
            w2cs = w2cs.unsqueeze(0)

        # bg_color = torch.ones(Ks.shape[0], 3, dtype=torch.float32,device=device) * 1.0
        # bg_color.requires_grad_(False)

        bg_color = None

        if w is None:
            w = self.resolution[0]
        if h is None:
            h = self.resolution[1]

        (
            render_colors,
            alphas,
            _,
            _,
            _,
            _,
            _
            ) = rasterization_2dgs(
            means=gs[0],
            quats=gs[1],
            scales=gs[2],
            opacities=gs[3].squeeze(-1),
            colors=gs[4],
            viewmats=w2cs,
            backgrounds=bg_color,
            Ks=Ks,
            width=w,
            height=h,
            packed=False,
            render_mode="RGB")

        # render_colors, alphas, info = rasterization(means,quats,scales,opacities,colors,w2c,Ks,W,H)
        # render_colors = torch.cat([render_colors,alphas], dim=-1)

        if alpha:
            render_colors = torch.cat([render_colors,alphas], dim=-1)

        if not self.requires_grad:
            render_colors = render_colors.detach()
            render_colors.requires_grad_(False)

        if render_colors.shape[0] == 1:
            return render_colors[0]
        else:
            return render_colors
        
    def rotate_matrix(self, degree: int, axis:str, device) -> torch.Tensor:
        assert axis in ['x','y','z']        
        # Convert to radians
        rad = math.radians(degree)
        # Rotation matrix around Z axis
        r_mat3 = torch.tensor(R.from_euler(axis, rad).as_matrix(), device=device)
        # Create a 4x4 identity matrix
        r_mat4 = torch.eye(4, dtype=torch.float32, device=device)
        # Insert 3x3 rotation into top-left of 4x4 matrix
        r_mat4[:3, :3] = r_mat3

        return r_mat4
    
    def random_cam_view(self, w2cs_init: torch.Tensor, min: int, max: int) -> torch.Tensor:
        rot_w2cs = w2cs_init @ self.rotate_matrix(random.randint(-min,max), 'z', w2cs_init.device)
        return rot_w2cs
    
    def simple_track(self, Ks_init: torch.Tensor, w2cs_init: torch.Tensor, frames: int, axis: str) -> tuple[torch.Tensor, torch.Tensor]:
        Ks, w2cs = [], []
        rot_w2cs = w2cs_init
        for _ in range(frames):
            Ks.append(Ks_init)
            w2cs.append(rot_w2cs)
            rot_w2cs = rot_w2cs @ self.rotate_matrix(2, axis, rot_w2cs.device)
        Ks = torch.stack(Ks)
        w2cs = torch.stack(w2cs)

        return Ks, w2cs        
    
def render_batch(renderer:Renderer, batch_gs, batch_slots, batch_mask, batch_Ks, batch_w2cs):
    recon_combined = []
    for gs,ks,w2cs in zip(batch_gs,batch_Ks,batch_w2cs):
        gs = tuple(torch.split(gs, [3,4,3,1,3], dim=-1))
        recon_combined.append(renderer.rasterize_gs(gs,ks,w2cs)[...,0:3])
    recon_combined = torch.stack(recon_combined,dim=0)

    recon_slots = []
    for slots,mask,ks,w2cs in zip(batch_slots,batch_mask,batch_Ks,batch_w2cs):
        render_slots = []
        for gs,alpha in zip(slots,mask):
            means, quats, scales, _, colors = torch.split(gs, [3,4,3,1,3], dim=-1)
            render_slots.append(renderer.rasterize_gs((means, quats, scales, alpha, colors),ks,w2cs,alpha=True)[...,0:3])
        recon_slots.append(torch.stack(render_slots,dim=0))
    recon_slots = torch.stack(recon_slots,dim=0)
    
    # slots_alpha = []
    # slots = torch.cat([slot, mask, mask, mask], dim=-1) # [B, N_S, G, D+3]
    # for batch,ks,w2c in zip(slots,Ks,w2cs):
    #     for slot in batch:
    #         means, quats, scales, opacities, colors, alpha = torch.split(slot, [3,4,3,1,3,3], dim=-1)
    #         recon_slots.append(renderer.rasterize_gs((means, quats, scales, opacities, colors),ks,w2c))
    #         slots_alpha.append(renderer.rasterize_gs((means, quats, scales, torch.ones_like(opacities), alpha),ks,w2c))

    # recon_slots = torch.stack(recon_slots,dim=0)[...,0:3]
    # slots_alpha = torch.stack(slots_alpha,dim=0)[...,0:1]
    # recon_slots = torch.cat([recon_slots, slots_alpha], dim=-1)

    return recon_combined, recon_slots

def render_single(renderer:Renderer, gs, slot, mask, Ks:torch.Tensor, w2cs:torch.Tensor, color_code=False):
    gs = gs[0]
    slot = slot[0]
    mask = mask[0]
    Ks = Ks[0]
    w2cs = w2cs[0]   
    
    num_slot = slot.shape[0]
    color_unit = 1.0 / float(num_slot-1)
    cmap = plt.get_cmap('inferno')
    if color_code:
        color_code = []
        for k in range(num_slot):
            rgb = cmap(torch.tensor([k * color_unit],dtype=torch.float32))[0,0:3]
            color_code.append(torch.from_numpy(rgb).to(slot.device))
        color_code = torch.stack(color_code) # [N_S,3]
        color_code = color_code[:, None, :]
        slot[..., -3:] = color_code
        gs = torch.sum(slot * mask, dim=0)

    recon_combined = renderer.rasterize_gs(gs,Ks,w2cs)

    recon_slots = []
    for gs,alpha in zip(slot,mask):
        means, quats, scales, _, colors = torch.split(gs, [3,4,3,1,3], dim=-1)
        recon_slots.append(renderer.rasterize_gs((means, quats, scales, alpha, colors),Ks,w2cs,alpha=True))

    # slots_alpha = []
    # mask_slot = torch.cat([slot, mask, mask, mask], dim=-1) # [N_S, G, D+3]
    # for gs in mask_slot:
    #     means, quats, scales, opacities, colors, alpha = torch.split(gs, [3,4,3,1,3,3], dim=-1)
    #     recon_slots.append(renderer.rasterize_gs((means, quats, scales, opacities, colors),Ks,w2cs))
    #     slots_alpha.append(renderer.rasterize_gs((means, quats, scales, opacities, alpha),Ks,w2cs))

    recon_slots = torch.stack(recon_slots,dim=0)
    # recon_slots = torch.stack(recon_slots,dim=0)[...,0:3]
    # slots_alpha = torch.stack(slots_alpha,dim=0)[...,0:1]
    # recon_slots = torch.cat([recon_slots, slots_alpha], dim=-1)

    return recon_combined, recon_slots

def render_single_vid(renderer:Renderer, gs, slot, mask, Ks:torch.Tensor, w2cs:torch.Tensor, render_rotate=False, color_code=False):
    '''
    gs: [1,G,F*7+7]
    slot: [1,N_S,G,F*7+7]
    '''

    gs = gs[0]
    slot = slot[0]
    mask = mask[0]
    Ks = Ks[0]
    w2cs = w2cs[0]

    if render_rotate:
        Ks, w2cs = renderer.simple_track(Ks, w2cs, renderer.frame_num, 'z')
    else:
        Ks = Ks.unsqueeze(0).expand(renderer.frame_num,-1,-1)
        w2cs = w2cs.unsqueeze(0).expand(renderer.frame_num,-1,-1)

    num_slot = slot.shape[0]
    color_unit = 1.0 / float(num_slot-1)
    cmap = plt.get_cmap('inferno')
    if color_code:
        color_code = []
        for k in range(num_slot):
            rgb = cmap(torch.tensor([k * color_unit],dtype=torch.float32))[0,0:3]
            color_code.append(torch.from_numpy(rgb).to(slot.device))
        color_code = torch.stack(color_code) # [N_S,3]
        color_code = color_code[:, None, :]
        slot[..., -3:] = color_code
        gs = torch.sum(slot * mask, dim=0)
    
    mask = mask.unsqueeze(0).expand(renderer.frame_num,-1,-1,-1)

    gs_vid = renderer.split_4dgs(gs) # [F,G,D]
    slot_vid = []
    for k in slot:
        slot_vid.append(renderer.split_4dgs(k))
    slot_vid = torch.stack(slot_vid) # [N_S,F,G,D]
    slot_vid = slot_vid.permute(1,0,2,3) # [F,N_S,G,D]

    return render_batch(renderer,gs_vid,slot_vid,mask,Ks,w2cs)   
