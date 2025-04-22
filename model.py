from torch import nn
import torch
import torch.nn.functional as F
from renderer import *

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, mask, num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        if mask is not None:
            mask = mask.unsqueeze(-1)

        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', k, q) * self.scale
            attn = dots.softmax(dim=-1) + self.eps
            
            if mask is not None:
                attn = attn.masked_fill(mask == 0, 0)

            attn = attn / attn.sum(dim=-2, keepdim=True)
            updates = torch.einsum('bij,bid->bjd', attn, v)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots

class Gs_PositionEmbed(nn.Module):    
    def __init__(self, pos_dim, hidden_dim, feature_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        ### Try this!!!! ####
        # self.embedding = nn.Sequential(
        #     nn.Linear(pos_dim, feature_dim),
        # )

    def forward(self, input, pos):
        embedding = self.embedding(pos)
        return input + embedding

    
class Gs_Encoder(nn.Module):
    def __init__(self, gs_dim, hid_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(gs_dim, gs_dim*2),
            nn.ReLU(),
            nn.Linear(gs_dim*2, gs_dim*3),
            nn.ReLU(),
            nn.Linear(gs_dim*3, gs_dim*4),
            nn.ReLU(),
            nn.Linear(gs_dim*4, gs_dim*5),
        )
        self.encoder_pos = Gs_PositionEmbed(3, hid_dim, gs_dim*5)

    def forward(self, x, pos):
        x = self.mlp(x)
        x = self.encoder_pos(x, pos)
        return x
    
class Gs_Decoder(nn.Module):
    def __init__(self, gs_dim, hid_dim):
        super(Gs_Decoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(gs_dim),
            nn.Linear(gs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.encoder_pos = Gs_PositionEmbed(3, hid_dim, gs_dim)


    def forward(self, slots, pos) -> tuple[torch.Tensor,torch.Tensor]:
        slots = self.encoder_pos(slots, pos)
        out = self.mlp(slots)
        colors = out[:,:,:,:3]
        mask = out[:,:,:,3:4]
        colors = torch.sigmoid(colors)
        mask = torch.softmax(mask, dim=1)
        return colors, mask # (B, N_S, G, 4)
    
class Gs_Color_Decoder(nn.Module):
    def __init__(self, gs_dim, hid_dim):
        super(Gs_Color_Decoder, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(gs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.norm1 = nn.LayerNorm(3)
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(3, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 3),
        # )
        # self.norm2 = nn.LayerNorm(3)
        self.encoder_pos = Gs_PositionEmbed(3, hid_dim, gs_dim)
        self.gate = nn.Parameter(torch.tensor(0.5))


    def forward(self, slots, colors, pos) -> torch.Tensor:
        slots = self.encoder_pos(slots, pos)
        offset = self.mlp1(slots)
        offset = self.norm1(offset)
        colors = colors * (1 - self.gate) + offset * self.gate
        colors =torch.sigmoid(colors)
        return colors # (B, N_S, N_G, 3)
    
class Gs_Mask_Decoder(nn.Module):
    def __init__(self, gs_dim, hid_dim):
        super(Gs_Mask_Decoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(gs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.encoder_pos = Gs_PositionEmbed(3, hid_dim, gs_dim)


    def forward(self, slots, pos) -> torch.Tensor:
        slots = self.encoder_pos(slots, pos)
        mask = self.mlp(slots)
        mask = torch.softmax(mask, dim=1)
        return mask # (B, N_S, N_G, 1)

class SlotAttentionAutoEncoder(nn.Module):
    # def __init__(self, resolution, num_slots, num_iters, hid_dim):
    def __init__(self, data_cfg, cnn_cfg, attn_cfg):

        super().__init__()
        self.hid_dim = cnn_cfg.hid_dim
        self.gs_pos_embed = cnn_cfg.gs_pos_embed
        self.encode_gs = cnn_cfg.encode_gs
        self.resolution = tuple(data_cfg.resolution)
        self.num_slots = attn_cfg.num_slots
        self.num_iters =  attn_cfg.num_iters

        self.feature_mask = torch.tensor([data_cfg.use_xyz,
                                        data_cfg.use_rots,
                                        data_cfg.use_scale,
                                        data_cfg.use_opacity,
                                        data_cfg.use_color,
                                        data_cfg.use_motion], dtype=torch.bool)
        feature_len = torch.tensor([3, 4, 3, 1, 3, 3], dtype=torch.int32)
        gs_dim = 14

        self.slot_norm = nn.LayerNorm(gs_dim)

        self.encoder_gs = Gs_Encoder(gs_dim, self.hid_dim)
        # self.decoder = Gs_Decoder(gs_dim, self.hid_dim)
        # self.color_decoder = Gs_Color_Decoder(gs_dim, self.hid_dim)
        self.mask_decoder = Gs_Mask_Decoder(gs_dim, self.hid_dim)
        
        self.encoder_pos = Gs_PositionEmbed(3, self.hid_dim, gs_dim)
        
        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=gs_dim,
            iters=self.num_iters,
            eps = 1e-8, 
            hidden_dim = 256)
        
        self.renderer = Renderer(tuple(data_cfg.resolution), requires_grad=True)

    def forward(self, gs:torch.Tensor, pos:torch.Tensor, Ks:torch.Tensor, w2cs:torch.Tensor, mask=None, inference=False):
        # gs: [B, G, D]
        # pos: [B, G, 3]
        # mask: [B, G]
        B,G,D = gs.shape

        pos = pos.unsqueeze(1)
        # x = self.encoder_gs(gs, pos)
        x = gs

        # Slot Attention module.
        slots = self.slot_attention(x, mask) # [B, N_S, D]

        # Retrieve slots color and Min-Max norm
        colors = slots[:,:,11:14]
        # colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8) # [B, N_S, 3]

        colors = colors.unsqueeze(-2)
        colors = colors.repeat(1,1,G,1) # [B, N_S, G, 3]
        color_code = colors.repeat(1,1,self.num_slots,1) # [B, N_S, N_S, 3]
        color_code = (color_code - color_code.min()) / (color_code.max() - color_code.min() + 1e-8) # [B, N_S, 3]

        # Broadcast slots to all pos
        slots = slots.unsqueeze(-2) # [B, N_S, D]
        slots = slots.repeat(1,1,G,1) # [B, N_S, G, D]

        # Copy gs to match slots count
        gs_slot = gs.unsqueeze(1).repeat(1,self.num_slots,1,1) # [B, N_S, G, D]

        # Apply original textures to colors
        gray_weights = torch.tensor([0.299, 0.587, 0.114], device=gs_slot.device)
        textures = (gs_slot[:,:,:,11:14] * gray_weights).sum(dim=-1, keepdim=True)  # [B, N_S, G, 1]
        colors = colors * textures


        # # LayerNorm slots
        # slots = self.slot_norm(slots)

        # # MLP detection head for color
        # colors = self.color_decoder(slots, colors, pos) # [B, N_S, G, 3]

        # MLP detection head for mask
        color_mask = self.mask_decoder(slots, pos) # [B, N_S, G, 1]

        # MLP detection head for color and mask
        # colors, color_mask = self.decoder(slots,pos)
        
        gs_slot = torch.cat([gs_slot[:,:,:,:10],color_mask,colors], dim=-1)
        colors = torch.sum(colors * color_mask, dim=1)
        
        gs[:,:,10] = 1.0
        gs = torch.cat([gs[:,:,:11],colors], dim=-1)

        # 3D Gaussian renderer
        recon_combined = []
        recon_slots = []

        for batch,ks,w2c in zip(gs,Ks,w2cs):
            means, quats, scales, opacities, colors = torch.split(batch, [3,4,3,1,3], dim=-1)
            recon_combined.append(self.renderer.rasterize_gs(means, quats, scales, opacities, colors,ks,w2c))
        recon_combined = torch.stack(recon_combined,dim=0)

        if inference:
            for batch,ks,w2c in zip(gs_slot,Ks,w2cs):
                for slot in batch:
                    means, quats, scales, opacities, colors = torch.split(slot, [3,4,3,1,3], dim=-1)
                    recon_slots.append(self.renderer.rasterize_gs(means, quats, scales, opacities, colors,ks,w2c))
            recon_slots = torch.stack(recon_slots,dim=0)

        return recon_combined, recon_slots, color_code, gs, gs_slot