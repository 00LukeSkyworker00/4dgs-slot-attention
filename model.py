from torch import nn
import torch
import torch.nn.functional as F
from renderer import *

class SlotAttention(nn.Module):
    def __init__(self, num_slots, slot_dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, slot_dim))

        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(slot_dim, slot_dim)
        self.to_v = nn.Linear(slot_dim, slot_dim)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.fc1 = nn.Linear(slot_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, slot_dim)

        self.norm_input  = nn.LayerNorm(slot_dim)
        self.norm_slots  = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

    def forward(self, inputs, mask, num_slots = None):        
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        if mask is not None:
            mask = mask.unsqueeze(-1)

        inputs = self.norm_input(inputs)
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

class Gs_Embedding(nn.Module):    
    def __init__(self, pos_dim, feature_dim, use_fourier=False):
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.embedding = FourierEmbedding(pos_dim, 6, include_input=False)
            self._out_dim = feature_dim + self.embedding.out_dim
        else:
            self.embedding = nn.Linear(pos_dim, feature_dim)
            self._out_dim = feature_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, input, pos):
        # pos_min = pos.amin(dim=-2, keepdim=True).detach()
        # pos_max = pos.amax(dim=-2, keepdim=True).detach()
        # pos_norm = (pos.detach() - pos_min) / (pos_max - pos_min)
        # pe = self.embedding(pos_norm)
        pe = self.embedding(pos)
        if self.use_fourier:
            return torch.cat([input, pe], dim=-1)
        else:
            return input + pe
        
class TriPlaneEmbedding(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.xy = nn.Linear(2, feature_dim)
        self.yz = nn.Linear(2, feature_dim)
        self.zx = nn.Linear(2, feature_dim)

    def forward(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = pos[:,:,:,0:1]
        y = pos[:,:,:,1:2]
        z = pos[:,:,:,2:3]
        xy = self.xy(torch.cat([x,y], dim=-1).detach())
        yz = self.yz(torch.cat([y,z], dim=-1).detach())
        zx = self.zx(torch.cat([x,z], dim=-1).detach())
        return xy, yz, zx


class FourierEmbedding(nn.Module):
    """
    Fourier-feature positional embedding.
    Maps an input tensor of shape (..., D) to (..., D * (1 + 2 * num_freqs))
    by concatenating [x, sin(2^i * x), cos(2^i * x) for i in 0..num_freqs-1].
    """
    def __init__(self, input_dim: int,num_freqs: int, include_input: bool = True):
        super().__init__()
        self.dim = input_dim
        self.num_freqs = num_freqs
        self.include_input = include_input
        # Create frequency bands [1, 2, 4, ..., 2^(num_freqs-1)]
        freq_bands = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer('freq_bands', freq_bands)

    @property
    def out_dim(self) -> int:
        # If include_input: D + 2 * D * num_freqs, else 2 * D * num_freqs
        return self.dim * (1 + 2 * self.num_freqs) if self.include_input else self.dim * 2 * self.num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., D) tensor of input coordinates (e.g. 3D positions)
        returns: (..., D * (1 + 2 * num_freqs)) tensor of embeddings
        """
        embeds = []
        if self.include_input:
            embeds.append(x)

        # compute sin and cos of each frequency band
        for freq in self.freq_bands:
            embeds.append(torch.sin(x * freq))
            embeds.append(torch.cos(x * freq))

        return torch.cat(embeds, dim=-1)

class Gs_Decoder(nn.Module):
    def __init__(self, slot_dim, hid_dim):
        super(Gs_Decoder, self).__init__()

        # self.mlp_head = nn.Sequential(
        #     nn.Linear(slot_dim, hid_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hid_dim,14+1)
        # )

        # self.pcm_head = nn.Sequential(
        #     nn.Linear(slot_dim, hid_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hid_dim,3+3+1)
        # )

        self.pcm_head = TriPlane_Decoder(slot_dim, hid_dim, 1+3+1)

        # self.embedding = Gs_Embedding(3, slot_dim, use_fourier=False)

        # self.pos_head = Pos_Decoder(slot_dim, hid_dim)
        # self.col_head = Col_Decoder(slot_dim, hid_dim)
        # self.mask_head = Gs_Mask_Decoder(slot_dim, hid_dim)

    def forward(self, x, pos) -> torch.Tensor:

        # # Shared mlp
        # gs = self.mlp_head(x)
        # gs, mask = torch.split(gs, [14,1], dim=-1)
        # mask = F.softmax(mask, dim=1)

        # Shared mlp for pos col mask
        x_out, y_out, z_out = self.pcm_head(x, pos)
        pos = torch.cat([x_out[:,:,:,0:1],y_out[:,:,:,0:1],z_out[:,:,:,0:1]], dim=-1)
        out = x_out[:,:,:,1:5] + y_out[:,:,:,1:5] + z_out[:,:,:,1:5]
        color, mask = torch.split(out, [3,1], dim=-1)
        mask = F.softmax(mask, dim=1)

        # x_embed = self.embedding(x, pos)
        # pos = self.pos_head(x, pos)
        # color = self.col_head(x_embed)
        # mask = self.mask_head(x_embed)

        return pos, color, mask # (B, N_S, G, 14), (B, N_S, G, 1)
        return gs, mask # (B, N_S, G, 14), (B, N_S, G, 1)

class Pos_Decoder(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(Pos_Decoder, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, 3)
        )
    def forward(self, x) -> torch.Tensor:
        pos = self.mlp_head(x)
        return pos # (B, N_S, G, 3)
    
class TriPlane_Decoder(nn.Module):
    def __init__(self, input_dim, hid_dim, outdim):
        super(TriPlane_Decoder, self).__init__()
        self.x_head = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, outdim)
        )
        self.y_head = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, outdim)
        )
        self.z_head = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, outdim)
        )
        self.embedding = TriPlaneEmbedding(input_dim)
    def forward(self, input, pos) -> torch.Tensor:
        xy, yz, zx = self.embedding(pos)
        x = self.x_head(input+yz)
        y = self.y_head(input+zx)
        z = self.z_head(input+xy)
        return x, y, z
    
class Col_Decoder(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(Col_Decoder, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, 3)
        )
    def forward(self, x) -> torch.Tensor:
        color = self.mlp_head(x)
        return color # (B, N_S, G, 3)
    
class Gs_Mask_Decoder(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(Gs_Mask_Decoder, self).__init__()
        self.mask_head = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, x) -> torch.Tensor:
        mask = self.mask_head(x)
        mask = F.softmax(mask, dim=1)
        return mask # (B, N_S, G, 1)        

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

        # self.feature_mask = torch.tensor([data_cfg.use_xyz,
        #                                 data_cfg.use_rots,
        #                                 data_cfg.use_scale,
        #                                 data_cfg.use_opacity,
        #                                 data_cfg.use_color,
        #                                 data_cfg.use_motion], dtype=torch.bool)
        gs_dim = 14
        slot_dim = 96

        self.encoder = nn.Linear(gs_dim, slot_dim)
        # self.encode_embedding = Gs_Embedding(3, slot_dim, use_fourier=False)
        # slot_dim = self.encode_embedding.out_dim
        # self.encode_norm = nn.LayerNorm(slot_dim)
        
        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            slot_dim=slot_dim,
            iters=self.num_iters,
            eps = 1e-8, 
            hidden_dim = 128)
        
        self.decoder = Gs_Decoder(slot_dim, 96)
        
        self.renderer = Renderer(tuple(data_cfg.resolution), requires_grad=True)

    def forward(self, gs:torch.Tensor, pe:torch.Tensor, Ks:torch.Tensor, w2cs:torch.Tensor, mask=None, inference=False):
        """
        gs: [B, G, D]
        pos: [B, G, 3]
        mask: [B, G]
        """
        _,G,_ = gs.shape

        # Gs encoder to match slot dim
        # gs = torch.cat([gs[:,:,0:3],gs[:,:,11:14]], dim=-1) # [B, G, 6]
        x = self.encoder(gs)
        # x = self.encode_embedding(x, pos) # [B, N_S, G, D]
        # x = self.encode_norm(x) # [B, N_S, G, D]

        # x = gs

        # Slot Attention module.
        slots = self.slot_attention(x, mask) # [B, N_S, D]

        # Broadcast pos to all slots
        pe = pe.unsqueeze(1)
        pe = pe.repeat(1,self.num_slots,1,1) # [B, N_S, G, 3]

        # Broadcast slots to all points
        slots = slots.unsqueeze(-2) # [B, N_S, D]
        slots = slots.repeat(1,1,G,1) # [B, N_S, G, D]

        # MLP detection head for color and mask
        # gs_slot, gs_mask = self.decoder(slots)
        pos, color, gs_mask = self.decoder(slots, pe)
        
        # Copy gs to match slots count
        gs_slot = gs.unsqueeze(1).repeat(1,self.num_slots,1,1) # [B, N_S, G, D]
        # Inject decoded gs into gs_slot
        gs_slot = torch.cat([pos, gs_slot[:,:,:,3:11], color], dim=-1) # [B, N_S, G, D]

        # Recconstruct gs
        gs_out = torch.sum(gs_slot * gs_mask, dim=1)
        # print("gs: ", torch.isnan(gs).any())

        # 3D Gaussian renderer
        recon_combined = []
        recon_slots = []
        slots_alpha = []

        color_code = None

        if inference:
            gs_out = gs_out.detach()
            gs_slot = gs_slot.detach()
            gs_mask = gs_mask.detach()
            Ks = Ks.detach()
            w2cs = w2cs.detach()

            gs_slot = torch.cat([gs_slot, gs_mask, gs_mask, gs_mask], dim=-1) # [B, N_S, G, D+3]

            for batch,ks,w2c in zip(gs_out,Ks,w2cs):
                means, quats, scales, opacities, colors = torch.split(batch, [3,4,3,1,3], dim=-1)
                recon_combined.append(self.renderer.rasterize_gs(means, quats, scales, opacities, colors,ks,w2c))
            recon_combined = torch.stack(recon_combined,dim=0)[:,:,:,0:3]

            for batch,ks,w2c in zip(gs_slot,Ks,w2cs):
                for slot in batch:
                    means, quats, scales, opacities, colors,alpha = torch.split(slot, [3,4,3,1,3,3], dim=-1)
                    recon_slots.append(self.renderer.rasterize_gs(means, quats, scales, opacities, colors,ks,w2c))
                    slots_alpha.append(self.renderer.rasterize_gs(means, quats, scales, opacities, alpha,ks,w2c))

            recon_slots = torch.stack(recon_slots,dim=0)[:,:,:,0:3]
            slots_alpha = torch.stack(slots_alpha,dim=0)[:,:,:,0:1]
            recon_slots = torch.cat([recon_slots, slots_alpha], dim=-1)

            color_code = torch.ones_like(recon_combined).detach()
        
        loss = 0

        return recon_combined, recon_slots, color_code, gs_out, gs_slot, loss