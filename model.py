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

class Gs_Embedding(nn.Module):    
    def __init__(self, pos_dim, feature_dim):
        super().__init__()
        self.embedding = nn.Linear(pos_dim, feature_dim)
        # self.embedding = FourierEmbedding(pos_dim, 6, include_input=False)
        self._out_dim = feature_dim
        # self._out_dim = feature_dim + self.embedding.out_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, input, pos):
        pe = self.embedding(pos)
        # return torch.cat([input, pe], dim=-1)
        return input + pe

    
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
        self.encoder_pos = Gs_Embedding(3, hid_dim, gs_dim*5)

    def forward(self, x, pos):
        x = self.mlp(x)
        x = self.encoder_pos(x, pos)
        return x

class Gs_Object_Decoder(nn.Module):
    def __init__(self, gs_dim,slot_dim, hid_dim):
        super(Gs_Object_Decoder, self).__init__()
        self.pos_head = Gs_Pos_Decoder(slot_dim, hid_dim)
        self.color_head = Gs_Color_Decoder(slot_dim, hid_dim)
        self.mask_head = Gs_Mask_Decoder(slot_dim, hid_dim)
        self.share_embbeding = Gs_Embedding(gs_dim, hid_dim, slot_dim)

    def forward(self, slots, gs) -> torch.Tensor:
        x = self.share_embbeding(slots, gs)  # (B, N_S, G, gs_dim)
        pos = self.pos_head(x)    # offset in unit coords        
        color = self.color_head(x)  # bounded shift        
        mask = self.mask_head(x)
        return pos, color, mask # (B, N_S, G, 3), (B, N_S, G, 3), (B, N_S, G, 1)


class Gs_Decoder(nn.Module):
    def __init__(self, slot_dim, hid_dim):
        super(Gs_Decoder, self).__init__()
        self.gs_head = nn.Sequential(
            nn.Linear(slot_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 15),
            # nn.Tanh()   # keep color shifts bounded
        )

    def forward(self, slots) -> torch.Tensor:
        # color = slots[..., 11:14]      # slot color
        gs = self.gs_head(slots)
        gs, mask = torch.split(gs, [14, 1], dim=-1)
        mask = F.softmax(mask, dim=1)
        # color = color + color_shift
        return gs, mask # (B, N_S, G, 14), (B, N_S, G, 1)
    

class Gs_Color_Decoder(nn.Module):
    def __init__(self, slot_dim, hid_dim):
        super(Gs_Color_Decoder, self).__init__()
        self.color_head = nn.Sequential(
            nn.Linear(slot_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, 3),
            # nn.Tanh()   # keep color shifts bounded
        )

    def forward(self, slots) -> torch.Tensor:
        # color = slots[..., 11:14]      # slot color
        color = self.color_head(slots)  # bounded shift
        # color = color + color_shift
        return color # (B, N_S, G, 3)
    
class Gs_Pos_Decoder(nn.Module):
    def __init__(self, slot_dim, hid_dim):
        super(Gs_Pos_Decoder, self).__init__()
        self.pos_head = nn.Sequential(
            nn.Linear(slot_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, 3),
            nn.Tanh()
        )
        self.scale_head = nn.Sequential(
            nn.Linear(slot_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, 3),
            nn.Softplus()
        )

    def forward(self, slots) -> torch.Tensor:
        means   = slots[..., :3]
        # scale = slots[..., 7:10]
        std = self.pos_head(slots)
        scale = self.scale_head(slots)
        pos = means + std * scale
        return pos, scale # (B, N_S, G, 3)
    
class Gs_Mask_Decoder(nn.Module):
    def __init__(self, slot_dim, hid_dim):
        super(Gs_Mask_Decoder, self).__init__()
        self.mask_head = nn.Sequential(
            nn.Linear(slot_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1),
        )

    def forward(self, slots) -> torch.Tensor:
        mask = self.mask_head(slots)
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

        self.feature_mask = torch.tensor([data_cfg.use_xyz,
                                        data_cfg.use_rots,
                                        data_cfg.use_scale,
                                        data_cfg.use_opacity,
                                        data_cfg.use_color,
                                        data_cfg.use_motion], dtype=torch.bool)
        # feature_len = torch.tensor([3, 4, 3, 1, 3, 3], dtype=torch.int32)
        gs_dim = 14
        slot_dim = 64

        # self.slot_norm = nn.LayerNorm(gs_dim)


        self.encoder = nn.Linear(gs_dim, slot_dim)
        self.encode_embedding = Gs_Embedding(3, slot_dim)

        self.decode_embedding = Gs_Embedding(3, slot_dim)
        self.decoder = Gs_Decoder(self.share_embedding.out_dim, self.hid_dim)
        # self.pos_decode = Gs_Pos_Decoder(slot_dim, self.hid_dim)
        # self.color_decoder = Gs_Color_Decoder(self.share_embedding.out_dim, self.hid_dim)
        # self.mask_decoder = Gs_Mask_Decoder(self.share_embedding.out_dim, self.hid_dim)
        
        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            input_dim=gs_dim,
            slot_dim=slot_dim,
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
        pos = pos.repeat(1,self.num_slots,1,1) # [B, N_S, G, 3]
        # feature = gs.unsqueeze(1)
        x = self.encoder(gs)
        x = self.encode_embedding(x, pos) # [B, N_S, G, D]
        # x = gs

        # Slot Attention module.
        slots = self.slot_attention(x, mask) # [B, N_S, D]

        # Retrieve slots color
        # colors = slots[:,:,11:14]
        # colors = colors.unsqueeze(-2)
        # colors = colors.repeat(1,1,G,1) # [B, N_S, G, 3]
        # color_code = colors.repeat(1,1,self.num_slots,1) # [B, N_S, N_S, 3]
        # color_code = (color_code - color_code.min()) / (color_code.max() - color_code.min() + 1e-8) # [B, N_S, 3]

        # Broadcast slots to all pos
        slots = slots.unsqueeze(-2) # [B, N_S, D]
        slots = slots.repeat(1,1,G,1) # [B, N_S, G, D]

        # Copy gs to match slots count
        # gs_slot = gs.unsqueeze(1).repeat(1,self.num_slots,1,1) # [B, N_S, G, D]

        # Retrieve original textures to colors
        # gray_weights = torch.tensor([0.299, 0.587, 0.114], device=gs_slot.device)
        # textures = (gs_slot[:,:,:,11:14] * gray_weights).sum(dim=-1, keepdim=True)  # [B, N_S, G, 1]

        # Shared embedding
        slots = self.decode_embedding(slots, pos) # [B, N_S, G, D]

        # MLP detection head for color and mask
        gs_slot, gs_mask = self.decoder(slots)
        # # MLP detection head for pos, color and mask
        # colors = self.color_decoder(slots) # [B, N_S, G, 3]
        # colors = colors * textures # [B, N_S, G, 3]

        # MLP detection head for object shape
        # pos, scale = self.pos_decode(slots) # [B, N_S, G, 1]

        # MLP detection head for mask
        # gs_mask = self.mask_decoder(slots) # [B, N_S, G, 1]
        gs = torch.sum(gs_slot * gs_mask, dim=1)

        gs_slot = torch.cat([gs_slot, gs_mask, gs_mask, gs_mask ], dim=-1) # [B, N_S, G, D+3]
        # gs_slot = torch.cat([gs_slot[:,:,:,:10],color_mask,colors], dim=-1) # [B, N_S, G, D]

        # Weighted Sum pos and colors
        # pos = torch.sum(pos * gs_mask, dim=1)
        # colors = torch.sum(colors * gs_mask, dim=1)

        # Recconstruct gs
        # gs = torch.cat([gs[:,:,:11], colors], dim=-1)
        # gs = torch.cat([gs[:,:,:10], torch.ones_like(gs[:,:,10:11])*0.5,colors], dim=-1)

        # 3D Gaussian renderer
        recon_combined = []
        for batch,ks,w2c in zip(gs,Ks,w2cs):
            means, quats, scales, opacities, colors = torch.split(batch, [3,4,3,1,3], dim=-1)
            recon_combined.append(self.renderer.rasterize_gs(means, quats, scales, opacities, colors,ks,w2c))
        recon_combined = torch.stack(recon_combined,dim=0)[:,:,:,0:3]

        recon_slots = []
        slots_alpha = []
        if inference:
            for batch,ks,w2c in zip(gs_slot,Ks,w2cs):
                for slot in batch:
                    means, quats, scales, opacities, colors,alpha = torch.split(slot, [3,4,3,1,3,3], dim=-1)
                    recon_slots.append(self.renderer.rasterize_gs(means, quats, scales, opacities, colors,ks,w2c))
                    slots_alpha.append(self.renderer.rasterize_gs(means, quats, scales, opacities, alpha,ks,w2c))
            recon_slots = torch.stack(recon_slots,dim=0)[:,:,:,0:3]
            slots_alpha = torch.stack(slots_alpha,dim=0)[:,:,:,0:1]
            recon_slots = torch.cat([recon_slots, slots_alpha], dim=-1)
        
        loss = 0
        # loss = color_shift.mean() ** 2

        # gs_mask = gs_mask.unsqueeze(-1)
        # entropy = -(gs_mask * torch.log(gs_mask + 1e-8)).sum(dim=1)
        # loss_entropy = torch.mean(entropy)
        # loss = loss_entropy * 0.1

        color_code = torch.ones_like(recon_combined)

        return recon_combined, recon_slots, color_code, gs, gs_slot, loss