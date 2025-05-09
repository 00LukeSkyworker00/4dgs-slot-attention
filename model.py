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

    def forward(self, inputs, num_slots = None): 
        b, _, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', k, q) * self.scale
            attn = dots.softmax(dim=-1) + self.eps

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
        pe:torch.Tensor = self.embedding(pos)
        if self.use_fourier:
            return torch.cat([input, pe], dim=-1)
        else:
            return input + pe

class FourierEmbedding(nn.Module):
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
    
class Gs_Encoder(nn.Module):
    def __init__(self, gs_dim, slot_dim, pos_dim, col_dim):
        super(Gs_Encoder, self).__init__()
        self.pos_dim = pos_dim
        self.col_dim = col_dim
        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, gs_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gs_dim, pos_dim)
        )
        self.col_encoder = nn.Sequential(
            nn.Linear(col_dim, gs_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gs_dim, col_dim)
        )
        self.share_encoder = nn.Linear(gs_dim + pos_dim + col_dim, slot_dim)
        self.embedding = Gs_Embedding(24,slot_dim)


    def forward(self, x, pe) -> torch.Tensor:
        pos = self.pos_encoder(x[...,:self.pos_dim])
        col = self.col_encoder(x[...,-self.col_dim:])
        share = self.share_encoder(torch.cat([pos,x,col],dim=-1))
        out =  self.embedding(share,pe)
        return out # (B, N_S, G)
    

class Gs_Decoder(nn.Module):
    def __init__(self, slot_dim, hid_dim, frame_num):
        super(Gs_Decoder, self).__init__()

        # self.mlp_head = nn.Sequential(
        #     nn.Linear(slot_dim, hid_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hid_dim,14+1)
        # )

        self.embedding = Gs_Embedding(24,slot_dim)

        self.shared_mlp = nn.Sequential(
            nn.Linear(slot_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(inplace=True),
        )

        self.pos_head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim*2, frame_num*3),
        )
        self.col_head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, 3),
        )
        self.mask_head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, 1),
        )

        # self.pcm_head = TriPlane_Decoder(slot_dim, hid_dim, 1+3+1)

        # self.pos_head = Pos_Decoder(slot_dim, hid_dim)
        # self.col_head = Col_Decoder(slot_dim, hid_dim)
        # self.mask_head = Gs_Mask_Decoder(slot_dim, hid_dim)

    def forward(self, x, pe) -> torch.Tensor:

        # # Shared mlp
        # gs = self.mlp_head(x)
        # gs, mask = torch.split(gs, [14,1], dim=-1)

        # Shared mlp for pos col mask
        x_pe = self.embedding(x, pe)
        out = self.shared_mlp(x_pe)
        pos = self.pos_head(out)
        color = self.col_head(out)
        mask = self.mask_head(out)

        # # Triplane decoder
        # x_out, y_out, z_out = self.pcm_head(x, pos)
        # pos = torch.cat([x_out[...,0:1],y_out[...,0:1],z_out[...,0:1]], dim=-1)
        # out = x_out[...,1:5] + y_out[...,1:5] + z_out[...,1:5]
        # color, mask = torch.split(out, [3,1], dim=-1)

        # x_embed = self.embedding(x, pos)
        # pos = self.pos_head(x, pos)
        # color = self.col_head(x_embed)
        # mask = self.mask_head(x_embed)

        return pos, color, mask # (B, N_S, G, 3), (B, N_S, G, 3), (B, N_S, G, 1)
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
        return mask # (B, N_S, G, 1)

class SlotAttentionAutoEncoder(nn.Module):
    # def __init__(self, resolution, num_slots, num_iters, hid_dim):
    def __init__(self, data_cfg, cnn_cfg, attn_cfg, gs_dim):
        super().__init__()
        self.gs_dim = gs_dim

        self.hid_dim = cnn_cfg.hid_dim
        self.gs_pos_embed = cnn_cfg.gs_pos_embed
        self.encode_gs = cnn_cfg.encode_gs
        self.resolution = tuple(data_cfg.resolution)
        self.num_slots = attn_cfg.num_slots
        self.num_iters =  attn_cfg.num_iters

        self.frame_num = int(self.gs_dim / 7 - 1)
        slot_dim = max(192, self.gs_dim)

        self.encoder = Gs_Encoder(self.gs_dim,slot_dim, 3*self.frame_num, 3)
        
        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            slot_dim=slot_dim,
            iters=self.num_iters,
            eps = 1e-8, 
            hidden_dim = 192)
        
        self.decoder = Gs_Decoder(slot_dim, 64, self.frame_num)

    def forward(self, gs:torch.Tensor, pe:torch.Tensor, pad_mask=None):
        """
        gs: [B, G, D]
        pos: [B, G, 3]
        mask: [B, G]
        """
        _,G,_ = gs.shape
        x = gs
        x = self.encoder(x, pe)
        x = self.apply_mask(x, pad_mask, 0)

        # Slot Attention module.
        slots = self.slot_attention(x) # [B, N_S, D]

        # Broadcast pos to all slots
        pe = pe.unsqueeze(1).repeat(1,self.num_slots,1,1) # [B, N_S, G, 3]

        # Broadcast slots to all points
        slots = slots.unsqueeze(-2).repeat(1,1,G,1) # [B, N_S, G, D]

        # MLP detection head for color and mask
        # gs_slot, gs_mask = self.decoder(slots)
        pos, color, gs_mask = self.decoder(slots, pe) # [B, N_S, G, D]

        # Account for padding before softmax the gs mask
        if pad_mask is not None: pad_mask = pad_mask.unsqueeze(1)        
        gs_mask = self.apply_mask(gs_mask, pad_mask, -1e9)
        gs_mask = F.softmax(gs_mask, dim=1)

        # Duplicate gs to match slots count and inject prediction
        gs_slot = gs.unsqueeze(1).repeat(1,self.num_slots,1,1) # [B, N_S, G, D]
        gs_slot = torch.cat([pos, gs_slot[...,self.frame_num*3:-3], color], dim=-1) # [B, N_S, G, D]

        # Recconstruct gs
        gs_out = torch.sum(gs_slot * gs_mask, dim=1)

        # Intermediate loss
        loss = 0

        return gs_out, gs_slot, gs_mask, loss
    
    def apply_mask(self, input: torch.Tensor, mask: torch.Tensor, pad_value: int):
        """
        input: [B,G,D]
        mask: [B,G,1]
        """
        if mask is None:
            return input
        
        if mask.shape[-1] != 1:
            mask = mask.unsqueeze(-1)

        out = input.masked_fill(mask==0, pad_value)
        return out
    
class RasterizedSlotAttentionAutoEncoder(nn.Module):
    def __init__(self, data_cfg, cnn_cfg, attn_cfg):
        self.model = SlotAttentionAutoEncoder(data_cfg, cnn_cfg, attn_cfg)        
        self.renderer = Renderer(tuple(data_cfg.resolution), requires_grad=True)

    def forward(self, gs:torch.Tensor, pe:torch.Tensor, Ks: torch.Tensor, w2cs: torch.Tensor, pad_mask=None):
        gs_out, gs_slot, gs_mask, loss = self.model(gs,pe,pad_mask)
        recon_combined, recon_slots, recon_mask = render_batch(self.renderer, gs_out, gs_slot, gs_mask, Ks, w2cs)

        return recon_combined, recon_slots, recon_mask, loss