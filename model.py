from torch import nn
import torch
import torch.nn.functional as F
from renderer import *
from transforms import rot_to_mat4, to_homogeneous

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
            self.embedding = FourierEmbedding(pos_dim, 3, include_input=False)
            self._out_dim = feature_dim + self.embedding.out_dim
        else:
            self.embedding = nn.Linear(pos_dim, feature_dim)
            self._out_dim = feature_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, input, pos):
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
        # self.pos_encoder = nn.Sequential(
        #     nn.Linear(pos_dim, gs_dim),
        #     nn.ReLU(inplace=True)
        # )
        # self.col_encoder = nn.Sequential(
        #     nn.Linear(col_dim, gs_dim),
        #     nn.ReLU(inplace=True)
        # )
        # self.share_encoder = nn.Linear(gs_dim + gs_dim + gs_dim, slot_dim)
        self.encoder = nn.Sequential(
            nn.Linear(gs_dim, slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim, slot_dim),
        )
        self.embedding = Gs_Embedding(3,slot_dim)


    def forward(self, x, pe) -> torch.Tensor:
        # pos = self.pos_encoder(x[...,:self.pos_dim])
        # col = self.col_encoder(x[...,-self.col_dim:])
        # share = self.share_encoder(torch.cat([pos,x,col],dim=-1))

        share = self.encoder(x)
        out =  self.embedding(share,pe)
        return out # (B, N_S, G, D)
    
class Object_Decoder(nn.Module):
    def __init__(self, gs_dim, slot_dim, hid_dim, frame_num):
        super(Object_Decoder, self).__init__()

        self.frame_num = frame_num

        self.shared_mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(inplace=True),
        )

        self.track_head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim, frame_num*9),
        )
        self.col_head = nn.Sequential(
            nn.Linear(slot_dim, gs_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gs_dim, 3),
        )

    def forward(self, x) -> torch.Tensor:
        '''
        x: [B,N_S,D]
        pe: [B,N_S,3]
        '''
        B,N_S,_ = x.shape
        # Shared mlp for pos col mask
        out = self.shared_mlp(x)
        track = self.track_head(out)
        track = track.view(B, N_S, self.frame_num, 9).unsqueeze(-2) # [B, N_S, FRAME, 1, 9]
        color = self.col_head(out).unsqueeze(-2) # [B, N_S, 1, 3]

        return track, color, out # [B, N_S, FRAME, 1, 9], [B, N_S, 1, 3], [B, N_S, gs_dim]

class Gs_Decoder(nn.Module):
    def __init__(self, gs_dim, slot_dim, hid_dim, frame_num):
        super(Gs_Decoder, self).__init__()

        self.embedding = Gs_Embedding(3,slot_dim)
        self.frame_num = frame_num

        # self.shared_mlp = nn.Sequential(
        #     nn.Linear(slot_dim, gs_dim),
        #     nn.ReLU(inplace=True),
        # )

        self.pos_head = nn.Sequential(
            nn.Linear(slot_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, 3),
            # nn.Linear(hid_dim, 12),
        )
        self.col_head = nn.Sequential(
            nn.Linear(slot_dim, gs_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gs_dim, 3),
        )
        self.mask_head = nn.Sequential(
            nn.Linear(slot_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, 1),
        )

    def forward(self, x, pe, obj_track, obj_color) -> torch.Tensor:
        '''
        x: [B,N_S,G,D]
        pe: [B,N_S,G,3]
        obj_track: [B, N_S, FRAME, 1, 3, 1]
        obj_color: [B,N_S,1,3]
        '''
        B,N_S,G,_ = x.shape
        
        # Shared mlp for pos col mask
        out = self.embedding(x, pe)
        # out = self.shared_mlp(x_pe)
        mask = self.mask_head(out)

        local_pos = self.pos_head(out)
        local_pos = local_pos.unsqueeze(-3) # [B, N_S, 1, G, 3]
        local_pos = to_homogeneous(local_pos) # [B, N_S, 1, G, 4, 1]
        obj_mean, obj_rot_6d = torch.split(obj_track,(3,6),dim=-1) # [B, N_S, FRAME, 1, 3], [B, N_S, FRAME, 1, 6]
        obj_rotmat = rot_to_mat4(obj_rot_6d, obj_mean) # [B, N_S, FRAME, 1, 4, 4]
        pos = torch.matmul(obj_rotmat, local_pos).squeeze(-1) # [B, N_S, FRAME, G, 4]
        pos = pos[...,:3].permute(0, 1, 3, 2, 4).reshape(B, N_S, G, self.frame_num * 3) #[B, N_S, G, FRAME*3]

        # local_transfm = self.transform_head(out)
        # local_pos, local_rot = torch.split(local_transfm,(3,9),dim=-1)
        # local_pos = local_pos.unsqueeze(-3) # [B, N_S, 1, G, 3]
        # local_rot = local_rot.view(B, N_S, G, 3, 3).unsqueeze(-4) # [B, N_S, 1, G, 3, 3]

        # rot_track = local_rot @ obj_track # [B, N_S, FRAME, G, 3, 1]
        # pos = rot_track.squeeze(-1) + local_pos # [B, N_S, FRAME, G, 3]
        # pos = pos.permute(0, 1, 3, 2, 4).reshape(B, N_S, G, self.frame_num * 3) #[B, N_S, G, FRAME*3]

        local_color = self.col_head(out)
        color = obj_color + local_color

        return pos, color, mask # (B, N_S, G, 3*FRAME), (B, N_S, G, 3), (B, N_S, G, 1)

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
        slot_dim = max(256, self.gs_dim)

        self.encoder = Gs_Encoder(self.gs_dim,slot_dim, 3*self.frame_num, 3)
        
        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            slot_dim=slot_dim,
            iters=self.num_iters,
            eps = 1e-8, 
            hidden_dim = 256)
        
        self.obj_decoder = Object_Decoder(self.gs_dim,slot_dim, 34, self.frame_num)
        self.gs_decoder = Gs_Decoder(self.gs_dim,slot_dim, 34, self.frame_num)

    def forward(self, gs:torch.Tensor, pe:torch.Tensor, pad_mask=None, isInference=False):
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

        # Object decoder per slot
        obj_track, obj_color, slots = self.obj_decoder(slots) # [B, N_S, D]

        # Broadcast pos to all slots
        pe = pe.unsqueeze(1).repeat(1,self.num_slots,1,1) # [B, N_S, G, 3]

        # Broadcast slots to all points
        slots = slots.unsqueeze(-2).repeat(1,1,G,1) # [B, N_S, G, D]

        # MLP detection head for color and mask
        # gs_slot, gs_mask = self.decoder(slots)
        pos, color, gs_mask = self.gs_decoder(slots, pe, obj_track, obj_color) # [B, N_S, G, D]

        # Account for padding before softmax the gs mask
        if pad_mask is not None: pad_mask = pad_mask.unsqueeze(1)        
        gs_mask = self.apply_mask(gs_mask, pad_mask, -1e9)
        gs_mask = F.softmax(gs_mask, dim=1)

        if isInference:
            # Duplicate gs to match slots count and inject prediction
            gs_slot = gs.unsqueeze(1).repeat(1,self.num_slots,1,1) # [B, N_S, G, D]
            gs_slot = torch.cat([pos, gs_slot[...,self.frame_num*3:-3], color], dim=-1) # [B, N_S, G, D]
        else:
            gs_slot = torch.cat([pos, color], dim=-1) # [B, N_S, G, D]

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
        recon_combined, recon_slots = self.renderer.make_vid(gs_out, gs_slot, gs_mask, Ks, w2cs)

        return recon_combined, recon_slots, loss