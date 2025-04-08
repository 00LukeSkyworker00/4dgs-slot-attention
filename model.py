import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid.to(inputs.device))
        return inputs + grid

class Gs_PositionEmbed(nn.Module):    
    def __init__(self, pos_dim, hidden_dim, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, input, pos):
        return input + self.encoder(pos)

class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_dim, 5, padding = 2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0,2,3,1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x
    
class Gs_Encoder(nn.Module):
    def __init__(self, gs_dim, hid_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(gs_dim, hid_dim, 5, padding = 2)
        self.conv2 = nn.Conv1d(hid_dim, hid_dim, 5, padding = 2)
        self.conv3 = nn.Conv1d(hid_dim, hid_dim, 5, padding = 2)
        self.conv4 = nn.Conv1d(hid_dim, hid_dim, 5, padding = 2)
        self.encoder_pos = Gs_PositionEmbed(3, hid_dim, hid_dim)

    def forward(self, x, pos):
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0,2,1)
        x = self.encoder_pos(x, pos)
        # x = torch.flatten(x, 1, 2)
        return x

class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1)
        self.decoder_initial_size = (8, 8)
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
#         x = F.pad(x, (4,4,4,4)) # no longer needed
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0,2,3,1)
        return x
    
class Gs_Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1)
        self.decoder_initial_size = (8, 8)
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0,2,3,1)
        return x
    
class Gs_Slot_Broadcast(nn.Module):
    def __init__(self, num_slots, slot_size, grid_size=8):
        super(Gs_Slot_Broadcast, self).__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.grid_size = grid_size
        
        # Learnable projection for each slot to predict soft assignment over the grid
        self.projection_mlp = nn.Sequential(
            nn.Linear(slot_size, 128),
            nn.ReLU(),
            nn.Linear(128, grid_size ** 2)  # Predicts soft assignments for each grid point
        )
        
        # Optional feature transformation to adjust slot features before broadcasting
        self.feature_transform = nn.Linear(slot_size, slot_size)

    def forward(self, slots):
        """
        slots: [B, N, D]  ->  Returns: [B, N, 8, 8, D]
        """
        batch_size = slots.size(0)
        
        # Transform the slot features before broadcasting
        transformed_slots = self.feature_transform(slots)  # Shape: [B, N, D]
        
        # Predict the soft assignment for each slot across the grid
        soft_assignments = self.projection_mlp(transformed_slots)  # Shape: [B, N, 8*8]
        
        # Reshape to [B, N, 8, 8] for the grid assignments
        soft_assignments = soft_assignments.view(batch_size, self.num_slots, self.grid_size, self.grid_size)
        
        # Apply softmax normalization along the spatial dimensions
        soft_assignments = F.softmax(soft_assignments, dim=-1)  # Normalize across grid positions (8*8)
        
        # Broadcast the slot features across the grid
        grid_features = torch.einsum("bnwh, bnd -> bnwhd", soft_assignments, transformed_slots)
        
        # Flatten B and N into a single dimension
        grid_features = grid_features.view(batch_size * self.num_slots, self.grid_size, self.grid_size, self.slot_size)

        return grid_features  # [B, N, 8, 8, D]

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    # def __init__(self, resolution, num_slots, num_iters, hid_dim):
    def __init__(self, data_cfg, cnn_cfg, attn_cfg):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iters: Number of iterations in Slot Attention.
        """
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
        gs_dim = feature_len[self.feature_mask].sum().item()

        if self.gs_pos_embed:
            if self.feature_mask[0].item() and gs_dim > 3:
                gs_dim -= 3
            else:
                self.gs_pos_embed = False

        if not self.encode_gs:
            self.hid_dim = gs_dim

        self.encoder_cnn_gs = Gs_Encoder(gs_dim, self.hid_dim)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

        self.fc1 = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc2 = nn.Linear(self.hid_dim, self.hid_dim)
        
        self.encoder_pos = Gs_PositionEmbed(3, self.hid_dim, gs_dim)
        
        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=self.hid_dim,
            iters=self.num_iters,
            eps = 1e-8, 
            hidden_dim = 128)
        
        self.slot_broadcast = Gs_Slot_Broadcast(self.num_slots, self.hid_dim, 8)

    def forward(self, gs, pos, mask=None):
        # gs: [B, G, D]
        # pos_embed: [B, G, 3, 1]
        # mask: [B, G]

        if self.gs_pos_embed:
            gs = gs[:,:,3:]

        if self.encode_gs:
            # Convolutional encoder with position embedding.
            x = self.encoder_cnn_gs(gs, pos)  # CNN Backbone.
            x = nn.LayerNorm(x.shape[1:]).to(gs.device)(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)  # Feedforward network on set.
            # [B, G, D]
        else:
            # Inject raw 4DGS.
            if self.gs_pos_embed:
                x = self.encoder_pos(x)
            else:
                x = gs
            # [B, G, D]
            
        # Slot Attention module.
        slots = self.slot_attention(x, mask) # [B, N_S, D]

        """Broadcast slot features to a 2D grid and collapse slot dimension."""
        slots = self.slot_broadcast(slots) # [B*N_S, W_init, H_init, D].

        # slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        # slots = slots.repeat((1, 8, 8, 1)) # [B*N_S, W_init, H_init, D].


        x = self.decoder_cnn(slots) # [B*N_S, W, H, CHANNEL+1]

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(gs.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        # recons: [B, N_S, W, H, CHANNEL]   masks: [B, N_S, W, H, 1]

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # [B, W, H, CHANNEL].        

        return recon_combined, recons, masks, slots
