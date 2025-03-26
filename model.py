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

        # self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        # self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        # self.to_q = nn.Linear(dim, dim)
        # self.to_k = nn.Linear(dim, dim)
        # self.to_v = nn.Linear(dim, dim)

        # self.gru = nn.GRUCell(dim, dim)

        # hidden_dim = max(dim, hidden_dim)

        # self.fc1 = nn.Linear(dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, dim)

        # self.norm_input  = nn.LayerNorm(dim)
        # self.norm_slots  = nn.LayerNorm(dim, eps=1e-6)
        # self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, slots, num_slots = None):
        b, n, d = inputs.shape
        # n_s = num_slots if num_slots is not None else self.num_slots
        
        # mu = self.slots_mu.expand(b, n_s, -1)
        # sigma = F.softplus(self.slots_sigma).expand(b, n_s, -1)
        # # sigma = self.slots_sigma.expand(b, n_s, -1)
        # slots = torch.normal(mu, sigma)

        # inputs = self.norm_input(inputs)
        # k, v = self.to_k(inputs), self.to_v(inputs)
        k, v = inputs, inputs


        for _ in range(self.iters):
            # slots_prev = slots
            # print(i,slots[0][0])
            q = slots

            # slots = self.norm_slots(slots)
            # q = self.to_q(slots)
            # print(i,slots[0][0])

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            slots = torch.einsum('bjd,bij->bid', v, attn)
            # updates = torch.einsum('bjd,bij->bid', v, attn)

            # slots = self.gru(
            #     updates.reshape(-1, d),
            #     slots_prev.reshape(-1, d)
            # )

            # slots = slots.reshape(b, -1, d)
            # slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

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
    def __init__(self, hid_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(26, hid_dim, 5, padding = 2)
        self.conv2 = nn.Conv1d(hid_dim, hid_dim, 5, padding = 2)
        self.conv3 = nn.Conv1d(hid_dim, hid_dim, 5, padding = 2)
        self.conv4 = nn.Conv1d(hid_dim, hid_dim, 5, padding = 2)

    def forward(self, x):
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
    def __init__(self, resolution, num_slots, num_iterations, hid_dim):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim)
        self.encoder_cnn_gs = Gs_Encoder(self.hid_dim)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=hid_dim,
            iters = self.num_iterations,
            eps = 1e-8, 
            hidden_dim = 128)
        
        self.slot_broadcast = Gs_Slot_Broadcast(num_slots, hid_dim, 8)

    def forward(self, gs, slots, img):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        # x = self.encoder_cnn_gs(gs)  # CNN Backbone.
        # x = nn.LayerNorm(x.shape[1:]).to(img.device)(x)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, num_gaussians, input_size].

        # Inject encoded 4DGS.
        x = gs
        # `x` has shape: [batch_size, num_gaussians, slot_size].

        # Slot Attention module.
        # slots = self.slot_attention(x, slots)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        """Broadcast slot features to a 2D grid and collapse slot dimension."""
        # slots = self.slot_broadcast(slots)

        # slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        # slots = slots.repeat((1, 8, 8, 1))
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].


        x = self.decoder_cnn(slots)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(gs.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        # recon_combined = recon_combined.permute(0,3,1,2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].
        

        return recon_combined, recons, masks, slots
