import torch
from scipy.spatial.transform import Rotation as R

# Assume inputs are torch tensors
# means: (N, 3), quats: (N, 4), scales: (N, 3), opacity: (N,), color: (N, 3)
# K: (3, 3), w2c: (4, 4)

def transform_points(means, w2c):
    # Add a 1 to make it (N, 4)
    means_h = torch.cat([means, torch.ones_like(means[:, :1])], dim=1)  # (N, 4)
    means_cam = (w2c @ means_h.T).T[:, :3]
    return means_cam

def project(means_cam, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = means_cam[:, 0] / means_cam[:, 2]
    y = means_cam[:, 1] / means_cam[:, 2]

    u = fx * x + cx
    v = fy * y + cy
    return torch.stack([u, v, means_cam[:, 2]], dim=-1)  # (N, 3)

def quaternion_to_rotation_matrix(q):
    # q: (N, 4)
    r = R.from_quat(q.cpu().numpy())  # (x, y, z, w)
    return torch.tensor(r.as_matrix(), dtype=torch.float32, device=q.device)  # (N, 3, 3)

def compute_covariances(means_cam, scales, rotation_matrices, K):
    N = means_cam.shape[0]
    covariances = []
    for i in range(N):
        R_i = rotation_matrices[i]  # (3, 3)
        S_i = torch.diag(scales[i]**2)
        Sigma = R_i @ S_i @ R_i.T  # (3, 3)

        # Project into 2D
        J = torch.tensor([[K[0, 0] / means_cam[i, 2], 0, -K[0, 0] * means_cam[i, 0] / (means_cam[i, 2]**2)],
                          [0, K[1, 1] / means_cam[i, 2], -K[1, 1] * means_cam[i, 1] / (means_cam[i, 2]**2)]],
                          dtype=torch.float32, device=Sigma.device)
        cov2d = J @ Sigma @ J.T
        covariances.append(cov2d)
    return covariances  # List of (2,2)

def rasterize(points_2d, covariances, opacities, colors, H, W, radius=5):
    img = torch.zeros((H, W, 3), dtype=torch.float32, device=colors.device)
    alpha = torch.zeros((H, W), dtype=torch.float32, device=colors.device)

    for i in range(points_2d.shape[0]):
        u, v, depth = points_2d[i]
        if depth < 0: continue
        if not (0 <= u < W and 0 <= v < H): continue

        cov = covariances[i] + 1e-5 * torch.eye(2).to(colors.device)  # add epsilon for stability
        inv_cov = torch.inverse(cov)

        # Define local window to splat
        xmin = max(int(u - radius), 0)
        xmax = min(int(u + radius + 1), W)
        ymin = max(int(v - radius), 0)
        ymax = min(int(v + radius + 1), H)

        uv = torch.tensor([u, v], dtype=torch.float32, device=colors.device)

        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                p = torch.tensor([x, y], dtype=torch.float32, device=colors.device)
                diff = p - uv
                weight = torch.exp(-0.5 * (diff @ inv_cov @ diff.T))
                img[y, x] += weight * colors[i] * opacities[i]
                alpha[y, x] += weight * opacities[i]

    rgba = torch.cat([img / (alpha.unsqueeze(-1) + 1e-6), alpha.unsqueeze(-1)], dim=-1)
    return rgba.clamp(0, 1)

def render(means, quats, scales, opacity, color, K, w2c, H=128, W=128):
    means_cam = transform_points(means, w2c)
    points_2d = project(means_cam, K)
    rot_mats = quaternion_to_rotation_matrix(quats)
    covariances = compute_covariances(means_cam, scales, rot_mats, K)
    img = rasterize(points_2d, covariances, opacity, color, H, W)
    return img
