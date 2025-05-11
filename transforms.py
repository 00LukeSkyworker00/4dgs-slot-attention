import torch
import torch.nn.functional as F


def rot_to_mat4(
    R: torch.Tensor, t: torch.Tensor, s: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Args:
        R (torch.Tensor): (..., 3, 3) or (..., 4).
        t (torch.Tensor): (..., 3).
        s (torch.Tensor): (...,).

    Returns:
        torch.Tensor: (..., 4, 4)
    """
    if R.shape[-1] == 6:
        R = cont_6d_to_rmat(R)

    mat34 = torch.cat([R, t[..., None]], dim=-1)
    if s is None:
        bottom = (
            mat34.new_tensor([[0.0, 0.0, 0.0, 1.0]])
            .reshape((1,) * (mat34.dim() - 2) + (1, 4))
            .expand(mat34.shape[:-2] + (1, 4))
        )
    else:
        bottom = F.pad(1.0 / s[..., None, None], (3, 0), value=0.0)
    mat4 = torch.cat([mat34, bottom], dim=-2)
    return mat4

def to_homogeneous(x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(x[..., :1])  # same shape as [..., 1]
    return torch.cat([x, ones], dim=-1).unsqueeze(-1)  # [..., 4, 1]

def rmat_to_cont_6d(matrix):
    """
    :param matrix (*, 3, 3)
    :returns 6d vector (*, 6)
    """
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)


def cont_6d_to_rmat(cont_6d):
    """
    :param 6d vector (*, 6)
    :returns matrix (*, 3, 3)
    """
    x1 = cont_6d[..., 0:3]
    y1 = cont_6d[..., 3:6]

    x = F.normalize(x1, dim=-1)
    y = F.normalize(y1 - (y1 * x).sum(dim=-1, keepdim=True) * x, dim=-1)
    z = torch.linalg.cross(x, y, dim=-1)

    return torch.stack([x, y, z], dim=-1)

    
def compute_transforms(transls: torch.Tensor, rots: torch.Tensor, ts: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
    """
    :param transls (K, B, 3)
    :param rots (K, B, 6)
    :param ts (B)
    :param coefs (G, K)
    returns transforms (G, B, 3, 4)
    """
    transls = transls[:, ts]  # (K, B, 3)
    rots = rots[:, ts]  # (K, B, 6)
    transls = torch.einsum("pk,kni->pni", coefs, transls)  # (G, B, 3)
    rots = torch.einsum("pk,kni->pni", coefs, rots)  # (G, B, 6)
    rotmats = cont_6d_to_rmat(rots)  # (G, B, 3, 3)
    return torch.cat([rotmats, transls[..., None]], dim=-1)  # (G, B, 3, 4)