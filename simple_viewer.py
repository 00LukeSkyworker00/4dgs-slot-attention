"""A simple example to render a (large-scale) Gaussian Splats

```bash
python examples/simple_viewer.py --scene_grid 13
```
"""

# import argparse
# import math
# import os
import time
from typing import Tuple

# import imageio
import nerfview
# import numpy as np
import torch
# import torch.nn.functional as F
# import tqdm
import viser

# from gsplat._helper import load_test_data
# from gsplat.distributed import cli
from gsplat.rendering import rasterization


def viewer(args):
    torch.manual_seed(42)
    device = torch.device("cuda")
    
    means, quats, scales, opacities, colors = torch.split(args.gs.to(device), [3,4,3,1,3], dim=-1)
    opacities = opacities.squeeze(-1)

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()
        bg_color = torch.full((1, 3), 1.0, device=device).float()

        render_colors, render_alphas, meta = rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            backgrounds=bg_color,
            # sh_degree=sh_degree,
            render_mode="RGB",
            # this is to speedup large-scale rendering by skipping far-away Gaussians.
            radius_clip=3,
        )
        render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
        return render_rgbs

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    try:
        time.sleep(100000)
    except KeyboardInterrupt:
        # this is important.
        print(f"Process interrupted.")
        server.stop()
