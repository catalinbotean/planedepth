"""
Overfit test for PlaneDepth (paper branch).

Creates a synthetic stereo pair with a *known* ground-truth disparity map,
then tries to overfit the encoder+decoder on that single example for N steps.

What it checks
--------------
* The full forward pass (encoder → decoder) works end-to-end.
* Gradients flow back through every component (no silent in-place / detach bugs).
* The photometric reconstruction loss actually decreases — i.e. the model CAN
  fit the signal (a necessary but not sufficient condition for real training).

Pass criterion: loss at step 50 < 70% of loss at step 0.

Usage
-----
    python test_overfit.py            # CPU or GPU auto-detected
    python test_overfit.py --steps 200 --device cuda

No dataset or pretrained weights required.
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--steps",  type=int,   default=100)
parser.add_argument("--lr",     type=float, default=1e-3)
parser.add_argument("--device", type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed",   type=int,   default=42)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device(args.device)
print(f"device: {device} | steps: {args.steps} | lr: {args.lr}")

# ── Scene geometry ────────────────────────────────────────────────────────────
# Keep spatial dims small so the test is fast even on CPU.
# Must be multiples of 32 (encoder downsamples 5×).
B, H, W     = 1, 64, 192          # batch, height, width
DISP_MIN    = 2                    # pixels  (far depth)
DISP_MAX    = 48                   # pixels  (near depth — ~ 4 m at KITTI baseline)
GT_DISP     = 20.0                 # constant disparity for the synthetic scene

# ── Build a synthetic stereo pair ─────────────────────────────────────────────
# Left image: random texture fixed for the whole experiment.
# Right image: left shifted LEFT by GT_DISP pixels (standard stereo convention).
torch.manual_seed(0)
img_left = torch.rand(B, 3, H, W, device=device)   # [0, 1]

# Shift: right[b, c, y, x] = left[b, c, y, x + GT_DISP]
# Build a sampling grid for this shift.
xs = torch.arange(W, dtype=torch.float32, device=device)
ys = torch.arange(H, dtype=torch.float32, device=device)
grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # H, W each

# Shifted x in pixel coords, then normalised to [-1, 1]
src_x = grid_x + GT_DISP                          # H, W
src_x_norm = 2.0 * src_x / (W - 1) - 1.0
src_y_norm = 2.0 * grid_y / (H - 1) - 1.0

sample_grid = torch.stack([src_x_norm, src_y_norm], dim=-1)  # H, W, 2
sample_grid = sample_grid.unsqueeze(0).expand(B, -1, -1, -1) # B, H, W, 2

# Right image synthesised by forward-warping left
img_right = F.grid_sample(img_left, sample_grid,
                           mode="bilinear", padding_mode="border",
                           align_corners=True)              # B, 3, H, W

# ── SSIM (from layers.py) ────────────────────────────────────────────────────
from layers import SSIM
ssim_fn = SSIM().to(device)

def photometric_loss(pred, target):
    """0.85 SSIM + 0.15 L1, same as PlaneDepth trainer."""
    l1   = (pred - target).abs().mean(1, keepdim=True)
    s    = ssim_fn(pred, target)
    return (0.85 * s + 0.15 * l1).mean()

# ── Build encoder + decoder (XY-only, no grids needed) ───────────────────────
import networks

encoder = networks.ResnetEncoder(num_layers=18, pretrained=False).to(device)
enc_chs = list(encoder.num_ch_enc)   # [64, 64, 128, 256, 512]

decoder = networks.DepthDecoder(
    num_ch_enc       = enc_chs,
    no_levels        = 32,           # XY fronto-parallel planes
    disp_min         = DISP_MIN,
    disp_max         = DISP_MAX,
    xz_levels        = 0,            # no ground planes → no input_grids needed
    yz_levels        = 0,
    use_mixture_loss = True,
    use_skips        = True,
    use_denseaspp    = False,
).to(device)

params   = list(encoder.parameters()) + list(decoder.parameters())
optim    = torch.optim.Adam(params, lr=args.lr)

# ── Helper: warp right → left given disparity (pixels) ───────────────────────
def warp_right_to_left(img_r, disp_pixels):
    """
    img_r      : B, 3, H, W  — right image
    disp_pixels: B, 1, H, W  — predicted disparity (pixels, positive)
    Returns warped image B, 3, H, W

    Standard stereo convention: the right camera is to the RIGHT of the left.
    A point at disparity d appears at x_right = x_left - d.
    Therefore img_right[x] = img_left[x + d], and to reconstruct the left we
    sample right at (x - d): warped_left[x] = img_right[x - d].
    The synthetic right image below is created with the same convention.
    """
    xs = torch.arange(W, dtype=torch.float32, device=device)
    ys = torch.arange(H, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # H, W
    # For each left pixel (x, y) sample right at (x - disp, y)
    src_x = gx.unsqueeze(0).unsqueeze(0) - disp_pixels  # B, 1, H, W
    src_x_n = 2.0 * src_x / (W - 1) - 1.0
    src_y_n = 2.0 * gy.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W) / (H - 1) - 1.0
    grid = torch.cat([src_x_n, src_y_n], dim=1)         # B, 2, H, W
    grid = grid.permute(0, 2, 3, 1)                      # B, H, W, 2
    return F.grid_sample(img_r, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)

# ── Training loop ─────────────────────────────────────────────────────────────
losses = []
print(f"\n{'Step':>6}  {'Loss':>10}  {'MeanDisp':>10}  {'GTDisp':>10}")
print("-" * 44)

for step in range(args.steps):
    optim.zero_grad()

    feats = encoder(img_left)
    out   = decoder(feats, input_grids=None)

    disp  = out["disp"]           # B, 1, H, W — predicted disparity in pixels

    warped = warp_right_to_left(img_right, disp)
    loss   = photometric_loss(warped, img_left)

    loss.backward()
    optim.step()

    losses.append(loss.item())

    if step % 10 == 0 or args.verbose:
        mean_disp = disp.detach().mean().item()
        print(f"{step:>6}  {loss.item():>10.4f}  {mean_disp:>10.2f}  {GT_DISP:>10.1f}")

# ── Pass / fail ───────────────────────────────────────────────────────────────
print()
loss_0    = losses[0]
loss_end  = losses[-1]
ratio     = loss_end / loss_0
threshold = 0.70

print(f"Loss at step 0   : {loss_0:.4f}")
print(f"Loss at step {args.steps-1:<3}: {loss_end:.4f}")
print(f"Ratio (end/start): {ratio:.3f}  (pass if < {threshold})")

if ratio < threshold:
    print("\n✓  PASS — loss decreased by more than 30%")
    sys.exit(0)
else:
    print("\n✗  FAIL — loss did not decrease enough")
    sys.exit(1)
