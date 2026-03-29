"""
Comprehensive overfit test for PlaneDepth (paper branch).

Tests every branch feature — individually and in combinations — by overfitting
the encoder+decoder on a single synthetic stereo pair with a known ground-truth
disparity.  No dataset or pretrained weights required.

Pass criterion per scenario
---------------------------
  * Photometric loss at the end < 50 % of photometric loss at step 0
  * Final mean disparity within `disp_tol` px of GT_DISP=20

Scenarios
---------
  Individual:
    00  Baseline (mixture loss only)
    01  Cross-plane attention          (branch 1)
    02  Adaptive plane range           (branch 2)
    03  Pixelwise plane residual       (branch 3)
    04  Learned plane families         (branch 4)
    05  Multi-scale logit aggregation  (branch 10)
    06  Plane annealing                (branch 12)
    07  Semantic gate                  (paper)
    08  Entropy regularisation         (branch 7)
    09  Focal photometric loss         (branch 11)
    10  Confidence-weighted smoothness (branch 9)
    11  LR consistency loss            (branch 6)
    12  Surface normal smoothness      (branch 8)

  Combinations:
    13  Cross-attn + Learned families  (the fixed dimension bug)
    14  Multi-scale + Adaptive range
    15  Semantic gate + Entropy
    16  All decoder flags
    17  All loss augmentations
    18  Kitchen sink (everything)

Usage
-----
    python test_overfit.py
    python test_overfit.py --steps 80 --device cuda
    python test_overfit.py --only 00 01 13 18
"""

import argparse, sys, time
import numpy as np
import torch
import torch.nn.functional as F

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--steps",   type=int,   default=60)
parser.add_argument("--lr",      type=float, default=1e-3)
parser.add_argument("--device",  type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed",    type=int,   default=42)
parser.add_argument("--only",    nargs="*",  default=None,
                    help="run only these scenario ids, e.g. --only 00 13 18")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device(args.device)
print(f"device: {device} | steps/scenario: {args.steps} | lr: {args.lr}\n")

# ── Scene dimensions (multiples of 32 required by encoder) ───────────────────
B, H, W       = 1, 64, 192
DISP_MIN      = 2
DISP_MAX      = 48
GT_DISP       = 20.0
N_XY          = 32
N_XZ          = 4
N_SEM         = 19          # Cityscapes — triggers the hand-crafted prior

# ── Synthetic stereo pair ─────────────────────────────────────────────────────
# Sinusoidal pattern: every local patch has a clear horizontal gradient so
# SSIM responds strongly to the disparity value.
# IMPORTANT: use pixel coordinates [0..W-1] for the disparity shift, then
# normalise to [-1,1].  Using a linspace(0,1) grid here is wrong because
# the shift GT_DISP is in *pixels*, not in normalised units.
_xs_pix = torch.arange(W, dtype=torch.float32, device=device)
_ys_pix = torch.arange(H, dtype=torch.float32, device=device)
_gy_pix, _gx_pix = torch.meshgrid(_ys_pix, _xs_pix, indexing="ij")   # H, W (pixels)

# Image built on normalised coords [0,1] for the sinusoidal pattern
_gx_n = _gx_pix / (W - 1)   # ∈ [0, 1]

phase  = torch.tensor([0.0, 2*np.pi/3, 4*np.pi/3], device=device)
img_left = (0.5 + 0.5 * torch.sin(
    4*np.pi * _gx_n[None] + phase[:, None, None]
)).unsqueeze(0)                                              # B,3,H,W

# img_right[x] = img_left[x + GT_DISP]  (standard stereo, right cam to the right)
# Shift in *pixel space*, then normalise to [-1, 1] for grid_sample.
_sx = 2.0 * (_gx_pix + GT_DISP) / (W - 1) - 1.0           # pixel shift, then normalise
_sy = 2.0 * _gy_pix / (H - 1) - 1.0
img_right = F.grid_sample(
    img_left,
    torch.stack([_sx, _sy], -1).unsqueeze(0),
    mode="bilinear", padding_mode="border", align_corners=True)

# ── Positional grid (required by xz_levels > 0) ───────────────────────────────
def make_grid():
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy2, gx2 = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx2, gy2], 0).unsqueeze(0)           # 1,2,H,W

GRID = make_grid()

# ── Camera intrinsics (surface normal test) ───────────────────────────────────
fx = W * 0.58
K_np    = np.array([[fx, 0, W/2, 0], [0, fx, H/2, 0],
                    [0, 0, 1, 0],    [0, 0, 0, 1]], dtype=np.float32)
inv_K_t = torch.from_numpy(np.linalg.inv(K_np)).unsqueeze(0).to(device)

# ── SSIM ──────────────────────────────────────────────────────────────────────
from layers import SSIM, get_smooth_loss_disp_confidence
ssim_fn = SSIM().to(device)

import networks
from networks import DepthDecoder, SemanticPlaneGate

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_encoder():
    enc = networks.ResnetEncoder(num_layers=18, pretrained=False).to(device)
    return enc, list(enc.num_ch_enc)


def make_decoder(enc_chs, **kw):
    base = dict(
        num_ch_enc=enc_chs, no_levels=N_XY,
        disp_min=DISP_MIN, disp_max=DISP_MAX,
        xz_levels=0, xz_min=0.18, xz_max=0.37,
        yz_levels=0, use_mixture_loss=True,
        use_skips=True, use_denseaspp=False,
    )
    base.update(kw)
    return DepthDecoder(**base).to(device)


def warp_r2l(img_r, disp_pix):
    """warped_left[x] = img_right[x - disp]  (standard stereo)"""
    xs = torch.arange(W, dtype=torch.float32, device=device)
    ys = torch.arange(H, dtype=torch.float32, device=device)
    gy2, gx2 = torch.meshgrid(ys, xs, indexing="ij")
    sx  = 2.0*(gx2[None, None] - disp_pix)/(W-1) - 1.0
    sy  = (2.0*gy2/(H-1) - 1.0)[None, None].expand(B, 1, H, W)
    g   = torch.cat([sx, sy], 1).permute(0, 2, 3, 1)
    return F.grid_sample(img_r, g, mode="bilinear",
                         padding_mode="border", align_corners=True)


def photo(warped, target):
    l1 = (warped - target).abs().mean(1, True)
    s  = ssim_fn(warped, target)
    return (0.85*s + 0.15*l1).mean()


def surface_normal_loss(depth, img):
    """Edge-aware second-order depth smoothness (proxy for normal smoothness).

    True surface-normal loss requires back-projecting depth to 3-D points,
    which needs a camera matrix.  For the overfit test we use the second
    finite difference of depth as a lightweight proxy: it is zero for planar
    surfaces and large for curved or noisy depth maps, with the same
    edge-awareness as the full normal loss.
    """
    d  = depth                                                     # B,1,H,W
    tu = d[:, :, 1:-1, 2:] - d[:, :, 1:-1, :-2]                  # B,1,H-2,W-2
    tv = d[:, :, 2:, 1:-1] - d[:, :, :-2, 1:-1]                  # B,1,H-2,W-2
    ic = img[:, :, 1:-1, 1:-1]                                    # B,3,H-2,W-2
    wu = torch.exp(-(ic[:,:,:,1:]-ic[:,:,:,:-1]).abs().mean(1,True))   # B,1,H-2,W-3
    wv = torch.exp(-(ic[:,:,1:,:]-ic[:,:,:-1,:]).abs().mean(1,True))   # B,1,H-3,W-2
    # second differences penalise curvature (non-planar depth)
    loss_u = (tu[:,:,:,:-1] - tu[:,:,:,1:]).abs() * wu[:,:,:,:-1]      # B,1,H-2,W-3
    loss_v = (tv[:,:,:-1,:] - tv[:,:,1:,:]).abs() * wv[:,:,:-1,:]      # B,1,H-3,W-2
    return loss_u.mean() + loss_v.mean()


def lr_consistency_loss(disp_l, disp_r):
    """d_L(u) ≈ d_R(u + d_L(u))"""
    xs = torch.linspace(-1, 1, W, device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    gy2, gx2 = torch.meshgrid(ys, xs, indexing="ij")
    gx2 = gx2[None, None].expand(B, 1, H, W)
    gy2 = gy2[None, None].expand(B, 1, H, W)
    sx  = (gx2 + disp_l * 2.0/(W-1)).clamp(-1, 1)
    sg  = torch.stack([sx[:, 0], gy2[:, 0]], -1)
    d_r_w = F.grid_sample(disp_r, sg, padding_mode="zeros", align_corners=True)
    valid = (d_r_w > 0).float().detach()
    return (torch.abs(disp_l - d_r_w) * valid).sum() / (valid.sum() + 1e-8)


# ── Core overfit loop ─────────────────────────────────────────────────────────

def run_overfit(name, dec_kw, extra_fn=None, use_grid=False,
                sem_cfg=None, lr_consist=False,
                anneal_n=None, disp_tol=8.0):
    """
    dec_kw     : kwargs forwarded to make_decoder (xz_levels handled inside)
    extra_fn   : fn(out, img_l, img_r) -> scalar extra-loss tensor, or None
    use_grid   : pass GRID to decoder (required by xz_levels > 0)
    sem_cfg    : dict(n_xy, n_xz, n_yz, n_learned) → build SemanticPlaneGate
    lr_consist : also run decoder on img_right, add LR consistency term
    anneal_n   : if set, call dec.set_active_levels(anneal_n) before loop
    disp_tol   : max acceptable |mean_disp - GT_DISP| at the end
    """
    enc, enc_chs = make_encoder()
    dec          = make_decoder(enc_chs, **dec_kw)

    if anneal_n is not None:
        dec.set_active_levels(anneal_n)

    params = list(enc.parameters()) + list(dec.parameters())

    gate = None
    if sem_cfg is not None:
        gate = SemanticPlaneGate(num_classes=N_SEM, **sem_cfg).to(device)
        params += list(gate.parameters())

    opt  = torch.optim.Adam(params, lr=args.lr)
    grid = GRID if use_grid else None

    ph0 = ph_end = last_disp = None

    for step in range(args.steps):
        opt.zero_grad()

        feats_l = enc(img_left)

        sem_bias = None
        if gate is not None:
            fake_sem = torch.randn(B, N_SEM, H//4, W//4, device=device)
            sem_bias = gate(fake_sem)

        out_l  = dec(feats_l, input_grids=grid, sem_bias=sem_bias)
        warped = warp_r2l(img_right, out_l["disp"])
        ph     = photo(warped, img_left)
        total  = ph

        if extra_fn is not None:
            total = total + extra_fn(out_l, img_left, img_right)

        if lr_consist:
            feats_r = enc(img_right)
            out_r   = dec(feats_r, input_grids=grid)
            total   = total + 0.1 * lr_consistency_loss(
                          out_l["disp"], out_r["disp"])

        total.backward()
        opt.step()

        ph_val    = ph.item()
        disp_mean = out_l["disp"].detach().mean().item()
        if step == 0:
            ph0 = ph_val
        ph_end    = ph_val
        last_disp = disp_mean

        if args.verbose or step % 10 == 0:
            print(f"    step {step:>3}  ph={ph_val:.4f}  disp={disp_mean:.2f}")

    ratio    = ph_end / ph0
    disp_err = abs(last_disp - GT_DISP)
    passed   = (ratio < 0.50) and (disp_err < disp_tol)
    return passed, ratio, disp_err


# ── Extra-loss functions ──────────────────────────────────────────────────────

def extra_entropy(out, img_l, img_r):
    """Entropy regularisation: encourages peaky plane distributions."""
    pi = out["probability"]
    return 1e-3 * (-(pi * (pi + 1e-8).log()).sum(1).mean())

def extra_focal(out, img_l, img_r):
    """Focal weighting: scales per-pixel photometric loss by difficulty."""
    warped = warp_r2l(img_r, out["disp"])
    err    = (warped - img_l).abs().mean(1, True)
    with torch.no_grad():
        w = (err / (err.mean() + 1e-8)).pow(2.0).clamp(max=4.0)
    return 0.15 * (err * (w - 1)).mean()   # incremental over plain L1

def extra_conf_smooth(out, img_l, img_r):
    """Confidence-weighted disparity smoothness."""
    if "depth_confidence" not in out:
        return torch.tensor(0.0, device=device)
    return 1e-4 * get_smooth_loss_disp_confidence(
        out["disp"], img_l, out["depth_confidence"])

def extra_normal(out, img_l, img_r):
    """Surface normal (depth curvature) smoothness."""
    return 1e-4 * surface_normal_loss(out["depth"], img_l)

def extra_all(out, img_l, img_r):
    return (extra_entropy(out, img_l, img_r)
            + extra_conf_smooth(out, img_l, img_r)
            + extra_normal(out, img_l, img_r))


# ── Scenario table ────────────────────────────────────────────────────────────
# Each row: (id, name, dec_kw, extra_fn, use_grid, sem_cfg, lr_consist, anneal_n, disp_tol)

SCENARIOS = [
    ("00", "Baseline",
     {}, None, False, None, False, None, 8),

    ("01", "Cross-plane attention",
     {"xz_levels": N_XZ, "use_cross_plane_attn": True},
     None, True, None, False, None, 8),

    ("02", "Adaptive plane range",
     {"adaptive_plane_range": True},
     None, False, None, False, None, 8),

    ("03", "Pixelwise plane residual",
     {"pixelwise_plane_residual": True},
     None, False, None, False, None, 8),

    ("04", "Learned plane families",
     {"num_learned_families": 2, "learned_planes_per_family": 5},
     None, True, None, False, None, 8),   # learned families access input_grids → need grid

    ("05", "Multi-scale logit aggregation",
     {"use_multiscale_logits": True},
     None, False, None, False, None, 8),

    ("06", "Plane annealing (start low)",
     {},
     None, False, None, False, 4, 99),  # anneal_n=4: only 4 far-depth planes active,
                                        # so mean disp won't be 20 — only check ph_ratio

    ("07", "Semantic gate (paper branch)",
     {},
     None, False, {"n_xy": N_XY, "n_xz": 0, "n_yz": 0, "n_learned": 0},
     False, None, 8),

    ("08", "Entropy regularisation",
     {}, extra_entropy, False, None, False, None, 8),

    ("09", "Focal photometric loss",
     {}, extra_focal, False, None, False, None, 8),

    ("10", "Confidence-weighted smoothness",
     {}, extra_conf_smooth, False, None, False, None, 8),

    ("11", "LR consistency loss",
     {}, None, False, None, True, None, 8),

    ("12", "Surface normal smoothness",
     {}, extra_normal, False, None, False, None, 8),

    # ── Combinations ─────────────────────────────────────────────────────────

    ("13", "Cross-attn + Learned families  [fixed bug]",
     {"xz_levels": N_XZ, "use_cross_plane_attn": True,
      "num_learned_families": 2, "learned_planes_per_family": 5},
     None, True, None, False, None, 8),

    ("14", "Multi-scale + Adaptive range",
     {"use_multiscale_logits": True, "adaptive_plane_range": True},
     None, False, None, False, None, 8),

    ("15", "Semantic gate + Entropy",
     {}, extra_entropy,
     False, {"n_xy": N_XY, "n_xz": 0, "n_yz": 0, "n_learned": 0},
     False, None, 8),

    ("16", "All decoder flags",
     {"xz_levels": N_XZ, "use_cross_plane_attn": True,
      "adaptive_plane_range": True, "pixelwise_plane_residual": True,
      "num_learned_families": 2, "learned_planes_per_family": 5,
      "use_multiscale_logits": True},
     None, True, None, False, None, 10),

    ("17", "All loss augmentations",
     {}, extra_all, False, None, False, None, 10),

    ("18", "Kitchen sink (everything)",
     {"xz_levels": N_XZ, "use_cross_plane_attn": True,
      "adaptive_plane_range": True, "pixelwise_plane_residual": True,
      "num_learned_families": 2, "learned_planes_per_family": 5,
      "use_multiscale_logits": True},
     extra_all, True,
     {"n_xy": N_XY, "n_xz": N_XZ, "n_yz": 0, "n_learned": 10},
     True, 4, 12),
]

# ── Run all (or selected) scenarios ──────────────────────────────────────────
run_ids = set(args.only) if args.only else None
results = []

for row in SCENARIOS:
    sid, sname, dec_kw, extra_fn, use_grid, sem_cfg, lr_c, anneal_n, tol = row

    if run_ids and sid not in run_ids:
        continue

    print(f"[{sid}] {sname}", flush=True)
    if args.verbose:
        print()

    t0 = time.time()
    try:
        passed, ratio, derr = run_overfit(
            sname, dec_kw,
            extra_fn   = extra_fn,
            use_grid   = use_grid,
            sem_cfg    = sem_cfg,
            lr_consist = lr_c,
            anneal_n   = anneal_n,
            disp_tol   = tol,
        )
        status = "PASS" if passed else "FAIL"
        print(f"  → {status}  ph_ratio={ratio:.3f}  "
              f"disp_err={derr:.2f}px  ({time.time()-t0:.1f}s)")
    except Exception as e:
        import traceback
        passed = False
        ratio  = derr = float("nan")
        print(f"  → ERROR ({time.time()-t0:.1f}s): {e}")
        if args.verbose:
            traceback.print_exc()

    results.append((sid, sname, passed, ratio, derr))

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print(f"{'ID':>4}  {'Scenario':<42}  {'ph_ratio':>8}  {'disp_err':>8}  {'':>6}")
print("-" * 72)
all_pass = True
for sid, sname, passed, ratio, derr in results:
    tag      = "✓ PASS" if passed else "✗ FAIL"
    all_pass = all_pass and passed
    print(f"{sid:>4}  {sname:<42}  {ratio:>8.3f}  {derr:>8.2f}  {tag}")
print("=" * 72)

n_pass = sum(1 for *_, p, _, _ in results if p)
n_fail = len(results) - n_pass
print(f"\n{n_pass}/{len(results)} passed", end="")
print(" — all good!" if all_pass else f", {n_fail} failed.")
sys.exit(0 if all_pass else 1)
