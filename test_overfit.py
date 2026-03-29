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

import argparse, os, sys, time
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
parser.add_argument("--kitti",   action="store_true",
                    help="also run key scenarios on real KITTI stereo pairs "
                         "from test_assets/ (kitti_left_N.png + kitti_right_N.png)")
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


def warp_l2r(img_l, disp_pix):
    """warped_right[x] = img_left[x + disp]  (inverse stereo warp)"""
    xs = torch.arange(W, dtype=torch.float32, device=device)
    ys = torch.arange(H, dtype=torch.float32, device=device)
    gy2, gx2 = torch.meshgrid(ys, xs, indexing="ij")
    sx  = 2.0*(gx2[None, None] + disp_pix)/(W-1) - 1.0
    sy  = (2.0*gy2/(H-1) - 1.0)[None, None].expand(B, 1, H, W)
    g   = torch.cat([sx, sy], 1).permute(0, 2, 3, 1)
    return F.grid_sample(img_l, g, mode="bilinear",
                         padding_mode="border", align_corners=True)


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
    loss_u = (tu[:,:,:,:-1] - tu[:,:,:,1:]).abs() * wu        # both (B,1,H-2,W-3)
    loss_v = (tv[:,:,:-1,:] - tv[:,:,1:,:]).abs() * wv        # both (B,1,H-3,W-2)
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
                anneal_n=None, disp_tol=8.0, lr_override=None,
                imgs=None, steps_override=None):
    """
    dec_kw     : kwargs forwarded to make_decoder (xz_levels handled inside)
    extra_fn   : fn(out, img_l, img_r) -> scalar extra-loss tensor, or None
    use_grid   : pass GRID to decoder (required by xz_levels > 0)
    sem_cfg    : dict(n_xy, n_xz, n_yz, n_learned) → build SemanticPlaneGate
    lr_consist : also run decoder on img_right, add LR consistency term
    anneal_n   : if set, call dec.set_active_levels(anneal_n) before loop
    disp_tol   : max acceptable |mean_disp - GT_DISP| at the end (synthetic)
                 or None to skip GT check (real KITTI)
    imgs       : (img_l, img_r) tensors (B,3,H,W) to use instead of the
                 global synthetic pair; pass criterion becomes loss-ratio-only
                 plus sanity range [2, 80] px instead of GT check.
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

    opt  = torch.optim.Adam(params, lr=lr_override if lr_override else args.lr)
    grid = GRID if use_grid else None

    # Use provided images or fall back to global synthetic pair
    img_l = imgs[0] if imgs is not None else img_left
    img_r = imgs[1] if imgs is not None else img_right
    real_imgs = (imgs is not None)

    ph0 = ph_end = last_disp = None
    n_steps = steps_override if steps_override is not None else args.steps

    for step in range(n_steps):
        opt.zero_grad()

        feats_l = enc(img_l)

        sem_bias = None
        if gate is not None:
            fake_sem = torch.randn(B, N_SEM, H//4, W//4, device=device)
            sem_bias = gate(fake_sem)

        out_l  = dec(feats_l, input_grids=grid, sem_bias=sem_bias)
        warped = warp_r2l(img_r, out_l["disp"])
        ph     = photo(warped, img_l)
        total  = ph

        if extra_fn is not None:
            total = total + extra_fn(out_l, img_l, img_r)

        if lr_consist:
            feats_r = enc(img_r)
            out_r   = dec(feats_r, input_grids=grid)
            # Right decoder also needs a photometric signal, otherwise it only
            # receives gradients from the consistency term and anchors both
            # decoders at an equilibrium far from GT_DISP.
            ph_r  = photo(warp_l2r(img_l, out_r["disp"]), img_r)
            total = total + ph_r + 0.01 * lr_consistency_loss(
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

    ratio = ph_end / ph0
    if real_imgs:
        # For real images we don't know GT disparity; check:
        #  (1) loss decreases meaningfully, (2) disparity is in a sane range
        in_range = (2.0 <= last_disp <= 80.0)
        passed   = (ratio < 0.50) and in_range
        disp_err = last_disp   # report mean disparity instead of error
    else:
        disp_err = abs(last_disp - GT_DISP)
        passed   = (ratio < 0.50) and (disp_err < disp_tol)
    return passed, ratio, disp_err, real_imgs


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
# Each row: (id, name, dec_kw, extra_fn, use_grid, sem_cfg, lr_consist, anneal_n, disp_tol, lr_override)
# lr_override=None → use args.lr; set to a float to use a different lr for that scenario only.

SCENARIOS = [
    ("00", "Baseline",
     {}, None, False, None, False, None, 8, None),

    ("01", "Cross-plane attention",
     {"xz_levels": N_XZ, "use_cross_plane_attn": True},
     None, True, None, False, None, 8, None),

    ("02", "Adaptive plane range",
     {"adaptive_plane_range": True},
     None, False, None, False, None, 8, None),

    ("03", "Pixelwise plane residual",
     {"pixelwise_plane_residual": True},
     None, False, None, False, None, 8, None),

    ("04", "Learned plane families",
     {"num_learned_families": 2, "learned_planes_per_family": 5},
     None, True, None, False, None, 8, None),   # needs grid (accesses input_grids[:,0])

    # Multi-scale: aux heads at coarser scales use noisy features with random init;
    # use a 5× lower lr to prevent them destabilising the fine-scale head early on.
    ("05", "Multi-scale logit aggregation",
     {"use_multiscale_logits": True},
     None, False, None, False, None, 8, 2e-4),

    ("06", "Plane annealing (start low)",
     {},
     None, False, None, False, 16, 8, None),  # anneal_n=16: planes k=0..15 (48→~10px);
                                              # GT=20 is at k≈9, well within active set

    ("07", "Semantic gate (paper branch)",
     {},
     None, False, {"n_xy": N_XY, "n_xz": 0, "n_yz": 0, "n_learned": 0},
     False, None, 8, None),

    ("08", "Entropy regularisation",
     {}, extra_entropy, False, None, False, None, 8, None),

    ("09", "Focal photometric loss",
     {}, extra_focal, False, None, False, None, 8, None),

    ("10", "Confidence-weighted smoothness",
     {}, extra_conf_smooth, False, None, False, None, 8, None),

    ("11", "LR consistency loss",
     {}, None, False, None, True, None, 8, None),

    ("12", "Surface normal smoothness",
     {}, extra_normal, False, None, False, None, 8, None),

    # ── Combinations ─────────────────────────────────────────────────────────

    ("13", "Cross-attn + Learned families  [fixed bug]",
     {"xz_levels": N_XZ, "use_cross_plane_attn": True,
      "num_learned_families": 2, "learned_planes_per_family": 5},
     None, True, None, False, None, 8, None),

    ("14", "Multi-scale + Adaptive range",
     {"use_multiscale_logits": True, "adaptive_plane_range": True},
     None, False, None, False, None, 8, 2e-4),  # same lr fix as scenario 05

    ("15", "Semantic gate + Entropy",
     {}, extra_entropy,
     False, {"n_xy": N_XY, "n_xz": 0, "n_yz": 0, "n_learned": 0},
     False, None, 8, None),

    ("16", "All decoder flags",
     {"xz_levels": N_XZ, "use_cross_plane_attn": True,
      "adaptive_plane_range": True, "pixelwise_plane_residual": True,
      "num_learned_families": 2, "learned_planes_per_family": 5,
      "use_multiscale_logits": True},
     None, True, None, False, None, 10, 2e-4),  # lr fix for multiscale component

    ("17", "All loss augmentations",
     {}, extra_all, False, None, False, None, 10, None),

    ("18", "Kitchen sink (everything)",
     {"xz_levels": N_XZ, "use_cross_plane_attn": True,
      "adaptive_plane_range": True, "pixelwise_plane_residual": True,
      "num_learned_families": 2, "learned_planes_per_family": 5,
      "use_multiscale_logits": True},
     extra_all, True,
     {"n_xy": N_XY, "n_xz": N_XZ, "n_yz": 0, "n_learned": 10},
     True, None, 12, 2e-4),  # lr fix for multiscale
]

# ── KITTI image loader ────────────────────────────────────────────────────────

def load_kitti_pair(idx: int):
    """Load a real KITTI stereo pair from test_assets/ and return (left, right)
    tensors shaped (1,3,H,W) resized to the global H×W.

    Requires Pillow (listed in requirements.txt).
    """
    assets = os.path.join(os.path.dirname(__file__), "test_assets")
    lpath = os.path.join(assets, f"kitti_left_{idx}.png")
    rpath = os.path.join(assets, f"kitti_right_{idx}.png")

    try:
        from PIL import Image as PILImage
        import torchvision.transforms.functional as TF

        limg = PILImage.open(lpath).convert("RGB")
        rimg = PILImage.open(rpath).convert("RGB")
        lt   = TF.to_tensor(TF.resize(limg, [H, W])).unsqueeze(0).to(device)
        rt   = TF.to_tensor(TF.resize(rimg, [H, W])).unsqueeze(0).to(device)
        return lt, rt
    except ImportError:
        pass

    # Fallback: use cv2 if available
    try:
        import cv2
        def _load(path):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H))
            t   = torch.from_numpy(img.astype(np.float32) / 255.0)
            return t.permute(2, 0, 1).unsqueeze(0).to(device)
        return _load(lpath), _load(rpath)
    except (ImportError, cv2.error):
        pass

    raise RuntimeError(
        "Cannot load KITTI images: neither Pillow+torchvision nor cv2 found. "
        "Install Pillow (pip install Pillow) or opencv-python."
    )


# ── KITTI scenario table ───────────────────────────────────────────────────────
# Run a representative subset on real stereo pairs (no GT disparity known).
# Pass criterion: ph_ratio < 0.50  AND  mean disparity ∈ [2, 80] px.
KITTI_SCENARIO_IDS = ["00", "01", "07", "08", "11", "18"]


# ── Run all (or selected) scenarios ──────────────────────────────────────────
run_ids = set(args.only) if args.only else None
results = []

def _run_one(sid, sname, dec_kw, extra_fn, use_grid, sem_cfg,
             lr_c, anneal_n, tol, lr_ov, imgs=None):
    t0 = time.time()
    label = sname + (" [KITTI]" if imgs is not None else "")
    print(f"[{sid}] {label}", flush=True)
    if args.verbose:
        print()
    try:
        # Real images have rougher loss landscapes than the smooth synthetic
        # pair — lr=1e-3 causes overshooting on complex architectures.
        # Cap KITTI lr at 2e-4 unless a lower override is already requested.
        # LR consistency couples two decoders through a shared encoder, so
        # the joint optimization converges slower on real images — use 2x steps.
        KITTI_LR_CAP = 2e-4
        eff_lr_ov    = lr_ov
        steps_ov     = None
        if imgs is not None:
            base_lr  = lr_ov if lr_ov is not None else args.lr
            eff_lr_ov = min(base_lr, KITTI_LR_CAP)
            if lr_c:
                steps_ov = args.steps * 2
        passed, ratio, derr, real = run_overfit(
            sname, dec_kw,
            extra_fn       = extra_fn,
            use_grid       = use_grid,
            sem_cfg        = sem_cfg,
            lr_consist     = lr_c,
            anneal_n       = anneal_n,
            disp_tol       = tol,
            lr_override    = eff_lr_ov,
            imgs           = imgs,
            steps_override = steps_ov,
        )
        status = "PASS" if passed else "FAIL"
        if real:
            print(f"  → {status}  ph_ratio={ratio:.3f}  "
                  f"mean_disp={derr:.2f}px  ({time.time()-t0:.1f}s)")
        else:
            print(f"  → {status}  ph_ratio={ratio:.3f}  "
                  f"disp_err={derr:.2f}px  ({time.time()-t0:.1f}s)")
    except Exception as e:
        import traceback
        passed = False
        ratio  = derr = float("nan")
        print(f"  → ERROR ({time.time()-t0:.1f}s): {e}")
        if args.verbose:
            traceback.print_exc()
    return passed, ratio, derr

for row in SCENARIOS:
    sid, sname, dec_kw, extra_fn, use_grid, sem_cfg, lr_c, anneal_n, tol, lr_ov = row

    if run_ids and sid not in run_ids:
        continue

    passed, ratio, derr = _run_one(sid, sname, dec_kw, extra_fn, use_grid,
                                   sem_cfg, lr_c, anneal_n, tol, lr_ov)
    results.append((sid, sname, passed, ratio, derr, False))

# ── Real KITTI tests (optional) ───────────────────────────────────────────────
kitti_results = []
if args.kitti:
    print()
    print("── Real KITTI stereo pairs ──────────────────────────────────────────")
    # Find available pairs
    assets = os.path.join(os.path.dirname(__file__), "test_assets")
    pair_indices = sorted(
        int(f[len("kitti_left_"):-len(".png")])
        for f in os.listdir(assets)
        if f.startswith("kitti_left_") and f.endswith(".png")
        and os.path.exists(os.path.join(assets, f.replace("left", "right")))
    )
    if not pair_indices:
        print("  No KITTI pairs found in test_assets/ — skipping.")
    else:
        print(f"  Found {len(pair_indices)} stereo pair(s): {pair_indices}")
        scenario_map = {row[0]: row for row in SCENARIOS}
        for pair_idx in pair_indices:
            try:
                kitti_imgs = load_kitti_pair(pair_idx)
            except Exception as e:
                print(f"  Could not load pair {pair_idx}: {e}")
                continue

            for sid in KITTI_SCENARIO_IDS:
                if sid not in scenario_map:
                    continue
                row = scenario_map[sid]
                _, sname, dec_kw, extra_fn, use_grid, sem_cfg, lr_c, anneal_n, tol, lr_ov = row
                kid = f"K{pair_idx}{sid}"
                passed, ratio, derr = _run_one(
                    kid, sname, dec_kw, extra_fn, use_grid,
                    sem_cfg, lr_c, anneal_n, tol, lr_ov,
                    imgs=kitti_imgs,
                )
                kitti_results.append((kid, f"pair{pair_idx}/{sname}", passed, ratio, derr))

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print(f"{'ID':>4}  {'Scenario':<42}  {'ph_ratio':>8}  {'disp_err':>8}  {'':>6}")
print("-" * 72)
all_pass = True
for sid, sname, passed, ratio, derr, _ in results:
    tag      = "✓ PASS" if passed else "✗ FAIL"
    all_pass = all_pass and passed
    print(f"{sid:>4}  {sname:<42}  {ratio:>8.3f}  {derr:>8.2f}  {tag}")
print("=" * 72)

n_pass = sum(1 for *_, p, _, _, _ in results if p)
n_fail = len(results) - n_pass
print(f"\n{n_pass}/{len(results)} passed", end="")
print(" — all good!" if (all_pass and not kitti_results) else
      (f", {n_fail} failed." if not all_pass else ""))

if kitti_results:
    print()
    print("── KITTI real-image results ─────────────────────────────────────────")
    print(f"{'ID':>6}  {'Scenario':<45}  {'ph_ratio':>8}  {'mean_disp':>9}  {'':>6}")
    print("-" * 76)
    kitti_pass = True
    for kid, kname, kp, kr, kd in kitti_results:
        tag = "✓ PASS" if kp else "✗ FAIL"
        kitti_pass = kitti_pass and kp
        print(f"{kid:>6}  {kname:<45}  {kr:>8.3f}  {kd:>9.2f}  {tag}")
    print("=" * 76)
    nkp = sum(1 for *_, p, _, _ in kitti_results if p)
    print(f"\nKITTI: {nkp}/{len(kitti_results)} passed",
          "— all good!" if kitti_pass else f", {len(kitti_results)-nkp} failed.")
    all_pass = all_pass and kitti_pass

sys.exit(0 if all_pass else 1)
