"""
Smoke-test for all new features added across branches 1-12 + paper branch.

Run from the repo root:
    python test_new_features.py

No dataset or pretrained weights needed — everything uses small random tensors.
Prints PASS / FAIL for each component and exits with code 1 if anything fails.
"""

import sys, traceback, torch, torch.nn.functional as F, numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {DEVICE}\n")

TESTS   = []   # list of (name, fn) in definition order
RESULTS = {}

def register(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator

# ── shared tiny dimensions ────────────────────────────────────────────────
B, H, W        = 2, 64, 192
N_XY, N_XZ    = 10, 4
N_YZ           = 0
N_TOT          = N_XY + N_XZ + N_YZ
ENC_CHS        = [64, 128, 256, 512, 512]

def make_decoder(**kw):
    from networks import DepthDecoder
    cfg = dict(num_ch_enc=ENC_CHS, no_levels=N_XY,
               disp_min=2, disp_max=300,
               xz_levels=N_XZ, xz_min=0.18, xz_max=0.37,
               yz_levels=N_YZ, use_mixture_loss=True,
               use_skips=True, use_denseaspp=False)
    cfg.update(kw)
    return DepthDecoder(**cfg).to(DEVICE)

def make_feats():
    # The decoder upsamples once more at i=0 (no skip), so features must start
    # at (H//2, W//2) for the final decoder output to land at (H, W).
    feats, h, w = [], H // 2, W // 2
    for ch in ENC_CHS:
        feats.append(torch.randn(B, ch, h, w, device=DEVICE))
        h, w = max(h // 2, 1), max(w // 2, 1)
    return feats

def make_grid():
    ys = torch.linspace(-1, 1, H, device=DEVICE)
    xs = torch.linspace(-1, 1, W, device=DEVICE)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], 0).unsqueeze(0).expand(B, -1, -1, -1)

# ── 1. Baseline ───────────────────────────────────────────────────────────
@register("1. Baseline DepthDecoder forward")
def _():
    out = make_decoder()(make_feats(), make_grid())
    assert out["disp"].shape    == (B, 1, H, W)
    assert out["depth"].shape   == (B, 1, H, W)
    assert out["probability"].shape == (B, N_TOT, H, W)

# ── 2. Cross-plane attention ──────────────────────────────────────────────
@register("2a. CrossPlaneAttention module")
def _():
    from networks.depth_decoder import CrossPlaneAttention
    cpa  = CrossPlaneAttention(N_XY, N_XZ, 0, tau=1.0).to(DEVICE)
    logits       = torch.randn(B, N_TOT, H, W, device=DEVICE)
    disp_layered = torch.rand(B, N_TOT, H, W, device=DEVICE) + 1.0
    mask         = torch.ones_like(disp_layered)
    out = cpa(logits, disp_layered, mask)
    assert out.shape == logits.shape

@register("2b. DepthDecoder --use_cross_plane_attn")
def _():
    out = make_decoder(use_cross_plane_attn=True)(make_feats(), make_grid())
    assert "logits" in out

# ── 3. Adaptive plane range ───────────────────────────────────────────────
@register("3. DepthDecoder --adaptive_plane_range")
def _():
    out = make_decoder(adaptive_plane_range=True)(make_feats(), make_grid())
    assert out["disp"].shape == (B, 1, H, W)

# ── 4. Pixelwise plane residual ───────────────────────────────────────────
@register("4. DepthDecoder --pixelwise_plane_residual")
def _():
    out = make_decoder(plane_residual=True,
                       pixelwise_plane_residual=True)(make_feats(), make_grid())
    assert out["disp"].shape == (B, 1, H, W)

# ── 5. Learned plane families ─────────────────────────────────────────────
@register("5. DepthDecoder --num_learned_families 2")
def _():
    M, K = 2, 5
    out  = make_decoder(num_learned_families=M,
                        learned_planes_per_family=K)(make_feats(), make_grid())
    assert out["probability"].shape == (B, N_TOT + M * K, H, W)
    assert "learned_normals" in out

# ── 6. Mixture uncertainty ────────────────────────────────────────────────
@register("6. disp_var and depth_confidence present and valid")
def _():
    out = make_decoder(use_mixture_loss=True)(make_feats(), make_grid())
    assert "disp_var"         in out
    assert "depth_confidence" in out
    assert out["disp_var"].shape == (B, 1, H, W)
    assert (out["disp_var"] >= 0).all(), "variance must be non-negative"

# ── 7. SemanticPlaneGate ──────────────────────────────────────────────────
@register("7a. SemanticPlaneGate zero-init → zero bias")
def _():
    from networks import SemanticPlaneGate
    gate = SemanticPlaneGate(21, N_XY, N_XZ, 0).to(DEVICE)
    bias = gate(torch.randn(B, 21, H // 4, W // 4, device=DEVICE))
    assert bias.shape == (B, N_TOT, H // 4, W // 4)
    assert bias.abs().max().item() == 0.0, "gate_scale=0 → must be zero bias"

@register("7b. SemanticPlaneGate Cityscapes prior auto-init (19 classes)")
def _():
    from networks import SemanticPlaneGate
    gate = SemanticPlaneGate(19, N_XY, N_XZ, 0).to(DEVICE)
    assert gate.affinity.weight.abs().sum().item() > 0.0, \
        "Cityscapes prior should produce non-zero affinity weights"

@register("7c. Zero sem_bias → identical decoder output")
def _():
    dec  = make_decoder()
    feats, grid = make_feats(), make_grid()
    sem_bias = torch.zeros(B, N_TOT, H // 2, W // 2, device=DEVICE)
    out_none = dec(feats, grid, sem_bias=None)
    out_zero = dec(feats, grid, sem_bias=sem_bias)
    assert torch.allclose(out_none["disp"], out_zero["disp"], atol=1e-5)

@register("7d. Non-zero sem_bias changes logits")
def _():
    dec = make_decoder()
    torch.manual_seed(42)
    feats, grid = make_feats(), make_grid()
    out_base   = dec(feats, grid)
    sem_bias   = torch.ones(B, N_TOT, H, W, device=DEVICE) * 3.0
    out_biased = dec(feats, grid, sem_bias=sem_bias)
    assert not torch.allclose(out_base["logits"], out_biased["logits"])

# ── 8. Multi-scale logit aggregation ─────────────────────────────────────
@register("8. DepthDecoder --use_multiscale_logits")
def _():
    out = make_decoder(use_multiscale_logits=True)(make_feats(), make_grid())
    assert out["disp"].shape == (B, 1, H, W)

# ── 9. Coarse-to-fine plane annealing ────────────────────────────────────
@register("9. set_active_levels masks inactive XY planes")
def _():
    dec = make_decoder()
    dec.set_active_levels(3)
    assert dec.active_levels == 3
    out = dec(make_feats(), make_grid())
    inactive = out["probability"][:, 3:N_XY, :, :]
    assert inactive.max().item() < 1e-3, \
        f"inactive plane prob too high: {inactive.max().item():.6f}"

# ── 10. Loss functions ────────────────────────────────────────────────────
@register("10a. Entropy regularisation > 0")
def _():
    pi = F.softmax(torch.randn(B, N_TOT, H, W, device=DEVICE), dim=1)
    H_val = -(pi * (pi + 1e-8).log()).sum(1).mean()
    assert H_val.item() > 0

@register("10b. LR consistency loss non-negative")
def _():
    disp = torch.rand(B * 2, 1, H, W, device=DEVICE) * 50 + 2
    d_L  = disp[:B]
    d_R  = disp[B:].flip(-1)
    ys = torch.linspace(-1., 1., H, device=DEVICE)
    xs = torch.linspace(-1., 1., W, device=DEVICE)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gx = gx[None, None].expand(B, 1, -1, -1)
    gy = gy[None, None].expand(B, 1, -1, -1)
    sample_x   = (gx + d_L * 2.0 / max(W - 1, 1)).clamp(-1., 1.)
    sample_grid = torch.stack([sample_x[:, 0], gy[:, 0]], dim=-1)
    d_R_w = F.grid_sample(d_R, sample_grid, padding_mode="zeros", align_corners=True)
    valid = (d_R_w > 0).float()
    loss  = (torch.abs(d_L - d_R_w) * valid).sum() / (valid.sum() + 1e-8)
    assert loss.item() >= 0

@register("10c. get_smooth_loss_disp_confidence")
def _():
    from layers import get_smooth_loss_disp_confidence
    disp = torch.rand(B, 1, H, W, device=DEVICE) + 0.1
    img  = torch.rand(B, 3, H, W, device=DEVICE)
    conf = torch.rand(B, 1, H, W, device=DEVICE) + 0.1
    loss = get_smooth_loss_disp_confidence(disp, img, conf, gamma=1.0)
    assert loss.item() >= 0

@register("10d. Focal photometric weight != plain mean")
def _():
    ph = torch.rand(B, 1, H, W, device=DEVICE)
    w  = (ph.detach() / (ph.detach().mean() + 1e-8)).pow(1.0).clamp(max=4.0)
    assert (w * ph).mean().item() != ph.mean().item()

# ── 11. Variance stress test ──────────────────────────────────────────────
@register("11. disp_var >= 0 on 10 random seeds")
def _():
    dec = make_decoder(use_mixture_loss=True)
    for seed in range(10):
        torch.manual_seed(seed)
        out = dec(make_feats(), make_grid())
        assert (out["disp_var"] >= 0).all(), \
            f"Negative variance at seed {seed}: min={out['disp_var'].min():.6f}"

# ── 12. Full stack ────────────────────────────────────────────────────────
@register("12. Full stack: all flags + semantic gate open")
def _():
    M, K = 2, 5
    dec = make_decoder(
        use_cross_plane_attn=True,
        adaptive_plane_range=True,
        pixelwise_plane_residual=True,
        use_mixture_loss=True,
        use_multiscale_logits=True,
        num_learned_families=M,
        learned_planes_per_family=K,
    )
    from networks import SemanticPlaneGate
    n_all = N_TOT + M * K
    gate  = SemanticPlaneGate(21, N_XY, N_XZ, 0, n_learned=M * K).to(DEVICE)
    with torch.no_grad():
        gate.gate_scale.fill_(1.0)   # open the gate
    sem_bias = gate(torch.randn(B, 21, H // 4, W // 4, device=DEVICE))

    out = dec(make_feats(), make_grid(), sem_bias=sem_bias)
    assert out["disp"].shape             == (B, 1, H, W)
    assert out["disp_var"].shape         == (B, 1, H, W)
    assert out["depth_confidence"].shape == (B, 1, H, W)
    assert out["probability"].shape      == (B, n_all, H, W)
    assert "learned_normals" in out
    print(f"\n       disp  ∈ [{out['disp'].min():.3f}, {out['disp'].max():.3f}]")
    print(f"       var   ∈ [{out['disp_var'].min():.4f}, {out['disp_var'].max():.4f}]")
    print(f"       conf  ∈ [{out['depth_confidence'].min():.4f}, "
          f"{out['depth_confidence'].max():.4f}]")

# ── runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("PlaneDepth — new feature smoke tests")
    print("=" * 60)

    for name, fn in TESTS:
        try:
            fn()
            RESULTS[name] = "PASS"
            print(f"  [PASS] {name}")
        except Exception:
            RESULTS[name] = "FAIL"
            print(f"  [FAIL] {name}")
            traceback.print_exc()

    print()
    print("=" * 60)
    passed = sum(v == "PASS" for v in RESULTS.values())
    failed = sum(v == "FAIL" for v in RESULTS.values())
    print(f"Results: {passed} passed, {failed} failed / {len(RESULTS)} total")
    if failed == 0:
        print("All tests passed ✓")
    else:
        print("Some tests FAILED — see tracebacks above.")
        sys.exit(1)
