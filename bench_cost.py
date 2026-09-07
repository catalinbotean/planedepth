"""Inference-cost benchmark for SCOPE-Depth (params / MACs / latency / memory).

Reports the *complete* inference cost, including the frozen SegFormer-B0
semantic branch, which is part of the inference graph whenever the Semantic
Plane Gate is enabled.

Run from the repo root, e.g.

    python bench_cost.py --use_denseaspp --use_mixture_loss --plane_residual \
        --yz_levels 16 --use_semantic_gate --use_cross_plane_attn \
        --bench_res 640x192 1280x384 --bench_iters 100

All training flags accepted by ``options.py`` are accepted here, so the model
that is measured is exactly the model that is trained.  Bench-specific flags:

    --bench_res     one or more WxH strings (default: 640x192 1280x384)
    --bench_iters   timed iterations per configuration (default: 100)
    --bench_warmup  warm-up iterations (default: 20)
    --bench_device  cuda | cpu (default: cuda when available)
    --bench_no_sem  also measure the depth branch alone (no SegFormer)
"""
from __future__ import absolute_import, division, print_function

import argparse
import copy
import time

import torch

import networks
from options import MonodepthOptions


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def count_params(module):
    """(total, trainable) parameters."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def make_grid(h, w, device):
    """Same normalised coordinate grid the Resize transform feeds at eval."""
    gx, gy = torch.meshgrid(torch.linspace(-1, 1, w),
                            torch.linspace(-1, 1, h), indexing="xy")
    return torch.stack([gx, gy], dim=0)[None].to(device)


def macs_of(fn, inputs):
    """Best-effort MAC count. Returns (value_in_G, backend) or (None, reason)."""
    class _Wrap(torch.nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f

        def forward(self, *a):
            return self.f(*a)

    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            flops = FlopCountAnalysis(_Wrap(fn), inputs)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            return flops.total() / 1e9, "fvcore (MACs; unsupported ops count as 0)"
    except Exception as e:  # noqa: BLE001 - report and fall through
        fv_err = e
    try:
        from thop import profile
        with torch.no_grad():
            macs, _ = profile(_Wrap(fn), inputs=inputs, verbose=False)
            return macs / 1e9, "thop (MACs)"
    except Exception:  # noqa: BLE001
        reason = str(fv_err).strip().splitlines()
        reason = reason[0] if reason else fv_err.__class__.__name__
        return None, "install fvcore or thop for MAC counts ({:.200s})".format(reason)


def timeit(fn, inputs, iters, warmup, device):
    with torch.no_grad():
        for _ in range(warmup):
            fn(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else float("nan")
    return dt * 1e3, 1.0 / dt, peak


def build_models(opt, device):
    """Build exactly what trainer.create_models() builds, in eval mode."""
    encoder = networks.ResnetEncoder(opt.num_layers, False).to(device).eval()
    depth = networks.DepthDecoder(
        encoder.num_ch_enc,
        opt.disp_levels, opt.disp_min, opt.disp_max, opt.num_ep,
        pe_type=opt.pe_type,
        use_denseaspp=opt.use_denseaspp,
        xz_levels=opt.xz_levels,
        yz_levels=opt.yz_levels,
        use_mixture_loss=opt.use_mixture_loss,
        render_probability=opt.render_probability,
        plane_residual=opt.plane_residual,
        pixelwise_plane_residual=opt.pixelwise_plane_residual,
        use_cross_plane_attn=opt.use_cross_plane_attn,
        cross_plane_attn_tau=opt.cross_plane_attn_tau,
        adaptive_plane_range=opt.adaptive_plane_range,
        adaptive_range_margin=opt.adaptive_range_margin,
        num_learned_families=opt.num_learned_families,
        learned_planes_per_family=opt.learned_planes_per_family,
        use_multiscale_logits=opt.use_multiscale_logits).to(device).eval()

    seg = gate = None
    if opt.use_semantic_gate:
        seg = networks.SegFormerBackbone(model_id=opt.segformer_model).to(device).eval()
        gate = networks.SemanticPlaneGate(
            num_classes=opt.semantic_num_classes,
            n_xy=opt.disp_levels, n_xz=opt.xz_levels, n_yz=opt.yz_levels,
            n_learned=opt.num_learned_families * opt.learned_planes_per_family,
        ).to(device).eval()

    return encoder, depth, seg, gate


def strip_proposed(opt):
    """A copy of opt with every module added on top of PlaneDepth disabled.

    Used by --bench_baseline so the cost of the contribution is measured as a
    difference against the architecture it is added to, at the same resolution
    and on the same GPU, rather than against a number quoted from a paper.
    """
    base = copy.deepcopy(opt)
    base.use_semantic_gate = False
    base.use_cross_plane_attn = False
    base.use_multiscale_logits = False
    base.pixelwise_plane_residual = False
    base.adaptive_plane_range = False
    base.num_learned_families = 0
    base.yz_levels = 0
    return base


# --------------------------------------------------------------------------
def main():
    base = MonodepthOptions()
    opt, _ = base.parser.parse_known_args()

    bench = argparse.ArgumentParser(add_help=False)
    bench.add_argument("--bench_res", nargs="+", default=["640x192", "1280x384"])
    bench.add_argument("--bench_iters", type=int, default=100)
    bench.add_argument("--bench_warmup", type=int, default=20)
    bench.add_argument("--bench_device", type=str, default=None)
    bench.add_argument("--bench_no_sem", action="store_true")
    bench.add_argument("--bench_baseline", action="store_true",
                       help="also measure the plain PlaneDepth architecture "
                            "(every proposed module disabled), so the added "
                            "cost is a measured difference")
    bopt, _ = bench.parse_known_args()

    device = torch.device(bopt.bench_device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print("device: {}".format(device))
    print("dictionary: XY={} XZ={} YZ={}  cpa={} tau={}  gate={}".format(
        opt.disp_levels, opt.xz_levels, opt.yz_levels,
        opt.use_cross_plane_attn, opt.cross_plane_attn_tau, opt.use_semantic_gate))

    encoder, depth, seg, gate = build_models(opt, device)

    # ---- parameters --------------------------------------------------------
    print("\n--- parameters (M) ---")
    rows, tot, tr = [], 0, 0
    for name, m in [("ResNet-50 encoder", encoder), ("plane decoder", depth),
                    ("SegFormer-B0 (frozen)", seg), ("semantic plane gate", gate)]:
        if m is None:
            continue
        t, n = count_params(m)
        tot, tr = tot + t, tr + n
        rows.append((name, t, n))
    for name, t, n in rows:
        print("  {:<24s} {:8.3f}   (trainable {:8.3f})".format(name, t / 1e6, n / 1e6))
    print("  {:<24s} {:8.3f}   (trainable {:8.3f})".format("TOTAL", tot / 1e6, tr / 1e6))
    if gate is not None:
        cpa = sum(p.numel() for n_, p in depth.named_parameters() if "cross_plane_attn" in n_)
        g = sum(p.numel() for p in gate.parameters())
        print("  proposed modules only: gate {} params, CPA {} params".format(g, cpa))

    # Freeze *after* reporting parameters, so the trainable column above stays
    # truthful. fvcore traces the model and its tracer refuses to treat a
    # tensor that requires grad as a constant, which is what made MAC counting
    # fail; nothing here is trained, so dropping grad is free.
    for module in (encoder, depth, seg, gate):
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    # ---- optional PlaneDepth baseline, for a measured delta ----------------
    base_models = None
    if bopt.bench_baseline:
        base_models = build_models(strip_proposed(opt), device)
        for module in base_models:
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        b_total = sum(sum(p.numel() for p in m.parameters())
                      for m in base_models if m is not None)
        print("  {:<24s} {:8.3f}   (PlaneDepth baseline)".format("BASELINE TOTAL",
                                                                b_total / 1e6))

    # ---- per-resolution cost ----------------------------------------------
    for res in bopt.bench_res:
        w, h = (int(v) for v in res.lower().split("x"))
        img = torch.rand(1, 3, h, w, device=device)
        grid = make_grid(h, w, device)

        def full(image, coord):
            feats = encoder(image)
            bias = None
            if seg is not None:
                bias = gate(seg(image))
            return depth(feats, coord, bias)

        def depth_only(image, coord):
            return depth(encoder(image), coord, None)

        variants = [("full (depth + semantic branch)" if seg is not None
                     else "depth branch", full)]
        if bopt.bench_no_sem and seg is not None:
            variants.append(("depth branch only (no SegFormer)", depth_only))
        if base_models is not None:
            b_encoder, b_depth, _, _ = base_models

            def baseline(image, coord):
                return b_depth(b_encoder(image), coord, None)

            variants.append(("PlaneDepth baseline (proposed modules off)",
                             baseline))

        print("\n--- {}x{} ---".format(w, h))
        for label, fn in variants:
            g_macs, backend = macs_of(fn, (img, grid))
            ms, fps, peak = timeit(fn, (img, grid), bopt.bench_iters,
                                   bopt.bench_warmup, device)
            print("  {}".format(label))
            if g_macs is not None:
                # fvcore and thop both count multiply-accumulates. Papers are
                # split on whether "GFLOPs" means that number or twice it, so
                # print both and say which is which.
                print("    MACs        : {:8.2f} G".format(g_macs))
                print("    FLOPs       : {:8.2f} G   (2 x MACs)   [{}]"
                      .format(2 * g_macs, backend))
            else:
                print("    MACs        : n/a  [{}]".format(backend))
            print("    latency     : {:8.2f} ms  ({:.1f} FPS, batch 1, fp32)".format(ms, fps))
            if device.type == "cuda":
                print("    peak memory : {:8.1f} MiB".format(peak))


if __name__ == "__main__":
    main()
