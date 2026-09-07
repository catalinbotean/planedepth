"""Evaluate a monodepth2-lineage baseline on Make3D, under our exact protocol.

Why this file is self-contained
-------------------------------
It runs *inside the baseline's own repository* (it needs that repo's `networks`
package), so it cannot import anything from PlaneDepth. The Make3D loading,
the centre crop, the median scaling and the metrics are therefore duplicated
from evaluate_depth_make3d.py on purpose - keep the two in sync if the protocol
changes.

Supported architectures
-----------------------
  monodepth2   https://github.com/nianticlabs/monodepth2
               networks.ResnetEncoder + networks.DepthDecoder
  litemono     https://github.com/noahzn/Lite-Mono
               networks.LiteMono + networks.DepthDecoder(scales=range(3))
  tinydepth    https://github.com/ZYCheng777/TinyDepth
               networks.build_model(get_config(opt)) + networks.FusionDecoder

All three predict sigmoid disparity at ("disp", 0) and turn it into depth with
disp_to_depth(disp, 0.1, 100), which is reimplemented here so no import from
the host repo's layers.py / layer.py is needed.

Usage
-----
    cp scripts/make3d_baseline_eval.py /path/to/monodepth2/
    cd /path/to/monodepth2
    python make3d_baseline_eval.py --arch monodepth2 \\
        --weights ~/models/mono+stereo_640x192 --data_path ~/make3d

Monodepth2's published Make3D numbers are abs_rel 0.322, sq_rel 3.589,
rmse 7.417, log10 0.163 (mono) - reproducing them is the point of running this:
it shows our protocol matches the one used in the literature.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

cv2.setNumThreads(0)

IMG_PREFIX = "img-"
DEPTH_PREFIX = "depth_sph_corr-"
MIN_DEPTH = 1e-3


# ── protocol: identical to evaluate_depth_make3d.py ───────────────────────
def crop_band(height, crop_ratio):
    """Rows kept by the standard Make3D centre crop, as (top, bottom)."""
    h_ratio = 1. / (1.33333 * crop_ratio)
    return int(height * (1. - h_ratio) / 2.), int(height * (1. + h_ratio) / 2.)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    log10 = np.mean(np.abs(np.log10(gt) - np.log10(pred)))
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3


def disp_to_depth(disp, min_depth=0.1, max_depth=100.):
    """monodepth2's sigmoid disparity -> depth, reimplemented to avoid an import."""
    min_disp = 1. / max_depth
    max_disp = 1. / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    return scaled_disp, 1. / scaled_disp


def batch_post_process_disparity(l_disp, r_disp):
    """Monodepthv1 flip post-processing."""
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def resolve_dir(data_path, name, pattern):
    candidates = [os.path.join(data_path, name), data_path]
    candidates += sorted(glob.glob(os.path.join(data_path, "*", name)))
    for candidate in candidates:
        if glob.glob(os.path.join(candidate, pattern)):
            return candidate
    raise SystemExit("Could not find '{}' under {} (looked for a '{}' folder)."
                     .format(pattern, data_path, name))


def load_make3d(data_path, height, width, crop_ratio, allow_truncated):
    """Return (colour tensor B,3,H,W in [0,1], list of ground-truth arrays)."""
    from scipy.io import loadmat

    if allow_truncated:
        import PIL.ImageFile
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

    img_dir = resolve_dir(data_path, "Test134", IMG_PREFIX + "*.jpg")
    depth_dir = resolve_dir(data_path, "Gridlaserdata", DEPTH_PREFIX + "*.mat")
    to_tensor = torchvision.transforms.ToTensor()

    colors, gts, stems = [], [], []
    for image_path in sorted(glob.glob(os.path.join(img_dir, IMG_PREFIX + "*.jpg"))):
        stem = os.path.basename(image_path)[len(IMG_PREFIX):-len(".jpg")]
        depth_path = os.path.join(depth_dir, DEPTH_PREFIX + stem + ".mat")
        if not os.path.isfile(depth_path):
            continue

        color = to_tensor(Image.open(image_path).convert("RGB"))
        top, bottom = crop_band(color.shape[1], crop_ratio)
        color = color[:, top:bottom, :]
        color = F.interpolate(color[None], size=(height, width),
                              mode="bicubic", align_corners=True)[0]
        colors.append(color.clamp(0., 1.))

        gt = np.asarray(loadmat(depth_path)["Position3DGrid"][:, :, 3], dtype=np.float32)
        top, bottom = crop_band(gt.shape[0], crop_ratio)
        gts.append(gt[top:bottom, :])
        stems.append(stem)

    if not colors:
        raise SystemExit("No Make3D image/ground-truth pairs under " + data_path)
    return torch.stack(colors), gts, stems


# ── per-architecture model construction ───────────────────────────────────
def build_monodepth2(args, encoder_dict):
    import networks
    encoder = networks.ResnetEncoder(args.num_layers, False)
    encoder.load_state_dict(
        {k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))
    decoder.load_state_dict(torch.load(os.path.join(args.weights, "depth.pth"),
                                       map_location="cpu"))
    return encoder, decoder


def build_litemono(args, encoder_dict):
    import networks
    encoder = networks.LiteMono(model=args.model,
                                height=args.height, width=args.width)
    encoder.load_state_dict(
        {k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    decoder_dict = torch.load(os.path.join(args.weights, "depth.pth"),
                              map_location="cpu")
    decoder.load_state_dict(
        {k: v for k, v in decoder_dict.items() if k in decoder.state_dict()})
    return encoder, decoder


def build_tinydepth(args, encoder_dict):
    import networks
    from networks.configuration import get_config

    # get_config() wants that repo's own options object. Parse it with an empty
    # argv so our flags do not reach their parser, then override the size.
    from options import MonodepthOptions
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        opt = MonodepthOptions().parse()
    finally:
        sys.argv = argv
    opt.height, opt.width = args.height, args.width

    encoder = networks.build_model(get_config(opt))
    encoder.load_state_dict(
        {k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    decoder = networks.FusionDecoder([64, 64, 128, 160, 320])
    decoder.load_state_dict(torch.load(os.path.join(args.weights, "depth.pth"),
                                       map_location="cpu"), strict=False)
    return encoder, decoder


BUILDERS = {"monodepth2": build_monodepth2,
            "litemono": build_litemono,
            "tinydepth": build_tinydepth}


def main():
    parser = argparse.ArgumentParser(
        description="Make3D evaluation for a monodepth2-lineage baseline")
    parser.add_argument("--arch", required=True, choices=sorted(BUILDERS),
                        help="which baseline repository this is being run in")
    parser.add_argument("--weights", required=True,
                        help="folder holding encoder.pth and depth.pth")
    parser.add_argument("--data_path", default="./make3d",
                        help="Make3D root (holds Test134/ and Gridlaserdata/)")
    parser.add_argument("--model", default="lite-mono",
                        help="litemono only: lite-mono, lite-mono-small, "
                             "lite-mono-tiny or lite-mono-8m")
    parser.add_argument("--num_layers", type=int, default=18,
                        help="monodepth2 only: ResNet depth of the checkpoint")
    parser.add_argument("--height", type=int, default=0,
                        help="network input height (default: from the checkpoint)")
    parser.add_argument("--width", type=int, default=0,
                        help="network input width (default: from the checkpoint)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_depth", type=float, default=70.,
                        help="ground-truth cap: 70 is the C1 error, 80 is C2")
    parser.add_argument("--crop_ratio", type=float, default=2.)
    parser.add_argument("--post_process", action="store_true",
                        help="Monodepthv1 flip post-processing (off by default: "
                             "the published Make3D tables do not use it)")
    parser.add_argument("--allow_truncated", action="store_true",
                        help="decode the truncated JPEG in Test134")
    args = parser.parse_args()

    if not os.path.isdir(args.weights):
        raise SystemExit("No such weights folder: " + args.weights)

    encoder_path = os.path.join(args.weights, "encoder.pth")
    if not os.path.isfile(encoder_path):
        raise SystemExit("No encoder.pth in " + args.weights)
    encoder_dict = torch.load(encoder_path, map_location="cpu")

    # monodepth2 and Lite-Mono store the training resolution in the checkpoint
    if not args.height:
        args.height = int(encoder_dict.get("height", 192))
    if not args.width:
        args.width = int(encoder_dict.get("width", 640))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = BUILDERS[args.arch](args, encoder_dict)
    encoder.to(device).eval()
    decoder.to(device).eval()

    print("-> {} at {}x{} on {}".format(args.arch, args.width, args.height, device))
    colors, gts, stems = load_make3d(args.data_path, args.height, args.width,
                                     args.crop_ratio, args.allow_truncated)
    print("-> {} Make3D test images".format(len(stems)))

    pred_disps = []
    with torch.no_grad():
        for start in range(0, colors.shape[0], args.batch_size):
            batch = colors[start:start + args.batch_size].to(device)
            if args.post_process:
                batch = torch.cat((batch, torch.flip(batch, [3])), 0)

            output = decoder(encoder(batch))
            disp = output[("disp", 0)]
            if disp.shape[1] > 1:           # TinyDepth returns several channels
                disp = disp[:, :1]
            disp = disp[:, 0].cpu().numpy()

            if args.post_process:
                n = disp.shape[0] // 2
                disp = batch_post_process_disparity(disp[:n], disp[n:, :, ::-1])
            pred_disps.append(disp)
    pred_disps = np.concatenate(pred_disps)

    errors, ratios = [], []
    for i in range(pred_disps.shape[0]):
        gt_depth = gts[i]
        gt_height, gt_width = gt_depth.shape[:2]

        disp = cv2.resize(pred_disps[i], (gt_width, gt_height))
        _, pred_depth = disp_to_depth(disp, 0.1, 100.)

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < args.max_depth)
        if not mask.any():
            continue
        pred_depth = pred_depth[mask]
        gt_masked = gt_depth[mask]

        ratio = np.median(gt_masked) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth = pred_depth * ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        errors.append(compute_errors(gt_masked, pred_depth))

    ratios = np.array(ratios)
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}"
          .format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)
    print("\n   Make3D C1 error (ground truth capped at {}m, {} images)"
          .format(args.max_depth, len(errors)))
    print("\n  " + ("{:>8} | " * 8).format(
        "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "a1", "a2", "a3"))
    print(("&{: 8.5f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    main()
