"""Run a monodepth2-lineage baseline over images and write coloured depth maps.

Inference only, and self-contained: it runs *inside the baseline's own
repository*, because it needs that repo's `networks` package, so it imports
nothing from PlaneDepth. The outputs match infer.py exactly - same colouring,
same file names - so one folder per method can be compared side by side or fed
to scripts/build_fig.py.

    cp scripts/infer_baseline.py /path/to/monodepth2/
    cd /path/to/monodepth2
    python infer_baseline.py --arch monodepth2 --weights ./models/stereo_1024x320 \
        --image_path ~/kitti/2011_09_26/2011_09_26_drive_0002_sync/image_02/data \
        --output_dir ~/qual_kitti/monodepth2

Architectures, verified against each repo's own inference code:

    monodepth2   networks.ResnetEncoder + networks.DepthDecoder
    litemono     networks.LiteMono + networks.DepthDecoder(scales=range(3))
    tinydepth    networks.build_model(get_config(opt)) + networks.FusionDecoder

All three emit sigmoid disparity at ("disp", 0), which is what gets coloured -
near is bright, as in the PlaneDepth figures.
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

cv2.setNumThreads(0)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".JPG", ".PNG")


def colorize(values, cmap="magma", vmin=None, vmax=None):
    """Map a 2D float array to an HxWx3 uint8 RGB image."""
    import matplotlib

    try:                       # matplotlib >= 3.5, and the only API in >= 3.9
        cmap_fn = matplotlib.colormaps[cmap]
    except AttributeError:
        import matplotlib.cm
        cmap_fn = matplotlib.cm.get_cmap(cmap)

    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values) & (values > 0)]
    if vmin is None:
        vmin = np.percentile(finite, 5) if finite.size else 0.
    if vmax is None:
        vmax = np.percentile(finite, 95) if finite.size else 1.
    if vmax <= vmin:
        vmax = vmin + 1e-6
    normed = np.clip((values - vmin) / (vmax - vmin), 0., 1.)
    return (cmap_fn(normed)[..., :3] * 255).astype(np.uint8)


def collect_images(image_path, data_path):
    """Every image to run on, as (stem, path) pairs.

    Accepts a single image, a folder searched recursively, or a KITTI split
    list, in which case data_path says where the drives live.
    """
    if os.path.isfile(image_path) and image_path.endswith(".txt"):
        pairs = []
        for line in open(image_path):
            parts = line.split()
            if not parts:
                continue
            folder = parts[0]
            frame = int(parts[1]) if len(parts) > 1 else 0
            for ext in (".png", ".jpg"):
                candidate = os.path.join(data_path, folder, "image_02", "data",
                                         "{:010d}{}".format(frame, ext))
                if os.path.isfile(candidate):
                    pairs.append(("{}_{:010d}".format(folder.split("/")[-1], frame),
                                  candidate))
                    break
        return pairs

    if os.path.isfile(image_path):
        return [(os.path.splitext(os.path.basename(image_path))[0], image_path)]

    if os.path.isdir(image_path):
        found = []
        for root, _, files in os.walk(image_path):
            for name in sorted(files):
                if name.endswith(IMAGE_EXTS):
                    found.append((os.path.splitext(name)[0], os.path.join(root, name)))
        if len({s for s, _ in found}) != len(found):
            found = [("{}_{}".format(os.path.basename(os.path.dirname(
                os.path.dirname(os.path.dirname(p)))), s), p) for s, p in found]
        return sorted(found)

    raise SystemExit("No such image, folder or list: " + image_path)


# ── per-architecture model construction ───────────────────────────────────
def build_monodepth2(args, encoder_dict, device):
    import networks
    encoder = networks.ResnetEncoder(args.num_layers, False)
    encoder.load_state_dict({k: v for k, v in encoder_dict.items()
                             if k in encoder.state_dict()})
    decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))
    decoder.load_state_dict(torch.load(os.path.join(args.weights, "depth.pth"),
                                       map_location=device))
    return encoder, decoder


def build_litemono(args, encoder_dict, device):
    import networks
    encoder = networks.LiteMono(model=args.model, height=args.height, width=args.width)
    encoder.load_state_dict({k: v for k, v in encoder_dict.items()
                             if k in encoder.state_dict()})
    decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    state = torch.load(os.path.join(args.weights, "depth.pth"), map_location=device)
    decoder.load_state_dict({k: v for k, v in state.items()
                             if k in decoder.state_dict()})
    return encoder, decoder


def build_tinydepth(args, encoder_dict, device):
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
    encoder.load_state_dict({k: v for k, v in encoder_dict.items()
                             if k in encoder.state_dict()})
    decoder = networks.FusionDecoder([64, 64, 128, 160, 320])
    decoder.load_state_dict(torch.load(os.path.join(args.weights, "depth.pth"),
                                       map_location=device), strict=False)
    return encoder, decoder


BUILDERS = {"monodepth2": build_monodepth2,
            "litemono": build_litemono,
            "tinydepth": build_tinydepth}


def main():
    ap = argparse.ArgumentParser(
        description="Inference-only depth maps from a monodepth2-lineage baseline")
    ap.add_argument("--arch", required=True, choices=sorted(BUILDERS),
                    help="which baseline repository this is being run in")
    ap.add_argument("--weights", required=True,
                    help="folder holding encoder.pth and depth.pth")
    ap.add_argument("--image_path", required=True,
                    help="an image, a folder searched recursively, or a split list")
    ap.add_argument("--data_path", default="./kitti",
                    help="drives root, when --image_path is a split list")
    ap.add_argument("--output_dir", default="./inference")
    ap.add_argument("--model", default="lite-mono",
                    help="litemono only: lite-mono, lite-mono-small, "
                         "lite-mono-tiny or lite-mono-8m")
    ap.add_argument("--num_layers", type=int, default=18,
                    help="monodepth2 only: ResNet depth of the checkpoint")
    ap.add_argument("--height", type=int, default=0,
                    help="network input height (default: from the checkpoint)")
    ap.add_argument("--width", type=int, default=0,
                    help="network input width (default: from the checkpoint)")
    ap.add_argument("--post_process", action="store_true",
                    help="average the horizontally flipped pass as well")
    args = ap.parse_args()

    encoder_path = os.path.join(args.weights, "encoder.pth")
    if not os.path.isfile(encoder_path):
        raise SystemExit("No encoder.pth in " + args.weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_dict = torch.load(encoder_path, map_location=device)

    # monodepth2 and Lite-Mono store the training resolution in the checkpoint
    if not args.height:
        args.height = int(encoder_dict.get("height", 192))
    if not args.width:
        args.width = int(encoder_dict.get("width", 640))

    encoder, decoder = BUILDERS[args.arch](args, encoder_dict, device)
    encoder.to(device).eval()
    decoder.to(device).eval()
    os.makedirs(args.output_dir, exist_ok=True)
    print("-> {} on {} at {}x{}".format(args.arch, device, args.width, args.height))

    images = collect_images(args.image_path, args.data_path)
    if not images:
        raise SystemExit("No images found under " + args.image_path)
    print("-> {} images".format(len(images)))

    to_tensor = torchvision.transforms.ToTensor()
    panels = []
    with torch.no_grad():
        for n, (stem, path) in enumerate(images, start=1):
            original = Image.open(path).convert("RGB")
            ow, oh = original.size

            image = to_tensor(original)[None].to(device)
            image = F.interpolate(image, size=(args.height, args.width),
                                  mode="bilinear", align_corners=False)
            if args.post_process:
                image = torch.cat((image, torch.flip(image, [3])), 0)

            disp = decoder(encoder(image))[("disp", 0)]
            if disp.shape[1] > 1:              # TinyDepth returns several channels
                disp = disp[:, :1]
            disp = disp[:, 0].cpu().numpy()
            disp = 0.5 * (disp[0] + disp[1, :, ::-1]) if args.post_process else disp[0]

            pred = colorize(disp)
            pred = cv2.resize(pred, (ow, oh), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(args.output_dir, stem + "_pred.png"),
                        cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

            rgb = np.asarray(original, dtype=np.uint8)
            gap = np.full((8, ow, 3), 255, dtype=np.uint8)
            panel = np.concatenate([rgb, gap, pred], axis=0)
            cv2.imwrite(os.path.join(args.output_dir, stem + "_panel.png"),
                        cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            panels.append(panel)

            if n % 25 == 0 or n == len(images):
                print("   {}/{}".format(n, len(images)))

    picks = np.linspace(0, len(panels) - 1, min(12, len(panels))).astype(int)
    thumbs = [cv2.resize(panels[p], (426, 256)) for p in picks]
    while len(thumbs) % 4:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.concatenate(thumbs[r:r + 4], axis=1) for r in range(0, len(thumbs), 4)]
    cv2.imwrite(os.path.join(args.output_dir, "overview.png"),
                cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))

    print("-> wrote {} predictions to {}".format(len(images), args.output_dir))
    print("-> quick look: {}".format(os.path.join(args.output_dir, "overview.png")))


if __name__ == "__main__":
    main()
