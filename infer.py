"""Run a trained model on images and write coloured depth maps.

Inference only: no ground truth, no metrics, no split files, no velodyne. Point
it at a folder of images (or a single image, or a KITTI split list) and it
writes one folder of pictures.

    python infer.py --image_path ./kitti/2011_09_26/2011_09_26_drive_0002_sync/image_02/data \
        --load_weights_folder ./log/planedepth_sd/best_models \
        --use_denseaspp --plane_residual --use_mixture_loss \
        --output_dir ./qual_kitti/planedepth --width 1280 --height 384

Per image it writes `<stem>_pred.png` (the coloured depth map, at the original
image size), `<stem>_panel.png` (input above prediction) and, at the end,
`overview.png` with twelve of them.

The architecture flags are the ones the checkpoint was trained with, exactly as
in the evaluation scripts, so the same command shape works for PlaneDepth and
for the proposed model.
"""

from __future__ import absolute_import, division, print_function

import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from options import MonodepthOptions
import networks

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


def collect_images(image_path, data_path, side="l"):
    """Every image to run on, as (stem, path) pairs.

    Accepts a single image, a folder (searched recursively), or a KITTI split
    list, in which case --data_path says where the drives live.
    """
    if os.path.isfile(image_path) and image_path.endswith(".txt"):
        pairs = []
        camera = "image_02" if side == "l" else "image_03"
        for line in open(image_path):
            parts = line.split()
            if not parts:
                continue
            folder, frame = parts[0], int(parts[1]) if len(parts) > 1 else 0
            for ext in (".png", ".jpg"):
                candidate = os.path.join(data_path, folder, camera, "data",
                                         "{:010d}{}".format(frame, ext))
                if os.path.isfile(candidate):
                    stem = "{}_{:010d}".format(folder.split("/")[-1], frame)
                    pairs.append((stem, candidate))
                    break
        return pairs

    if os.path.isfile(image_path):
        return [(os.path.splitext(os.path.basename(image_path))[0], image_path)]

    if os.path.isdir(image_path):
        found = []
        for root, _, files in os.walk(image_path):
            for name in sorted(files):
                if name.endswith(IMAGE_EXTS):
                    found.append((os.path.splitext(name)[0],
                                  os.path.join(root, name)))
        # a plain frame number is ambiguous once several drives are walked
        if len({s for s, _ in found}) != len(found):
            found = [("{}_{}".format(os.path.basename(os.path.dirname(
                os.path.dirname(os.path.dirname(p)))), s), p) for s, p in found]
        return sorted(found)

    raise SystemExit("No such image, folder or list: " + image_path)


def build_model(opt, device):
    """The same construction the evaluation scripts use."""
    folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(folder), "Cannot find a folder at {}".format(folder)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    encoder_dict = torch.load(os.path.join(folder, "encoder.pth"),
                              map_location=device)
    encoder.load_state_dict({k: v for k, v in encoder_dict.items()
                             if k in encoder.state_dict()})

    depth = networks.DepthDecoder(
        encoder.num_ch_enc, opt.disp_levels, opt.disp_min, opt.disp_max,
        opt.num_ep, pe_type=opt.pe_type, use_denseaspp=opt.use_denseaspp,
        xz_levels=opt.xz_levels, yz_levels=opt.yz_levels,
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
        use_multiscale_logits=opt.use_multiscale_logits)
    depth.load_state_dict(torch.load(os.path.join(folder, "depth.pth"),
                                     map_location=device))

    encoder.to(device).eval()
    depth.to(device).eval()

    backbone = gate = None
    if opt.use_semantic_gate:
        backbone = networks.SegFormerBackbone(
            model_id=opt.segformer_model).to(device).eval()
        gate = networks.SemanticPlaneGate(
            num_classes=opt.semantic_num_classes, n_xy=opt.disp_levels,
            n_xz=opt.xz_levels, n_yz=opt.yz_levels,
            n_learned=opt.num_learned_families * opt.learned_planes_per_family)
        gate_path = os.path.join(folder, "semantic_gate.pth")
        assert os.path.isfile(gate_path), (
            "--use_semantic_gate was set but {} does not exist".format(gate_path))
        gate.load_state_dict(torch.load(gate_path, map_location=device))
        gate.to(device).eval()

    return encoder, depth, backbone, gate


def main():
    opt = MonodepthOptions().parser.parse_known_args()[0]
    if not opt.image_path:
        raise SystemExit("--image_path is required (an image, a folder, or a split list)")
    out_dir = opt.output_dir or "./inference"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, depth, backbone, gate = build_model(opt, device)
    print("-> {} on {} at {}x{}".format(
        os.path.basename(opt.load_weights_folder.rstrip("/")), device,
        opt.width, opt.height))

    images = collect_images(opt.image_path, opt.data_path)
    if not images:
        raise SystemExit("No images found under " + opt.image_path)
    print("-> {} images".format(len(images)))

    to_tensor = torchvision.transforms.ToTensor()
    grid = torch.meshgrid(torch.linspace(-1, 1, opt.width),
                          torch.linspace(-1, 1, opt.height), indexing="xy")
    grid = torch.stack(grid, dim=0)[None].to(device)

    panels = []
    with torch.no_grad():
        for n, (stem, path) in enumerate(images, start=1):
            original = Image.open(path).convert("RGB")
            ow, oh = original.size

            image = to_tensor(original)[None].to(device)
            image = F.interpolate(image, size=(opt.height, opt.width),
                                  mode="bicubic", align_corners=True).clamp(0., 1.)

            batch = torch.cat((image, torch.flip(image, [3])), 0) \
                if opt.post_process else image
            grids = grid.expand(batch.shape[0], -1, -1, -1)

            bias = gate(backbone(batch)) if gate is not None else None
            disp = depth(encoder(batch), grids, bias)["disp"][:, 0].cpu().numpy()
            if opt.post_process:
                disp = 0.5 * (disp[0] + disp[1, :, ::-1])
            else:
                disp = disp[0]

            pred = colorize(disp)
            pred = cv2.resize(pred, (ow, oh), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(out_dir, stem + "_pred.png"),
                        cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

            rgb = np.asarray(original, dtype=np.uint8)
            gap = np.full((8, ow, 3), 255, dtype=np.uint8)
            panel = np.concatenate([rgb, gap, pred], axis=0)
            cv2.imwrite(os.path.join(out_dir, stem + "_panel.png"),
                        cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            panels.append(panel)

            if n % 25 == 0 or n == len(images):
                print("   {}/{}".format(n, len(images)))

    picks = np.linspace(0, len(panels) - 1, min(12, len(panels))).astype(int)
    thumbs = [cv2.resize(panels[p], (426, 256)) for p in picks]
    while len(thumbs) % 4:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.concatenate(thumbs[r:r + 4], axis=1) for r in range(0, len(thumbs), 4)]
    cv2.imwrite(os.path.join(out_dir, "overview.png"),
                cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))

    print("-> wrote {} predictions to {}".format(len(images), out_dir))
    print("-> quick look: {}".format(os.path.join(out_dir, "overview.png")))


if __name__ == "__main__":
    main()
