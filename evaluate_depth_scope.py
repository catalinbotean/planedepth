from __future__ import absolute_import, division, print_function
import os, sys
sys.path.append('../')

from PIL import Image
import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

# from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "./splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return m_disp#r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


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


def kitti_stem(line):
    """Filesystem-safe name for a split entry like '2011_09_26/..._sync 69 l'."""
    parts = line.split()
    drive = parts[0].split("/")[-1]
    frame = int(parts[1]) if len(parts) > 1 else 0
    return "{}_{:010d}".format(drive, frame)


def save_visualisations(out_dir, dataset, filenames, pred_disps, gt_depths, max_depth):
    """Write one rgb / prediction / ground-truth panel per image, plus a montage.

    Same file layout and colouring as evaluate_depth_make3d.py, so the same
    scene-ranking and figure-assembly scripts work on either dataset.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    panels = []
    for i in range(pred_disps.shape[0]):
        stem = kitti_stem(filenames[i])

        color = dataset[i][("color", "l")].numpy()             # 3, H, W in [0, 1]
        color = (np.transpose(color, (1, 2, 0)) * 255).astype(np.uint8)
        h, w = color.shape[:2]

        pred_rgb = colorize(pred_disps[i])
        pred_rgb = cv2.resize(pred_rgb, (w, h), interpolation=cv2.INTER_NEAREST)

        # the KITTI ground truth is sparse LiDAR: show it as disparity so it
        # shares the prediction's colour ordering, and leave gaps black
        gt = np.asarray(gt_depths[i], dtype=np.float32).copy()
        gt[gt > max_depth] = 0.
        with np.errstate(divide="ignore"):
            gt_disp = np.where(gt > 0, 1. / np.maximum(gt, 1e-3), 0.)
        gt_rgb = colorize(gt_disp)
        gt_rgb[gt <= 0] = 0
        gt_rgb = cv2.resize(gt_rgb, (w, h), interpolation=cv2.INTER_NEAREST)

        gap = np.full((8, w, 3), 255, dtype=np.uint8)
        panel = np.concatenate([color, gap, pred_rgb, gap, gt_rgb], axis=0)

        cv2.imwrite(os.path.join(out_dir, stem + "_pred.png"),
                    cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir, stem + "_panel.png"),
                    cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        panels.append(panel)

    if panels:
        picks = np.linspace(0, len(panels) - 1, min(12, len(panels))).astype(int)
        thumbs = [cv2.resize(panels[p], (426, 384)) for p in picks]
        while len(thumbs) % 4:
            thumbs.append(np.zeros_like(thumbs[0]))
        rows = [np.concatenate(thumbs[r:r + 4], axis=1)
                for r in range(0, len(thumbs), 4)]
        cv2.imwrite(os.path.join(out_dir, "kitti_overview.png"),
                    cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))

    print("-> Saved {} panels (rgb / prediction / ground truth) to {}"
          .format(pred_disps.shape[0], out_dir))
    print("-> Quick look: {}".format(os.path.join(out_dir, "kitti_overview.png")))


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set.

    Identical to evaluate_depth_HR.py except that the Semantic Plane Gate
    and Cross-Plane Attention are instantiated at inference, so a model
    trained with --use_semantic_gate / --use_cross_plane_attn is evaluated
    with the architecture it was trained with.
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                       opt.height, opt.width,
                                       opt.novel_frame_ids, is_train=False,
                                       use_crop=False, use_colmap=False, img_ext=".png")

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)
        
        
        if opt.net_type == "ResNet":
            encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
            encoder_dict = torch.load(encoder_path)
            encoder = networks.ResnetEncoder(opt.num_layers, False)
            depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, 
                                                    opt.disp_levels, 
                                                    opt.disp_min, 
                                                    opt.disp_max, 
                                                    opt.num_ep, 
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
                                                    use_multiscale_logits=opt.use_multiscale_logits)

            model_dict = encoder.state_dict()
            encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
            depth_decoder.load_state_dict(torch.load(decoder_path))
            
            encoder.cuda()
            encoder.eval()
            depth_decoder.cuda()
            depth_decoder.eval()

            # --- semantic branch (part of the inference graph when enabled) --
            semantic_backbone = None
            semantic_gate = None
            if opt.use_semantic_gate:
                semantic_backbone = networks.SegFormerBackbone(
                    model_id=opt.segformer_model).cuda().eval()
                semantic_gate = networks.SemanticPlaneGate(
                    num_classes=opt.semantic_num_classes,
                    n_xy=opt.disp_levels,
                    n_xz=opt.xz_levels,
                    n_yz=opt.yz_levels,
                    n_learned=opt.num_learned_families * opt.learned_planes_per_family)
                gate_path = os.path.join(opt.load_weights_folder, "semantic_gate.pth")
                assert os.path.isfile(gate_path), (
                    "--use_semantic_gate was set but {} does not exist. The gate "
                    "must be evaluated with the weights it was trained with."
                    .format(gate_path))
                semantic_gate.load_state_dict(torch.load(gate_path))
                semantic_gate.cuda().eval()
        elif opt.net_type == "PladeNet":
            model = networks.PladeNet(False, 
                                      opt.disp_levels, 
                                      opt.disp_min, 
                                      opt.disp_max, 
                                      opt.num_ep, 
                                      xz_levels=opt.xz_levels, 
                                      use_mixture_loss=opt.use_mixture_loss, 
                                      render_probability=opt.render_probability, 
                                      plane_residual=opt.plane_residual)
            model.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "plade.pth")))
            model.cuda()
            model.eval()
        elif opt.net_type == "FalNet":
            model = networks.FalNet(False, opt.height, opt.width, opt.disp_levels, opt.disp_min, opt.disp_max)
            model.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "fal.pth")))
            model.cuda()
            model.eval()

        pred_disps = []
        probabilities_max = []

        print("-> Computing predictions with size {}x{}".format(
            opt.width, opt.height))
        
        grid = torch.meshgrid(torch.linspace(-1, 1, opt.width), torch.linspace(-1, 1, opt.height), indexing="xy")
        grid = torch.stack(grid, dim=0)

        i = 0

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", "l")].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                grids = grid[None, ...].expand(input_color.shape[0], -1, -1, -1).cuda()
                
                if opt.net_type == "ResNet":
                    sem_bias = None
                    if semantic_gate is not None:
                        sem_bias = semantic_gate(semantic_backbone(input_color))
                    output = depth_decoder(encoder(input_color), grids, sem_bias)
                elif opt.net_type == "FalNet":
                    output = model(input_color)
                elif opt.net_type == "PladeNet":
                    output = model(input_color, grids)

                pred_disp = output["disp"][:, 0].cpu().numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                probabilities_max.append(output["probability"].amax(1).mean(-1).mean(-1).cpu().numpy())

        pred_disps = np.concatenate(pred_disps)
        probabilities_max = np.concatenate(probabilities_max)
        print(probabilities_max.mean())
        
    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]

        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 0.1 * 0.58 * opt.width / (pred_disp)

        if opt.eval_split == "eigen_raw" or opt.eval_split == "eigen_improved":
            gt_depth[gt_depth < MIN_DEPTH] = MIN_DEPTH
            gt_depth[gt_depth > MAX_DEPTH] = MAX_DEPTH
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            #the following crop is used in FalNet and PladeNet, which has a slight unfair improvement than Eigen crop.
            # crop = np.array([gt_height - 219, gt_height - 4,
            #                  44,  1180]).astype(np.int32)
            crop_mask = np.zeros(gt_depth.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            # ratio = np.mean(gt_depth) / np.mean(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.5f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

    if opt.eval_out_dir:
        print()
        save_visualisations(opt.eval_out_dir, dataset, filenames, pred_disps,
                            gt_depths, MAX_DEPTH)

    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
