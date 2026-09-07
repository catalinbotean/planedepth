# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

from layers import HomographyWarp

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="./log")

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_zhou_mini", "eigen_full", "eigen_full_left", "odom", "benchmark"],
                                 default="eigen_full_left")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=50,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--alpha_smooth",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0.04)
        self.parser.add_argument("--self_distillation",
                                 type=float,
                                 help="self_distillation weight",
                                 default=0.)
        self.parser.add_argument("--gamma_smooth",
                                 type=float,
                                 help="gamma of smooth loss",
                                 default=2)
        self.parser.add_argument("--alpha_pc",
                                 type=float,
                                 help="perceptual loss weight",
                                 default=0.1)
        self.parser.add_argument("--disp_min",
                                 type=float,
                                 help="minimum depth",
                                 default=2.)
        self.parser.add_argument("--disp_max",
                                 type=float,
                                 help="maximum depth",
                                 default=300.)
        self.parser.add_argument("--disp_levels",
                                 type=int,
                                 help="num levels of disp",
                                 default=49)
        self.parser.add_argument("--disp_layers",
                                 type=int,
                                 help="num layers of disp",
                                 default=2)
        self.parser.add_argument("--novel_frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[])
        self.parser.add_argument("--net_type",
                                 type=str,
                                 help="train which network",
                                 default="ResNet",
                                 choices=["PladeNet", "ResNet", "FalNet"])
        self.parser.add_argument("--num_ep",
                                 type=int,
                                 help="train which stage",
                                 default=8)
        self.parser.add_argument("--warp_type",
                                 type=str,
                                 help="the type of warp",
                                 default="disp_warp",
                                 choices=["depth_warp", "disp_warp", "homography_warp"])
        self.parser.add_argument("--match_aug",
                                 action="store_true",
                                 help="if set, use color augmented data to compute loss")
        self.parser.add_argument("--use_denseaspp",
                                 action="store_true",
                                 help="use DenseAspp block in ResNet")
        self.parser.add_argument("--use_mom",
                                 action="store_true",
                                 help="use mirror occlusion mask")
        self.parser.add_argument("--flip_right",
                                 action="store_true",
                                 help="use fliped right image to train")
        self.parser.add_argument("--pc_net",
                                 type=str,
                                 help="the type of net to compute pc loss",
                                 default="vgg19",
                                 choices=["vgg19", "resnet18"])
        self.parser.add_argument("--xz_levels",
                                 type=int,
                                 help="num levels of xz plane",
                                 default=14)
        self.parser.add_argument("--yz_levels",
                                 type=int,
                                 help="num levels of yz plane",
                                 default=0)
        self.parser.add_argument("--use_mixture_loss",
                                 action="store_true",
                                 help="use mixture loss")
        self.parser.add_argument("--alpha_self",
                                 type=float,
                                 help="perceptual loss weight",
                                 default=0.)
        self.parser.add_argument("--depth_regression_space",
                                 type=str,
                                 help="how to compute regression depth",
                                 default="inv",
                                 choices=["inv", "exp"])
        self.parser.add_argument("--render_probability",
                                 action="store_true",
                                 help="If set, render probability as NeRF")
        self.parser.add_argument("--plane_residual",
                                 action="store_true",
                                 help="If set, use residual plane based on init plane")
        self.parser.add_argument("--pixelwise_plane_residual",
                                 action="store_true",
                                 help="If set, predict a per-pixel sub-level plane offset "
                                      "instead of the global (spatially-pooled) residual. "
                                      "Implies --plane_residual.")
        self.parser.add_argument("--use_cross_plane_attn",
                                 action="store_true",
                                 help="If set, apply cross-plane attention to enforce "
                                      "geometric consistency between XY/XZ/YZ families")
        self.parser.add_argument("--cross_plane_attn_tau",
                                 type=float,
                                 default=0.2,
                                 help="Temperature tau for the cross-plane geometric "
                                      "compatibility kernel, in LOG-disparity units "
                                      "(lower = sharper gates). The XY dictionary is "
                                      "log-uniform with a spacing of "
                                      "log(disp_min/disp_max)/(disp_levels-1) ~= 0.104, "
                                      "so tau=0.2 spans about two plane levels at every "
                                      "depth. Values >~1.0 make the kernel near-uniform.")
        self.parser.add_argument("--adaptive_plane_range",
                                 action="store_true",
                                 help="If set, predict per-image (d_near, d_far) from "
                                      "the encoder bottleneck instead of using fixed "
                                      "disp_min/disp_max for all images")
        self.parser.add_argument("--adaptive_range_margin",
                                 type=float,
                                 default=1.099,
                                 help="Max log-disparity shift for adaptive range head "
                                      "(default log(3)≈1.099 allows factor-of-3 shift)")
        self.parser.add_argument("--num_learned_families",
                                 type=int,
                                 default=0,
                                 help="Number of extra plane families with fully learnable "
                                      "unit normals (0 = disabled).  Each family covers "
                                      "--learned_planes_per_family depth levels and handles "
                                      "surfaces at arbitrary orientations (oblique planes).")
        self.parser.add_argument("--learned_planes_per_family",
                                 type=int,
                                 default=20,
                                 help="Depth levels per learned plane family")
        self.parser.add_argument("--no_crop",
                                 action="store_true",
                                 help="if set, do not use resize crop data aug")
        self.parser.add_argument("--pe_type",
                                 type=str,
                                 help="the type of positional embedding",
                                 default="neural",
                                 choices=["neural", "frequency"])
        self.parser.add_argument("--use_colmap",
                                 action="store_true",
                                 help="if set, use colmap instead of predicting pose by posenet")
        self.parser.add_argument("--colmap_path",
                                 type=str,
                                 help="path to the colmap data",
                                 default="./kitti_colmap")
        self.parser.add_argument("--no_stereo",
                                 action="store_true",
                                 help="if set, disable stereo supervised")
        self.parser.add_argument("--uncertainty_weighted_distillation",
                                 action="store_true",
                                 help="If set, weight the self-distillation loss by the teacher's "
                                      "mixture-model confidence (low-variance teacher predictions "
                                      "receive higher weight). Requires --use_mixture_loss.")
        self.parser.add_argument("--semantic_distill_weight",
                                 type=float,
                                 default=0.,
                                 help="If > 0, modulate the self-distillation loss by the "
                                      "agreement between the frozen segmenter's expected plane "
                                      "family (road->XZ, wall->YZ, sky->XY) and the model's own "
                                      "per-pixel family assignment. Pixels where semantics and "
                                      "geometry agree are trusted more as distillation targets; "
                                      "ambiguous pixels are down-weighted. Requires "
                                      "--use_semantic_gate and --self_distillation > 0.")
        self.parser.add_argument("--semantic_edge_smoothness",
                                 action="store_true",
                                 help="If set, relax disparity smoothness at semantic class "
                                      "boundaries (from the frozen segmenter) in addition to RGB "
                                      "image edges. Suppresses texture-induced false edges and "
                                      "aligns depth discontinuities with object boundaries. "
                                      "Requires --use_semantic_gate.")
        self.parser.add_argument("--gamma_smooth_semantic",
                                 type=float,
                                 default=2.,
                                 help="Edge sensitivity for the semantic-boundary term of "
                                      "--semantic_edge_smoothness (analogous to --gamma_smooth "
                                      "for RGB edges).")
        self.parser.add_argument("--use_semantic_gate",
                                 action="store_true",
                                 help="If set, run a frozen pretrained SegFormer-B0 "
                                      "(Cityscapes-19) segmenter on each input image and use "
                                      "the per-pixel class probabilities to bias plane-family "
                                      "logits via the learnable SemanticPlaneGate module: "
                                      "sky→XY, road/terrain→XZ, building/wall→YZ. Requires "
                                      "the `transformers` package (pip install transformers).")
        self.parser.add_argument("--segformer_model",
                                 type=str,
                                 default="nvidia/segformer-b0-finetuned-cityscapes-512-1024",
                                 help="HuggingFace model ID for the frozen SegFormer backbone "
                                      "used by --use_semantic_gate. Must be a 19-class "
                                      "Cityscapes model so the Cityscapes prior is activated.")
        self.parser.add_argument("--semantic_num_classes",
                                 type=int,
                                 default=19,
                                 help="Number of output classes from the semantic backbone "
                                      "(19 for Cityscapes SegFormer, which activates the "
                                      "hand-crafted Cityscapes prior in SemanticPlaneGate).")
        self.parser.add_argument("--plane_anneal_start",
                                 type=int,
                                 default=49,
                                 help="Number of active XY plane levels at epoch 0 for "
                                      "coarse-to-fine plane annealing. Set < disp_levels "
                                      "to enable (e.g. 10 or 25). Levels grow linearly to "
                                      "disp_levels over --plane_anneal_epochs epochs.")
        self.parser.add_argument("--plane_anneal_epochs",
                                 type=int,
                                 default=20,
                                 help="Number of epochs over which active XY plane levels "
                                      "grow from plane_anneal_start to disp_levels.")
        self.parser.add_argument("--focal_ph_gamma",
                                 type=float,
                                 default=0.0,
                                 help="Focal exponent for the photometric loss. gamma=0 "
                                      "gives standard L1/mixture loss. gamma>0 upweights "
                                      "hard pixels (large error) and downweights easy ones "
                                      "via w = (err/mean_err)^gamma (clamped at 4). "
                                      "Typical range: 0.5–2.0.")
        self.parser.add_argument("--use_multiscale_logits",
                                 action="store_true",
                                 help="If set, attach auxiliary plane-logit heads at decoder "
                                      "scales 1, 2, 3 (H/2, H/4, H/8) and add their upsampled "
                                      "outputs as a residual to the finest-scale logits. "
                                      "Zero-initialised so the base model is recovered at init. "
                                      "Lets coarser context (sky/road layout) correct fine-scale "
                                      "predictions without replacing them.")
        self.parser.add_argument("--use_confidence_smooth",
                                 action="store_true",
                                 help="If set, weight the disparity smoothness loss by the "
                                      "inverse mixture variance (depth_confidence from "
                                      "--use_mixture_loss). High-confidence regions are "
                                      "pushed to be smooth; uncertain regions (boundaries, "
                                      "sky) are relaxed. Falls back to plain smoothness "
                                      "when disp_var is not available.")
        self.parser.add_argument("--alpha_normal_smooth",
                                 type=float,
                                 default=0.0,
                                 help="Weight for image-edge-weighted surface normal "
                                      "smoothness loss. Scale-invariant complement to "
                                      "--alpha_smooth: penalises changes in surface "
                                      "orientation rather than disparity magnitude. "
                                      "Typical range: 0.01–0.1.")
        self.parser.add_argument("--alpha_entropy",
                                 type=float,
                                 default=0.0,
                                 help="Weight for mixture entropy regularization. "
                                      "Penalises high-entropy (near-uniform) plane "
                                      "distributions, forcing the network to commit to "
                                      "a single depth plane rather than averaging many. "
                                      "Typical range: 0.001–0.01.")
        self.parser.add_argument("--alpha_lr_consistency",
                                 type=float,
                                 default=0.0,
                                 help="Weight for the left-right geometric consistency loss. "
                                      "Requires --flip_right. At each left pixel with disparity "
                                      "d_L the right disparity map (warped by d_L) must agree: "
                                      "d_L(u) ≈ d_R(u + d_L(u)).  Typical range: 0.01–0.1.")

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=8)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--beta_1",
                                 type=float,
                                 help="beta1 of Adam",
                                 default=0.5)
        self.parser.add_argument("--beta_2",
                                 type=float,
                                 help="beta2 of Adam",
                                 default=0.999)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=50)
        self.parser.add_argument("--start_epoch",
                                 type=int,
                                 help="number of epochs",
                                 default=0)
        self.parser.add_argument('--milestones', 
                                 default=[30, 40], nargs='*',
                                 help='epochs at which learning rate is divided by 2')
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="epochs at which learning rate times 0.1",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--automask",
                                 help="if set, do auto-masking",
                                 action="store_true")

        # SYSTEM options
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth"])
        self.parser.add_argument("--stage1_weights_folder",
                                 type=str,
                                 help="path of teacher model to load")

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=500)
        
        self.parser.add_argument("--log_img_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        
        self.parser.add_argument("--use_ssim",
                                 help="if set, use ssim in the loss",
                                 action="store_true")

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen_raw",
                                 choices=[
                                    "eigen_raw", "eigen_improved", "eigen_benchmark", "benchmark", "odom_9", "odom_10", "city", "make3d"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--image_path",
                                 type=str,
                                 help="inference only (infer.py): an image, a folder "
                                      "of images searched recursively, or a split "
                                      "list, in which case --data_path says where "
                                      "the drives are")
        self.parser.add_argument("--output_dir",
                                 type=str,
                                 help="inference only (infer.py): where to write the "
                                      "coloured depth maps (default ./inference)")
        self.parser.add_argument("--make3d_max_depth",
                                 type=float,
                                 default=70.,
                                 help="ground truth depths above this are ignored when "
                                      "evaluating on Make3D. 70m is the standard C1 "
                                      "error protocol; pass 80 for C2.")
        self.parser.add_argument("--make3d_crop_ratio",
                                 type=float,
                                 default=2.,
                                 help="aspect ratio of the centre band kept from the "
                                      "(nearly square) Make3D images and from the laser "
                                      "grid: a fraction 1/(1.33333*ratio) of the height "
                                      "is kept. 2 is the standard protocol.")
        self.parser.add_argument("--make3d_allow_truncated",
                                 action="store_true",
                                 help="decode truncated Make3D JPEGs instead of "
                                      "stopping. A few images in the Test134 archive "
                                      "are short by a few bytes; the missing rows "
                                      "decode as grey, so that image's metrics are "
                                      "biased. Check the download first with "
                                      "scripts/check_make3d.py.")
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
