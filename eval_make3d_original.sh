#!/usr/bin/env bash
# Evaluate an ORIGINAL PlaneDepth checkpoint (the weights released with the
# paper) on Make3D. Only the flags the original models were trained with are
# passed here - none of the later additions (semantic gate, cross-plane
# attention, learned families, multiscale logits, adaptive range), so the
# state_dict loads exactly as upstream.
#
# Usage:
#   bash eval_make3d_original.sh <weights_folder> [width] [height] [make3d_root]
#
#   stage1                 -> width 640,  height 192
#   HRfinetune / stage3_sd -> width 1280, height 384
#
# Example:
#   bash eval_make3d_original.sh ./log/planedepth_sd/best_models 1280 384 ./make3d

set -e

WEIGHTS=${1:?usage: bash eval_make3d_original.sh <weights_folder> [width] [height] [make3d_root]}
WIDTH=${2:-1280}
HEIGHT=${3:-384}
DATA=${4:-./make3d}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python evaluate_depth_make3d.py \
--eval_stereo \
--eval_split make3d \
--data_path "$DATA" \
--load_weights_folder "$WEIGHTS" \
--models_to_load encoder depth \
--use_denseaspp \
--plane_residual \
--use_mixture_loss \
--post_process \
--batch_size 1 \
--width "$WIDTH" \
--height "$HEIGHT"
