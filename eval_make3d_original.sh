#!/usr/bin/env bash
# Evaluate an ORIGINAL PlaneDepth checkpoint (the weights released with the
# paper) on Make3D. Only the flags the original models were trained with are
# passed here - none of the later additions (semantic gate, cross-plane
# attention, learned families, multiscale logits, adaptive range), so the
# state_dict loads exactly as upstream.
#
# Usage:
#   bash eval_make3d_original.sh                                  # all defaults
#   bash eval_make3d_original.sh <weights_folder>
#   bash eval_make3d_original.sh <weights_folder> <width> <height> [make3d_root]
#   bash eval_make3d_original.sh <weights_folder> 640 192 ./make3d --no_eval
#
# Anything after the fourth argument is forwarded to evaluate_depth_make3d.py
# unchanged.
#
# With no arguments it looks for a single folder containing depth.pth, and uses
# 1280x384 (HRfinetune / self-distillation resolution) with ./make3d as the
# dataset root. For stage1 pass 640 192.
#
# Environment:
#   CUDA_VISIBLE_DEVICES   GPU to use (default 0)
#   MAKE3D_STRICT=1        do not pass --make3d_allow_truncated

set -e

WEIGHTS=${1:-}
WIDTH=${2:-1280}
HEIGHT=${3:-384}
DATA=${4:-./make3d}
[ $# -gt 4 ] && shift 4 || shift $#
EXTRA=("$@")

# ── locate the weights when not given ─────────────────────────────────────
if [ -z "$WEIGHTS" ]; then
    FOUND=$(find . -name depth.pth -not -path "*/\.*" 2>/dev/null | sed 's|/depth\.pth$||' | sort)
    COUNT=$(printf '%s\n' "$FOUND" | sed '/^$/d' | wc -l | tr -d ' ')
    if [ "$COUNT" = "0" ]; then
        echo "No depth.pth found under $(pwd)." >&2
        echo "Pass the weights folder explicitly: bash $0 <weights_folder>" >&2
        exit 1
    elif [ "$COUNT" = "1" ]; then
        WEIGHTS=$FOUND
    else
        echo "Several checkpoints found - pass the one you want as the first argument:" >&2
        printf '  %s\n' $FOUND >&2
        exit 1
    fi
fi

if [ ! -f "$WEIGHTS/depth.pth" ] || [ ! -f "$WEIGHTS/encoder.pth" ]; then
    echo "$WEIGHTS does not contain both encoder.pth and depth.pth." >&2
    exit 1
fi

# One image of Test134 ships truncated by 31 bytes. The missing rows are at the
# bottom of the frame, outside the centre band this protocol keeps, so decoding
# it changes no metric - but set MAKE3D_STRICT=1 to refuse it anyway.
TRUNCATED_FLAG=(--make3d_allow_truncated)
[ "${MAKE3D_STRICT:-0}" = "1" ] && TRUNCATED_FLAG=()

echo "weights : $WEIGHTS"
echo "input   : ${WIDTH}x${HEIGHT}"
echo "make3d  : $DATA"
[ ${#EXTRA[@]} -gt 0 ] && echo "extra   : ${EXTRA[*]}"
echo

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
--height "$HEIGHT" \
"${TRUNCATED_FLAG[@]}" "${EXTRA[@]}"
