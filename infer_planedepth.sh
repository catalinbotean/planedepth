#!/usr/bin/env bash
# Run a PlaneDepth checkpoint over KITTI images and write the coloured depth
# maps. Inference only: no ground truth, no metrics, no velodyne.
#
# Usage:
#   bash infer_planedepth.sh                          # everything by default
#   bash infer_planedepth.sh <weights> [images] [out_dir] [width] [height] [extra flags...]
#
# Defaults: the only checkpoint it can find, ./kitti as the images, panels into
# ./qual_kitti/planedepth, at 1280x384.
#
#   IMAGES may be a folder (searched recursively), a single image, or a split
#   list such as splits/eigen_raw/test_files.txt, in which case KITTI_ROOT says
#   where the drives live.
#
# Environment:
#   CUDA_VISIBLE_DEVICES   GPU to use (default 0)
#   KITTI_ROOT             drives root when IMAGES is a split list (default ./kitti)
#   NO_PP=1                drop --post_process

set -e

WEIGHTS=${1:-}
IMAGES=${2:-./kitti}
OUT=${3:-./qual_kitti/planedepth}
WIDTH=${4:-1280}
HEIGHT=${5:-384}
[ $# -gt 5 ] && shift 5 || shift $#
EXTRA=("$@")

[ -f infer.py ] || { echo "Run this from the repo root." >&2; exit 1; }

PY=${PYTHON:-$(command -v python || command -v python3)}
[ -n "$PY" ] || { echo "No python interpreter on PATH." >&2; exit 1; }

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
for f in encoder.pth depth.pth; do
    [ -f "$WEIGHTS/$f" ] || { echo "$WEIGHTS does not contain $f." >&2; exit 1; }
done
[ -e "$IMAGES" ] || { echo "No such image, folder or list: $IMAGES" >&2; exit 1; }

PP_FLAG=(--post_process)
[ "${NO_PP:-0}" = "1" ] && PP_FLAG=()

echo "weights : $WEIGHTS"
echo "images  : $IMAGES"
echo "output  : $OUT"
echo "input   : ${WIDTH}x${HEIGHT}"
echo "postproc: ${PP_FLAG[*]:-off}"
[ ${#EXTRA[@]} -gt 0 ] && echo "extra   : ${EXTRA[*]}"
echo

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PY" infer.py \
--image_path "$IMAGES" \
--data_path "${KITTI_ROOT:-./kitti}" \
--load_weights_folder "$WEIGHTS" \
--use_denseaspp \
--plane_residual \
--use_mixture_loss \
--width "$WIDTH" \
--height "$HEIGHT" \
--output_dir "$OUT" \
"${PP_FLAG[@]}" "${EXTRA[@]}"
