#!/usr/bin/env bash
# Evaluate an ORIGINAL PlaneDepth checkpoint on KITTI and write the qualitative
# panels used by scripts/rank_scenes.py and scripts/build_fig.py.
#
# Only the flags the published models were trained with are passed, so the
# upstream state_dict loads exactly: every module added since publication
# allocates parameters only when its own flag is set.
#
# Usage:
#   bash eval_kitti_original.sh                                   # all defaults
#   bash eval_kitti_original.sh <weights_folder>
#   bash eval_kitti_original.sh <weights_folder> <width> <height> [kitti_root] [out_dir] [extra flags...]
#
# Defaults: 1280x384 (HRfinetune / self-distillation resolution; use 640 192 for
# stage1), ./kitti, and ./qual_kitti/planedepth as the panel folder.
#
# Environment:
#   CUDA_VISIBLE_DEVICES   GPU to use (default 0)
#   EVAL_SPLIT             eigen_raw (default) or eigen_improved - the improved
#                          ground truth is much denser and reads better in a
#                          figure
#   NO_PP=1                drop --post_process

set -e

WEIGHTS=${1:-}
WIDTH=${2:-1280}
HEIGHT=${3:-384}
DATA=${4:-./kitti}
OUT=${5:-./qual_kitti/planedepth}
[ $# -gt 5 ] && shift 5 || shift $#
EXTRA=("$@")

SPLIT=${EVAL_SPLIT:-eigen_raw}

[ -f evaluate_depth_scope.py ] || { echo "Run this from the repo root." >&2; exit 1; }

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

# ── pre-flight on the data ────────────────────────────────────────────────
[ -d "$DATA" ] || { echo "No such KITTI root: $DATA" >&2; exit 1; }

GT="splits/$SPLIT/gt_depths.npz"
if [ ! -f "$GT" ]; then
    echo "Missing $GT - export the ground truth first:" >&2
    if [ "$SPLIT" = "eigen_improved" ]; then
        echo "  python splits/eigen_improved/prepare_groundtruth.py --improved_path ./kitti_depth" >&2
    else
        echo "  python splits/eigen_raw/export_gt_depth.py --data_path $DATA" >&2
    fi
    exit 1
fi

# the evaluation loader reads .png frames; a jpg-only copy of KITTI silently
# fails on the first batch
FIRST=$(head -1 "splits/$SPLIT/test_files.txt")
FOLDER=$(echo "$FIRST" | awk '{print $1}')
# 10# forces base 10: the test splits pad the frame index with zeros, which
# bash would otherwise read as octal
FRAME=$(printf '%010d' "$((10#$(echo "$FIRST" | awk '{print $2}')))")
if [ ! -f "$DATA/$FOLDER/image_02/data/$FRAME.png" ]; then
    if [ -f "$DATA/$FOLDER/image_02/data/$FRAME.jpg" ]; then
        echo "$DATA holds .jpg frames, but the evaluation loader reads .png." >&2
    else
        echo "Could not find $DATA/$FOLDER/image_02/data/$FRAME.png" >&2
    fi
    exit 1
fi

PP_FLAG=(--post_process)
[ "${NO_PP:-0}" = "1" ] && PP_FLAG=()

echo "weights : $WEIGHTS"
echo "input   : ${WIDTH}x${HEIGHT}"
echo "kitti   : $DATA   (split $SPLIT)"
echo "panels  : $OUT"
echo "postproc: ${PP_FLAG[*]:-off}"
[ ${#EXTRA[@]} -gt 0 ] && echo "extra   : ${EXTRA[*]}"
echo

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python evaluate_depth_scope.py \
--eval_stereo \
--eval_split "$SPLIT" \
--data_path "$DATA" \
--load_weights_folder "$WEIGHTS" \
--models_to_load encoder depth \
--use_denseaspp \
--plane_residual \
--use_mixture_loss \
--batch_size 1 \
--width "$WIDTH" \
--height "$HEIGHT" \
--eval_out_dir "$OUT" \
"${PP_FLAG[@]}" "${EXTRA[@]}"

echo
echo "-> Panels in $OUT; run the proposed model into a sibling folder, then:"
echo "     python scripts/rank_scenes.py $(dirname "$OUT") scope_depth planedepth"
