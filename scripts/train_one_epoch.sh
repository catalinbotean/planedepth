#!/usr/bin/env bash
# Train PlaneDepth for a single epoch on KITTI, with the stereo configuration
# of train_ResNet.sh, on one GPU.
#
# Everything about the model is identical to the paper's stage-1 recipe
# (--use_denseaspp --use_mixture_loss --plane_residual --flip_right); only
# --num_epochs is 1 and the launcher is plain `python` instead of torchrun,
# which trainer.py supports (it fills in RANK/WORLD_SIZE itself).
#
# Usage:
#   bash scripts/train_one_epoch.sh                          # ./kitti, name exp1_1ep
#   bash scripts/train_one_epoch.sh <data_path> <model_name> [extra flags...]
#
# Environment:
#   CUDA_VISIBLE_DEVICES   GPU to use (default 0)
#
# Note on batch size: trainer.py divides --batch_size by the number of visible
# GPUs and halves it again when --flip_right is set, so the default 8 becomes 4
# images per step here.

set -e

DATA=${1:-./kitti}
NAME=${2:-exp1_1ep}
[ $# -gt 2 ] && shift 2 || shift $#
EXTRA=("$@")

SPLIT=splits/eigen_full_left/train_files.txt
[ -f "$SPLIT" ] || { echo "Run this from the repo root: $SPLIT not found." >&2; exit 1; }
[ -d "$DATA" ] || { echo "No such data path: $DATA" >&2; exit 1; }

# ── pre-flight: does the split actually resolve against this data path? ───
# KITTIRAWDataset reads <data>/<folder>/image_0{2,3}/data/<frame:010d>.<ext>,
# and --png must match the extension on disk, so detect it rather than guess.
EXT=""
FIRST=$(head -1 "$SPLIT")
FOLDER=$(echo "$FIRST" | awk '{print $1}')
# 10# forces base 10: some splits pad the frame index with zeros, which bash
# would otherwise read as octal
FRAME=$(printf '%010d' "$((10#$(echo "$FIRST" | awk '{print $2}')))")
for candidate in png jpg; do
    if [ -f "$DATA/$FOLDER/image_02/data/$FRAME.$candidate" ]; then
        EXT=$candidate
        break
    fi
done
if [ -z "$EXT" ]; then
    echo "Could not find $DATA/$FOLDER/image_02/data/$FRAME.{png,jpg}" >&2
    echo "Point the first argument at the KITTI raw root (the folder holding" >&2
    echo "2011_09_26/, 2011_09_30/, ...), or download it first." >&2
    exit 1
fi

# stereo training needs the right camera too
[ -f "$DATA/$FOLDER/image_03/data/$FRAME.$EXT" ] || {
    echo "image_02 exists but image_03 does not - stereo training needs both" >&2
    exit 1
}

# how much of the split is actually on disk (a partial download trains, then
# dies thousands of steps in)
MISSING=0
CHECKED=0
while read -r folder frame side; do
    [ -n "$folder" ] || continue
    f=$(printf '%010d' "$((10#$frame))")
    [ -f "$DATA/$folder/image_02/data/$f.$EXT" ] || MISSING=$((MISSING + 1))
    CHECKED=$((CHECKED + 1))
    [ $CHECKED -ge 200 ] && break
done < "$SPLIT"

PNG_FLAG=()
[ "$EXT" = "png" ] && PNG_FLAG=(--png)

WORKERS=${WORKERS:-$( (nproc 2>/dev/null || echo 8) )}
[ "$WORKERS" -gt 12 ] && WORKERS=12

echo "data     : $DATA  (image extension .$EXT)"
echo "split    : $(wc -l < "$SPLIT") training samples"
echo "sampled  : $MISSING of the first $CHECKED are missing on disk"
echo "model    : $NAME  ->  ./log/ResNet/$NAME"
echo "workers  : $WORKERS"
[ ${#EXTRA[@]} -gt 0 ] && echo "extra    : ${EXTRA[*]}"
echo

if [ "$MISSING" -gt 0 ]; then
    echo "WARNING: part of the split is not on disk. Training will crash when it" >&2
    echo "reaches a missing frame. Finish the download first." >&2
    echo >&2
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train.py \
"${PNG_FLAG[@]}" \
--data_path "$DATA" \
--model_name "$NAME" \
--use_denseaspp \
--use_mixture_loss \
--plane_residual \
--flip_right \
--num_epochs 1 \
--num_workers "$WORKERS" \
"${EXTRA[@]}"

echo
echo "-> Weights: ./log/ResNet/$NAME/last_models (and best_models after validation)"
echo "-> Config : ./log/ResNet/$NAME/opt.json"
echo "-> Cost   : bash scripts/bench_cost.sh ./log/ResNet/$NAME"
