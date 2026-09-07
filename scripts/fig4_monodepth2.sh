#!/usr/bin/env bash
# Regenerate the four figure frames from a monodepth2-lineage baseline, on the
# same fixed depth scale as the PlaneDepth side, keeping the metric values.
#
# Copy this and infer_baseline.py into the baseline's repository:
#
#   cp scripts/fig4_monodepth2.sh scripts/infer_baseline.py /path/to/monodepth2/
#   cd /path/to/monodepth2
#   bash fig4_monodepth2.sh ./models/stereo_1024x320 ~/planedepth/kitti ~/planedepth/fig4
#
# Usage:
#   bash fig4_monodepth2.sh <weights> [kitti_root] [out_dir] [extra flags...]
#
# Environment:
#   ARCH          monodepth2 (default), litemono or tinydepth
#   FRAMES        space separated <drive>/<frame> pairs, overriding the defaults
#   CMAP          colormap (default turbo)
#   RANGE         "near far" in metres (default "3 80")
#
# A stereo checkpoint is assumed: its scale is the KITTI baseline, so the depth
# is comparable with the other methods. A mono checkpoint has an arbitrary scale
# and would need median scaling against ground truth, which this cannot do.

set -e

WEIGHTS=${1:?usage: bash fig4_monodepth2.sh <weights> [kitti_root] [out_dir]}
DATA=${2:-./kitti}
OUT=${3:-./fig4}
[ $# -gt 3 ] && shift 3 || shift $#
EXTRA=("$@")

ARCH=${ARCH:-monodepth2}
CMAP=${CMAP:-turbo}
RANGE=${RANGE:-3 80}
FRAMES=${FRAMES:-"2011_09_26_drive_0005_sync/0000000139 \
2011_09_26_drive_0009_sync/0000000066 \
2011_09_26_drive_0013_sync/0000000007 \
2011_09_26_drive_0009_sync/0000000241"}

[ -f infer_baseline.py ] || {
    echo "Run this from the baseline repository, with infer_baseline.py copied in." >&2
    exit 1
}
[ -f "$WEIGHTS/encoder.pth" ] || { echo "$WEIGHTS has no encoder.pth" >&2; exit 1; }
PY=${PYTHON:-$(command -v python || command -v python3)}

DIR="$OUT/$ARCH"
mkdir -p "$DIR"
echo "-> $ARCH -> $DIR"

for spec in $FRAMES; do
    drive=${spec%%/*}; frame=${spec##*/}
    date=${drive%%_drive*}
    img="$DATA/$date/$drive/image_02/data/$frame.png"
    [ -f "$img" ] || { echo "   missing $img" >&2; continue; }
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PY" infer_baseline.py \
        --arch "$ARCH" --weights "$WEIGHTS" --image_path "$img" \
        --post_process --cmap "$CMAP" --depth_range $RANGE --save_npy \
        --output_dir "$DIR" "${EXTRA[@]}" >/dev/null
    for suffix in _pred.png _panel.png _depth.npy; do
        [ -f "$DIR/$frame$suffix" ] && mv "$DIR/$frame$suffix" "$DIR/${drive}_${frame}$suffix"
    done
    echo "   ${drive}_${frame}"
done
rm -f "$DIR/overview.png"

echo
echo "-> done. Send $OUT back for the figure."
