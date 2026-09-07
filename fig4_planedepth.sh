#!/usr/bin/env bash
# Regenerate the four figure frames from this repository, on a fixed depth
# scale and keeping the metric values - what the reviewer's "common depth scale
# and ground-truth error maps" needs.
#
# Usage:
#   bash fig4_planedepth.sh <planedepth_weights> [scope_weights] [kitti_root] [out_dir]
#
# With a second weights folder it also runs the proposed model, into a sibling
# folder, so both rows of the figure come out of one command.
#
# Environment:
#   FRAMES        space separated <drive>/<frame> pairs, overriding the defaults
#   SCOPE_FLAGS   architecture flags of the proposed model
#   CMAP          colormap (default turbo)
#   RANGE         "near far" in metres (default "3 80")
#   WIDTH/HEIGHT  network input (default 1280x384)

set -e

PLANE_W=${1:?usage: bash fig4_planedepth.sh <planedepth_weights> [scope_weights] [kitti_root] [out_dir]}
SCOPE_W=${2:-}
DATA=${3:-./kitti}
OUT=${4:-./fig4}

CMAP=${CMAP:-turbo}
RANGE=${RANGE:-3 80}
WIDTH=${WIDTH:-1280}
HEIGHT=${HEIGHT:-384}
PLANE_FLAGS="--use_denseaspp --plane_residual --use_mixture_loss"
SCOPE_FLAGS=${SCOPE_FLAGS:---use_denseaspp --plane_residual --use_mixture_loss --yz_levels 16 --use_semantic_gate --use_cross_plane_attn}

FRAMES=${FRAMES:-"2011_09_26_drive_0005_sync/0000000139 \
2011_09_26_drive_0009_sync/0000000066 \
2011_09_26_drive_0013_sync/0000000007 \
2011_09_26_drive_0009_sync/0000000241"}

[ -f infer.py ] || { echo "Run this from the repo root." >&2; exit 1; }
PY=${PYTHON:-$(command -v python || command -v python3)}

run_model() {   # name weights flags
    local name="$1" weights="$2"; shift 2
    for f in encoder.pth depth.pth; do
        [ -f "$weights/$f" ] || { echo "$weights does not contain $f." >&2; exit 1; }
    done
    local dir="$OUT/$name"
    mkdir -p "$dir"
    echo "-> $name -> $dir"
    for spec in $FRAMES; do
        drive=${spec%%/*}; frame=${spec##*/}
        date=${drive%%_drive*}
        img="$DATA/$date/$drive/image_02/data/$frame.png"
        [ -f "$img" ] || { echo "   missing $img" >&2; continue; }
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PY" infer.py \
            --image_path "$img" \
            --load_weights_folder "$weights" \
            --width "$WIDTH" --height "$HEIGHT" \
            --post_process --cmap "$CMAP" --depth_range $RANGE --save_npy \
            --output_dir "$dir" "$@" >/dev/null
        # infer.py names a lone image by its frame number; make it unique
        for suffix in _pred.png _panel.png _depth.npy; do
            [ -f "$dir/$frame$suffix" ] && mv "$dir/$frame$suffix" "$dir/${drive}_${frame}$suffix"
        done
        echo "   ${drive}_${frame}"
    done
    rm -f "$dir/overview.png"
}

# shellcheck disable=SC2086
run_model planedepth "$PLANE_W" $PLANE_FLAGS
if [ -n "$SCOPE_W" ]; then
    # shellcheck disable=SC2086
    run_model scopedepth "$SCOPE_W" $SCOPE_FLAGS
fi

echo
echo "-> done. Send $OUT (a few MB) back for the figure."
