#!/usr/bin/env bash
# One command, end to end: evaluate PlaneDepth and SCOPE-Depth on KITTI, write
# the qualitative panels, pick the scenes by measured agreement with the ground
# truth, and assemble the figure.
#
# Usage:
#   bash run_kitti_figure.sh <scope_weights> [planedepth_weights] [kitti_root] [out_dir] [figure]
#
# To leave it running after you close the terminal:
#   nohup bash run_kitti_figure.sh <scope_weights> > kitti_figure.log 2>&1 &
#   tail -f kitti_figure.log
#
# Steps whose output already exists are skipped, so a re-run only redoes what is
# missing; FORCE=1 redoes everything.
#
# Environment:
#   SCOPE_FLAGS   architecture flags of the proposed model (edit to match the
#                 configuration you are writing up)
#   EVAL_SPLIT    eigen_raw (default) or eigen_improved - the improved ground
#                 truth is denser and reads better in a figure
#   WIDTH/HEIGHT  evaluation resolution (default 1280x384)
#   SCENES        how many scenes go into the figure (default 4)
#   NO_PP=1       drop --post_process from both evaluations
#   FORCE=1       recompute panels even if they are already there
#   CUDA_VISIBLE_DEVICES   GPU to use (default 0)

set -e

SCOPE_W=${1:?usage: bash run_kitti_figure.sh <scope_weights> [planedepth_weights] [kitti_root] [out_dir] [figure]}
PLANE_W=${2:-}
DATA=${3:-./kitti}
OUT=${4:-./qual_kitti}
FIG=${5:-./figures/fig_kitti_qualitative.png}

SPLIT=${EVAL_SPLIT:-eigen_raw}
WIDTH=${WIDTH:-1280}
HEIGHT=${HEIGHT:-384}
SCENES=${SCENES:-4}
SCOPE_FLAGS=${SCOPE_FLAGS:---use_denseaspp --plane_residual --use_mixture_loss --yz_levels 16 --use_semantic_gate --use_cross_plane_attn}
PLANE_FLAGS="--use_denseaspp --plane_residual --use_mixture_loss"

[ -f evaluate_depth_scope.py ] || { echo "Run this from the repo root." >&2; exit 1; }

# conda environments provide `python`; some systems only have `python3`
PY=${PYTHON:-$(command -v python || command -v python3)}
[ -n "$PY" ] || { echo "No python interpreter on PATH." >&2; exit 1; }
mkdir -p "$OUT" "$(dirname "$FIG")"
LOG="$OUT/run.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%Y-%m-%d %H:%M:%S')  kitti figure run ==="

PP_FLAG=(--post_process)
[ "${NO_PP:-0}" = "1" ] && PP_FLAG=()

evaluate() {   # name weights flags
    local name="$1" weights="$2"; shift 2
    local dir="$OUT/$name"
    if [ "${FORCE:-0}" != "1" ] && [ -n "$(ls "$dir"/*_panel.png 2>/dev/null)" ]; then
        echo "-> $name: $(ls "$dir"/*_panel.png | wc -l | tr -d ' ') panels already in $dir, skipping"
        return
    fi
    echo "-> $name: evaluating into $dir"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PY" evaluate_depth_scope.py \
        --eval_stereo --eval_split "$SPLIT" --data_path "$DATA" \
        --load_weights_folder "$weights" --models_to_load encoder depth \
        --batch_size 1 --width "$WIDTH" --height "$HEIGHT" \
        --eval_out_dir "$dir" "${PP_FLAG[@]}" "$@"
}

# shellcheck disable=SC2086
evaluate scope_depth "$SCOPE_W" $SCOPE_FLAGS

MODELS=(scope_depth)
if [ -n "$PLANE_W" ]; then
    # shellcheck disable=SC2086
    evaluate planedepth "$PLANE_W" $PLANE_FLAGS
    MODELS+=(planedepth)
else
    echo "-> no PlaneDepth weights given; the figure will hold the proposed model only"
fi
for extra in monodepth2 litemono tinydepth; do
    [ -n "$(ls "$OUT/$extra"/*_panel.png 2>/dev/null)" ] && MODELS+=("$extra")
done

if [ ${#MODELS[@]} -lt 2 ]; then
    echo "Only one model has panels, so there is nothing to rank against." >&2
    echo "Pass PlaneDepth weights as the second argument." >&2
    exit 1
fi

echo
echo "-> ranking scenes: ${MODELS[*]}"
# one pass: the table is printed and the chosen scenes are written out
"$PY" scripts/rank_scenes.py "$OUT" "${MODELS[@]}" --top 10 \
    --emit_file "$OUT/scenes.txt" | tee "$OUT/ranking.txt"
head -n "$SCENES" "$OUT/scenes.txt" > "$OUT/scenes.head" || true

SPECS=()
while IFS= read -r line; do
    [ -n "$line" ] && SPECS+=(--scene "$line")
done < "$OUT/scenes.head"
rm -f "$OUT/scenes.head"

if [ ${#SPECS[@]} -eq 0 ]; then
    echo "No scene beats every baseline - nothing to put in a figure." >&2
    exit 1
fi

# build_fig.py draws the proposed model last, under the baselines
ORDER=()
for m in "${MODELS[@]:1}"; do ORDER+=("$m"); done
ORDER+=("${MODELS[0]}")
LABELS=$(printf '%s,' "${ORDER[@]}" | sed 's/,$//; s/scope_depth/SCOPE-Depth (ours)/; s/planedepth/PlaneDepth/; s/monodepth2/Monodepth2/; s/litemono/Lite-Mono/; s/tinydepth/TinyDepth/')

echo
echo "-> assembling $FIG"
"$PY" scripts/build_fig.py "$OUT" "$FIG" \
    --models "$(printf '%s,' "${ORDER[@]}" | sed 's/,$//')" \
    --labels "$LABELS" "${SPECS[@]}"

echo
echo "-> done: $FIG"
echo "   panels   : $OUT"
echo "   ranking  : $OUT/ranking.txt"
echo "   log      : $LOG"
