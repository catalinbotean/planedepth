#!/usr/bin/env bash
# Measure inference cost - parameters, MACs, FLOPs, latency, FPS, peak memory.
#
# No trained weights are needed: bench_cost.py builds the network from the
# architecture flags and leaves it randomly initialised, because parameter
# count, MACs and latency depend on the architecture alone - not on the values
# in the tensors. Nothing to train, no KITTI download.
#
# Usage:
#   bash scripts/bench_cost.sh                     # ours vs PlaneDepth baseline
#   bash scripts/bench_cost.sh ours 640x192        # ours, one resolution
#   bash scripts/bench_cost.sh baseline            # plain PlaneDepth only
#   bash scripts/bench_cost.sh ./log/ResNet/exp1   # flags from that model's opt.json
#
# The default "ours" run passes --bench_baseline (so the same script also
# measures PlaneDepth with every proposed module switched off, on this GPU, in
# this process) and --bench_no_sem (so the depth branch is timed separately
# from the frozen SegFormer).
#
# Environment:
#   OURS_FLAGS    override the flag set below, e.g.
#                 OURS_FLAGS="--yz_levels 16 --use_cross_plane_attn" bash scripts/bench_cost.sh
#   BENCH_ITERS   timed iterations per configuration (default 100)
#   BENCH_WARMUP  warm-up iterations (default 20)

set -e

# ---------------------------------------------------------------------------
# The proposed model. EDIT THIS to match the configuration you are writing up -
# it is the one place the flag set lives, and --bench_baseline switches every
# one of these off to produce the comparison row.
# ---------------------------------------------------------------------------
OURS_DEFAULT="--use_denseaspp --use_mixture_loss --plane_residual \
--yz_levels 16 --use_semantic_gate --use_cross_plane_attn"

# PlaneDepth stage 1, i.e. train_ResNet.sh without any of the additions.
BASELINE_FLAGS="--use_denseaspp --use_mixture_loss --plane_residual"

TARGET=${1:-ours}
shift || true
RESOLUTIONS=("$@")
[ ${#RESOLUTIONS[@]} -eq 0 ] && RESOLUTIONS=(640x192 1280x384)

python - <<'PY' || { echo "-> no MAC counter installed: pip install fvcore" >&2; echo "   (latency, FPS and parameters are still reported)" >&2; }
try:
    import fvcore  # noqa: F401
except ImportError:
    try:
        import thop  # noqa: F401
    except ImportError:
        raise SystemExit(1)
PY

FLAGS=()
case "$TARGET" in
    ours)
        # shellcheck disable=SC2206
        FLAGS=(${OURS_FLAGS:-$OURS_DEFAULT} --bench_baseline --bench_no_sem)
        echo "-> proposed model, with the PlaneDepth baseline measured alongside"
        ;;
    baseline)
        # shellcheck disable=SC2206
        FLAGS=($BASELINE_FLAGS)
        echo "-> PlaneDepth stage-1 baseline"
        ;;
    *)
        OPTS="$TARGET/opt.json"
        [ -f "$OPTS" ] || OPTS="$(dirname "$TARGET")/opt.json"
        [ -f "$OPTS" ] || {
            echo "'$TARGET' is neither 'ours', 'baseline', nor a folder with opt.json" >&2
            exit 1
        }
        echo "-> architecture flags from $OPTS"
        # store_true flags are emitted only when set; the rest as --key value
        while IFS= read -r line; do FLAGS+=("$line"); done < <(python - "$OPTS" <<'PY'
import json, sys

opt = json.load(open(sys.argv[1]))
BOOL = ["use_denseaspp", "use_mixture_loss", "plane_residual",
        "pixelwise_plane_residual", "render_probability",
        "use_cross_plane_attn", "adaptive_plane_range",
        "use_multiscale_logits", "use_semantic_gate"]
VALUE = ["net_type", "num_layers", "disp_levels", "disp_min", "disp_max",
         "xz_levels", "yz_levels", "num_ep", "pe_type",
         "cross_plane_attn_tau", "adaptive_range_margin",
         "num_learned_families", "learned_planes_per_family",
         "segformer_model", "semantic_num_classes"]

for key in BOOL:
    if opt.get(key):
        print("--" + key)
for key in VALUE:
    if key in opt and opt[key] is not None:
        print("--" + key)
        print(str(opt[key]))
PY
)
        FLAGS+=(--bench_baseline --bench_no_sem)
        ;;
esac

echo "-> weights are randomly initialised: cost depends on the architecture only"
echo "-> flags: ${FLAGS[*]}"
echo

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python bench_cost.py \
"${FLAGS[@]}" \
--bench_res "${RESOLUTIONS[@]}" \
--bench_iters "${BENCH_ITERS:-100}" \
--bench_warmup "${BENCH_WARMUP:-20}"
