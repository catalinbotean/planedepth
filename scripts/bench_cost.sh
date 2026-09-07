#!/usr/bin/env bash
# Measure inference cost - parameters, MACs, latency, FPS, peak memory - for a
# trained model, using the architecture flags that model was actually trained
# with.
#
# bench_cost.py accepts every training flag, so the model it builds is only the
# model you trained if you pass the same flags. Rather than retyping them, this
# reads them out of the opt.json that trainer.py writes next to the weights.
#
# Usage:
#   bash scripts/bench_cost.sh ./log/ResNet/exp1_1ep
#   bash scripts/bench_cost.sh ./log/ResNet/exp1_1ep 640x192 1280x384
#
# With no log folder it benchmarks the paper's stage-1 configuration.

set -e

LOG_DIR=${1:-}
shift || true
RESOLUTIONS=("$@")
[ ${#RESOLUTIONS[@]} -eq 0 ] && RESOLUTIONS=(640x192 1280x384)

python - <<'PY' || { echo; echo "Install a MAC counter first:  pip install fvcore" >&2; echo "(thop also works). Latency is still reported without one." >&2; }
try:
    import fvcore  # noqa: F401
except ImportError:
    try:
        import thop  # noqa: F401
    except ImportError:
        raise SystemExit(1)
PY

FLAGS=()
if [ -n "$LOG_DIR" ]; then
    OPTS="$LOG_DIR/opt.json"
    [ -f "$OPTS" ] || OPTS="$(dirname "$LOG_DIR")/opt.json"
    [ -f "$OPTS" ] || { echo "No opt.json in $LOG_DIR or its parent." >&2; exit 1; }
    echo "-> architecture flags from $OPTS"
    # store/true flags are emitted only when set; the rest as --key value
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
else
    echo "-> no log folder given, benchmarking the stage-1 configuration"
    FLAGS=(--use_denseaspp --use_mixture_loss --plane_residual)
fi

echo "-> flags: ${FLAGS[*]}"
echo

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python bench_cost.py \
"${FLAGS[@]}" \
--bench_res "${RESOLUTIONS[@]}" \
--bench_iters "${BENCH_ITERS:-100}" \
--bench_warmup "${BENCH_WARMUP:-20}"
