#!/usr/bin/env bash
# Measure peak inference memory for the rows of the inference-cost table.
#
# Reviewer 2 asked for memory consumption by name. This is the one column that
# cannot be filled in from a shared bench_cost.py run: --bench_baseline keeps
# two architectures resident in the same process, and a run with
# --use_semantic_gate keeps SegFormer resident even while the depth branch
# alone is being timed. Both inflate the peak. So each configuration is
# measured in its OWN process here, with nothing else on the GPU.
#
# No data is needed. bench_cost.py feeds the network a random 1x3xHxW tensor
# and the matching normalised coordinate grid, and leaves the weights randomly
# initialised: memory, MACs and latency depend on the architecture and the
# input shape, not on the values in the tensors. No KITTI, no checkpoint.
#
# Usage:
#   bash scripts/bench_memory.sh                      # 3 configs x 2 resolutions
#   bash scripts/bench_memory.sh 1280x384             # one resolution
#   bash scripts/bench_memory.sh 640x192 1280x384
#
# Environment:
#   BENCH_ITERS   timed iterations per run (default 100, same as the cost table)
#   BENCH_WARMUP  warm-up iterations (default 20)
#   OURS_EXTRA    extra architecture flags appended to every configuration
#   OUT_DIR       where the raw logs are kept (default ./bench_memory_logs)

set -e

BASE="--use_denseaspp --use_mixture_loss --plane_residual"
PROPOSED="--yz_levels 16 --use_cross_plane_attn"

# label | flags   - the three architectures the cost table reports
CONFIGS=(
  "original dictionary|$BASE"
  "depth path|$BASE $PROPOSED"
  "full model|$BASE $PROPOSED --use_semantic_gate"
)

RESOLUTIONS=("$@")
[ ${#RESOLUTIONS[@]} -eq 0 ] && RESOLUTIONS=(640x192 1280x384)

OUT_DIR=${OUT_DIR:-./bench_memory_logs}
mkdir -p "$OUT_DIR"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("no CUDA device: peak memory is a GPU measurement, there is "
             "nothing to report on CPU")
print("-> device: {}".format(torch.cuda.get_device_name(0)))
PY

echo "-> random input, randomly initialised weights: no dataset, no checkpoint"
echo "-> one process per configuration, so nothing else is resident on the GPU"
echo

RESULTS="$OUT_DIR/summary.tsv"
printf 'config\tresolution\tparams_M\tMACs_G\tlatency_ms\tFPS\tpeak_alloc_MiB\tpeak_reserved_MiB\n' > "$RESULTS"

for res in "${RESOLUTIONS[@]}"; do
    for entry in "${CONFIGS[@]}"; do
        label=${entry%%|*}
        flags=${entry#*|}
        slug=$(echo "${label}_${res}" | tr ' ' '_')
        log="$OUT_DIR/$slug.log"

        echo "=== $label @ $res ==="
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python bench_cost.py \
            $flags ${OURS_EXTRA:-} \
            --bench_res "$res" \
            --bench_iters "${BENCH_ITERS:-100}" \
            --bench_warmup "${BENCH_WARMUP:-20}" 2>&1 | tee "$log"
        echo

        python - "$log" "$label" "$res" >> "$RESULTS" <<'PY'
import re, sys

log, label, res = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(log, encoding="utf-8", errors="replace").read()

def grab(pattern):
    m = re.search(pattern, text, re.M)
    return float(m.group(1)) if m else float("nan")

# "  TOTAL                      42.868   (trainable   39.149)"
params   = grab(r"^\s*TOTAL\s+([\d.]+)")
macs     = grab(r"MACs\s*:\s*([\d.]+)\s*G")
latency  = grab(r"latency\s*:\s*([\d.]+)\s*ms")
fps      = grab(r"\(([\d.]+)\s*FPS")
alloc    = grab(r"peak memory\s*:\s*([\d.]+)")
reserved = grab(r"peak reserved\s*:\s*([\d.]+)")

print("\t".join([label, res,
                 "%.3f" % params, "%.2f" % macs, "%.2f" % latency,
                 "%.1f" % fps, "%.1f" % alloc, "%.1f" % reserved]))
PY
    done
done

echo
echo "=== summary (peak memory in MiB, batch 1, fp32) ==="
column -t -s $'\t' "$RESULTS" 2>/dev/null || cat "$RESULTS"
echo
echo "-> raw logs in $OUT_DIR/, summary in $RESULTS"
echo "-> 'allocated' is the tensor high-water mark and is the figure to report;"
echo "   'reserved' is the caching allocator's pool and is what nvidia-smi shows."
