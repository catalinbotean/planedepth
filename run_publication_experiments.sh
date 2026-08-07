#!/usr/bin/env bash
set -Eeuo pipefail

# Publication-grade SG-PlaneDepth experiment matrix.
# Run from the repository root:
#   DATA_PATH=/path/to/kitti NPROC_PER_NODE=4 bash run_publication_experiments.sh
# Optional overrides:
#   REPO_DIR, RUN_ROOT, CUDA_VISIBLE_DEVICES, NPROC_PER_NODE,
#   STAGE1_BATCH, HR_BATCH, SD_BATCH, NUM_WORKERS, SEEDS

REPO_DIR="${REPO_DIR:-$(pwd)}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to the KITTI root directory}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/publication_runs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
STAGE1_BATCH="${STAGE1_BATCH:-32}"
HR_BATCH="${HR_BATCH:-4}"
SD_BATCH="${SD_BATCH:-4}"
NUM_WORKERS="${NUM_WORKERS:-12}"
SEEDS="${SEEDS:-1 2 3}"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export RUN_ROOT

cd "${REPO_DIR}"
mkdir -p "${RUN_ROOT}/command_logs" "${RUN_ROOT}/evaluations"

git rev-parse HEAD > "${RUN_ROOT}/git_commit.txt"
git diff > "${RUN_ROOT}/uncommitted_source.patch"
python --version > "${RUN_ROOT}/python_version.txt" 2>&1
python -m pip freeze > "${RUN_ROOT}/pip_freeze.txt"
nvidia-smi > "${RUN_ROOT}/nvidia_smi.txt"

for required in train.py evaluate_depth_HR.py options.py trainer.py; do
  test -f "${required}" || { echo "Missing ${REPO_DIR}/${required}" >&2; exit 1; }
done

grep -q -- '--seed' options.py || {
  echo "This runner requires the publication branch with --seed support." >&2
  exit 1
}
grep -q 'semantic_gate.pth' evaluate_depth_HR.py || {
  echo "evaluate_depth_HR.py cannot yet load semantic checkpoints." >&2
  exit 1
}
test -f splits/eigen_raw/gt_depths.npz || {
  echo "Missing splits/eigen_raw/gt_depths.npz. Run the PlaneDepth GT export first." >&2
  exit 1
}

# Cache the frozen semantic model once before launching multi-GPU jobs.
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/segformer-b0-finetuned-cityscapes-512-1024')"

COMMON=(
  --data_path "${DATA_PATH}"
  --log_dir "${RUN_ROOT}"
  --png
  --split eigen_full_left
  --net_type ResNet
  --num_layers 50
  --use_denseaspp
  --use_mixture_loss
  --plane_residual
  --flip_right
  --xz_levels 14
  --yz_levels 16
  --disp_levels 49
  --num_workers "${NUM_WORKERS}"
)

run_train() {
  local name="$1"
  shift
  local checkpoint="${RUN_ROOT}/ResNet/${name}/best_models"
  local logfile="${RUN_ROOT}/command_logs/${name}.log"
  if test -f "${checkpoint}/encoder.pth" && test -f "${checkpoint}/depth.pth"; then
    echo "[skip train] ${name}"
    return
  fi
  echo "[train] ${name}"
  torchrun --nproc_per_node="${NPROC_PER_NODE}" train.py \
    "${COMMON[@]}" --model_name "${name}" "$@" 2>&1 | tee "${logfile}"
}

run_eval() {
  local name="$1"
  local width="$2"
  local height="$3"
  local yz_levels="$4"
  local use_gate="$5"
  local use_attn="$6"
  local checkpoint="${RUN_ROOT}/ResNet/${name}/best_models"
  local outfile="${RUN_ROOT}/evaluations/${name}.txt"
  local extra=()
  test "${use_gate}" = 1 && extra+=(--use_semantic_gate)
  test "${use_attn}" = 1 && extra+=(--use_cross_plane_attn)
  echo "[evaluate] ${name}"
  python evaluate_depth_HR.py \
    --data_path "${DATA_PATH}" \
    --load_weights_folder "${checkpoint}" \
    --eval_stereo --eval_split eigen_raw \
    --net_type ResNet --num_layers 50 \
    --use_denseaspp --use_mixture_loss --plane_residual \
    --xz_levels 14 --yz_levels "${yz_levels}" --disp_levels 49 \
    --width "${width}" --height "${height}" \
    --batch_size 1 --num_workers "${NUM_WORKERS}" \
    "${extra[@]}" 2>&1 | tee "${outfile}"
}

for seed in ${SEEDS}; do
  # Stage 1: clean, single-change ablation at 192x640.
  run_train "s1_yz0_seed${seed}" \
    --seed "${seed}" --yz_levels 0 --batch_size "${STAGE1_BATCH}" \
    --learning_rate 1e-4 --num_epochs 50 --milestones 30 40
  run_eval "s1_yz0_seed${seed}" 640 192 0 0 0

  run_train "s1_yz16_seed${seed}" \
    --seed "${seed}" --batch_size "${STAGE1_BATCH}" \
    --learning_rate 1e-4 --num_epochs 50 --milestones 30 40
  run_eval "s1_yz16_seed${seed}" 640 192 16 0 0

  run_train "s1_gate_yz16_seed${seed}" \
    --seed "${seed}" --batch_size "${STAGE1_BATCH}" \
    --learning_rate 1e-4 --num_epochs 50 --milestones 30 40 \
    --use_semantic_gate
  run_eval "s1_gate_yz16_seed${seed}" 640 192 16 1 0

  run_train "s1_attn_yz16_seed${seed}" \
    --seed "${seed}" --batch_size "${STAGE1_BATCH}" \
    --learning_rate 1e-4 --num_epochs 50 --milestones 30 40 \
    --use_cross_plane_attn
  run_eval "s1_attn_yz16_seed${seed}" 640 192 16 0 1

  run_train "s1_gate_attn_yz16_seed${seed}" \
    --seed "${seed}" --batch_size "${STAGE1_BATCH}" \
    --learning_rate 1e-4 --num_epochs 50 --milestones 30 40 \
    --use_semantic_gate --use_cross_plane_attn
  run_eval "s1_gate_attn_yz16_seed${seed}" 640 192 16 1 1

done

echo "All Stage 1 experiments are complete; starting high-resolution fine-tuning."
for seed in ${SEEDS}; do

  # High-resolution fine-tuning: identical one-epoch schedule for every core row.
  run_train "hr_yz0_seed${seed}" \
    --seed "${seed}" --yz_levels 0 --batch_size "${HR_BATCH}" \
    --learning_rate 2.5e-5 --num_epochs 1 --milestones 1 \
    --width 1280 --height 384 --no_crop \
    --load_weights_folder "${RUN_ROOT}/ResNet/s1_yz0_seed${seed}/best_models" \
    --models_to_load encoder depth
  run_eval "hr_yz0_seed${seed}" 1280 384 0 0 0

  run_train "hr_yz16_seed${seed}" \
    --seed "${seed}" --batch_size "${HR_BATCH}" \
    --learning_rate 2.5e-5 --num_epochs 1 --milestones 1 \
    --width 1280 --height 384 --no_crop \
    --load_weights_folder "${RUN_ROOT}/ResNet/s1_yz16_seed${seed}/best_models" \
    --models_to_load encoder depth
  run_eval "hr_yz16_seed${seed}" 1280 384 16 0 0

  run_train "hr_gate_yz16_seed${seed}" \
    --seed "${seed}" --batch_size "${HR_BATCH}" \
    --learning_rate 2.5e-5 --num_epochs 1 --milestones 1 \
    --width 1280 --height 384 --no_crop --use_semantic_gate \
    --load_weights_folder "${RUN_ROOT}/ResNet/s1_gate_yz16_seed${seed}/best_models" \
    --models_to_load encoder depth semantic_gate
  run_eval "hr_gate_yz16_seed${seed}" 1280 384 16 1 0

  run_train "hr_attn_yz16_seed${seed}" \
    --seed "${seed}" --batch_size "${HR_BATCH}" \
    --learning_rate 2.5e-5 --num_epochs 1 --milestones 1 \
    --width 1280 --height 384 --no_crop --use_cross_plane_attn \
    --load_weights_folder "${RUN_ROOT}/ResNet/s1_attn_yz16_seed${seed}/best_models" \
    --models_to_load encoder depth
  run_eval "hr_attn_yz16_seed${seed}" 1280 384 16 0 1

  run_train "hr_gate_attn_yz16_seed${seed}" \
    --seed "${seed}" --batch_size "${HR_BATCH}" \
    --learning_rate 2.5e-5 --num_epochs 1 --milestones 1 \
    --width 1280 --height 384 --no_crop \
    --use_semantic_gate --use_cross_plane_attn \
    --load_weights_folder "${RUN_ROOT}/ResNet/s1_gate_attn_yz16_seed${seed}/best_models" \
    --models_to_load encoder depth semantic_gate
  run_eval "hr_gate_attn_yz16_seed${seed}" 1280 384 16 1 1

  # Loss-level ablation: same stage-1 checkpoint, only semantic edge smoothness differs.
  run_train "hr_gate_attn_semsmooth_yz16_seed${seed}" \
    --seed "${seed}" --batch_size "${HR_BATCH}" \
    --learning_rate 2.5e-5 --num_epochs 1 --milestones 1 \
    --width 1280 --height 384 --no_crop \
    --use_semantic_gate --use_cross_plane_attn --semantic_edge_smoothness \
    --load_weights_folder "${RUN_ROOT}/ResNet/s1_gate_attn_yz16_seed${seed}/best_models" \
    --models_to_load encoder depth semantic_gate
  run_eval "hr_gate_attn_semsmooth_yz16_seed${seed}" 1280 384 16 1 1

done

echo "All high-resolution experiments are complete; starting self-distillation."
for seed in ${SEEDS}; do

  # Self-distillation controls: tuned geometry baseline, architecture only, full method.
  run_train "sd_yz16_seed${seed}" \
    --seed "${seed}" --batch_size "${SD_BATCH}" \
    --learning_rate 2e-5 --num_epochs 10 --milestones 5 \
    --width 1280 --height 384 --no_crop --self_distillation 1.0 \
    --load_weights_folder "${RUN_ROOT}/ResNet/hr_yz16_seed${seed}/best_models" \
    --models_to_load encoder depth
  run_eval "sd_yz16_seed${seed}" 1280 384 16 0 0

  run_train "sd_gate_attn_yz16_seed${seed}" \
    --seed "${seed}" --batch_size "${SD_BATCH}" \
    --learning_rate 2e-5 --num_epochs 10 --milestones 5 \
    --width 1280 --height 384 --no_crop --self_distillation 1.0 \
    --use_semantic_gate --use_cross_plane_attn \
    --load_weights_folder "${RUN_ROOT}/ResNet/hr_gate_attn_yz16_seed${seed}/best_models" \
    --models_to_load encoder depth semantic_gate
  run_eval "sd_gate_attn_yz16_seed${seed}" 1280 384 16 1 1

  run_train "sd_full_yz16_seed${seed}" \
    --seed "${seed}" --batch_size "${SD_BATCH}" \
    --learning_rate 2e-5 --num_epochs 10 --milestones 5 \
    --width 1280 --height 384 --no_crop --self_distillation 1.0 \
    --use_semantic_gate --use_cross_plane_attn --semantic_edge_smoothness \
    --uncertainty_weighted_distillation --semantic_distill_weight 1.0 \
    --load_weights_folder "${RUN_ROOT}/ResNet/hr_gate_attn_semsmooth_yz16_seed${seed}/best_models" \
    --models_to_load encoder depth semantic_gate
  run_eval "sd_full_yz16_seed${seed}" 1280 384 16 1 1
done

# Collect every standard evaluator result into one machine-readable table.
python - <<'PY'
import csv
import collections
import os
import pathlib
import re
import statistics

root = pathlib.Path(os.environ["RUN_ROOT"])
pattern = re.compile(
    r"&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*&\s*"
    r"([0-9.]+)\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)")
rows = []
for path in sorted((root / "evaluations").glob("*.txt")):
    matches = pattern.findall(path.read_text(errors="replace"))
    if not matches:
        raise RuntimeError(f"No metric row found in {path}")
    name = path.stem
    seed_match = re.search(r"_seed(\d+)$", name)
    stage = name.split("_", 1)[0]
    rows.append([name, stage, int(seed_match.group(1)), *map(float, matches[-1])])

out = root / "results_summary.csv"
with out.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["experiment", "stage", "seed", "abs_rel", "sq_rel",
                     "rmse", "rmse_log", "a1", "a2", "a3"])
    writer.writerows(rows)
print(f"Wrote {out} with {len(rows)} evaluated checkpoints")

grouped = collections.defaultdict(list)
for row in rows:
    base = re.sub(r"_seed\d+$", "", row[0])
    grouped[base].append(row[3:])
aggregate = root / "results_mean_std.csv"
with aggregate.open("w", newline="") as f:
    writer = csv.writer(f)
    header = ["experiment", "n"]
    for metric in ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]:
        header.extend([f"{metric}_mean", f"{metric}_std"])
    writer.writerow(header)
    for name, values in sorted(grouped.items()):
        columns = list(zip(*values))
        summary = []
        for column in columns:
            summary.extend([statistics.fmean(column), statistics.stdev(column)
                            if len(column) > 1 else 0.0])
        writer.writerow([name, len(values), *summary])
print(f"Wrote {aggregate} with mean/std across seeds")
PY

echo "All experiments completed. Summary: ${RUN_ROOT}/results_summary.csv"
