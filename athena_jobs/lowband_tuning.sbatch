#!/bin/bash -l
#SBATCH --job-name=lb_tune
#SBATCH --account=plgadmmpinn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn/jobs/logs/%x-%j.out

# Step 1 of the practical-objective fix, on one GPU:
#   1. tuning grid on the SECOND crop (U-Net, seed 42): input prior rw/tv x
#      low-band anchor weight {0, 0.3, 1, 3} x sigma {12, 24};
#   2. the selected configuration (highest mean over clean/noisy x none/wells)
#      confirmed on the FAULT crop: both networks, dense/none/two wells, 5 seeds;
#   3. the fault-crop report.

set -eo pipefail
ROOT=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn
PY=$ROOT/env/py311_torch210/bin/python
CODE=$ROOT/code
RUNNER="$PY $CODE/run_marmousi_full.py"
OUT=$ROOT/runs/lowband_tuning
export OMP_NUM_THREADS=4
echo "job $SLURM_JOB_ID on $(hostname), started $(date -Is)"

$PY "$CODE/tune_second_crop.py" "$ROOT/runs/crop_second_e3" "$OUT"

read -r PRIOR LOWBAND SIGMA <<< "$($PY - <<PYEOF
import json, numpy as np
rows = json.load(open("$OUT/tuning_second_crop.json"))
cfgs = sorted({(r["prior"], r["lowband"], r["sigma"]) for r in rows})
best = max(cfgs, key=lambda c: np.mean([r["snr"] for r in rows if (r["prior"], r["lowband"], r["sigma"]) == c]))
print(best[0], best[1], best[2])
PYEOF
)"
echo "selected on the second crop: prior $PRIOR, low-band weight $LOWBAND, sigma $SIGMA"

# fault crop, selected configuration, five seeds, both networks (the baseline exists)
$RUNNER sweep --out "$ROOT/runs/crop_fault_e3" --networks unet rescnn --datas clean noisy \
    --supervisions dense none wells --well-traces 8160 8440 --seeds 42 123 7 2024 2025 \
    --train-set all --epochs 80 --transfer-dir "$ROOT/transfer" --device cuda --gpus 1 --per-gpu 4 \
    --threads 4 --prior "$PRIOR" --lowband "$LOWBAND" --lowband-sigma "$SIGMA"
# the same for the second crop (its own numbers at five seeds, for the table)
$RUNNER sweep --out "$ROOT/runs/crop_second_e3" --networks unet rescnn --datas clean noisy \
    --supervisions dense none wells --well-traces 4560 4840 --seeds 42 123 7 2024 2025 \
    --train-set all --epochs 80 --transfer-dir "$ROOT/transfer" --device cuda --gpus 1 --per-gpu 4 \
    --threads 4 --prior "$PRIOR" --lowband "$LOWBAND" --lowband-sigma "$SIGMA"
OMP_NUM_THREADS=16 $RUNNER report --out "$ROOT/runs/crop_fault_e3"
OMP_NUM_THREADS=16 $RUNNER report --out "$ROOT/runs/crop_second_e3"
echo "job finished $(date -Is)"
