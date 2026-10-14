#!/bin/bash -l
#SBATCH --job-name=poseidon_score
#SBATCH --account=plgadmmpinn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn/jobs/logs/%x-%j.out
# Phase B again with the declared own-lag sensitivity (phase A untouched; seals re-checked), then the report.
set -eo pipefail
ROOT=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn
PY=$ROOT/env/py311_torch210/bin/python
RUNNER="$PY $ROOT/code/poseidon/poseidon_runner.py"
PREP=$ROOT/field/poseidon/prepared
RUN=$ROOT/runs/${RUN_NAME:-poseidon_v1}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for c in "Proteus 1" "Pharos 1" "Poseidon 2" "Kronos 1" "Torosa 1"; do
    $RUNNER score --prepared "$PREP" --out "$RUN" --case "$c"
done
$RUNNER report --prepared "$PREP" --out "$RUN"
echo "job finished $(date -Is)"
