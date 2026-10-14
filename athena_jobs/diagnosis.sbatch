#!/bin/bash -l
#SBATCH --job-name=diagnosis
#SBATCH --account=plgadmmpinn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn/jobs/logs/%x-%j.out

# Bounded diagnosis of the label-free objective on the development crop
# (second crop), and the well-conditioned classical comparator on both crops.
set -eo pipefail
ROOT=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn
PY=$ROOT/env/py311_torch210/bin/python
CODE=$ROOT/code
OUT=$ROOT/runs/diagnosis
export OMP_NUM_THREADS=8
echo "job $SLURM_JOB_ID on $(hostname), started $(date -Is)"
mkdir -p "$OUT"

for prior in rw tv; do
    for data in noisy clean; do
        echo "=== diagnosis: second crop, prior $prior, $data  $(date -Is)"
        $PY "$CODE/diagnose_objective.py" --run "$ROOT/runs/crop_second_e3" --out "$OUT" \
            --data "$data" --prior "$prior" --network unet --epochs 80 --transfer-dir "$ROOT/transfer"
    done
done

# well-conditioned 2-D classical comparator: three penalty weights on the development
# crop, the same three on the fault crop for reference (the weight is chosen on the
# development crop only)
for run in crop_second_e3 crop_fault_e3; do
    case $run in crop_second_e3) W="4560 4840";; crop_fault_e3) W="8160 8440";; esac
    for weight in 0.1 1 10; do
        echo "=== well-conditioned baseline: $run, weight $weight  $(date -Is)"
        OMP_NUM_THREADS=16 $PY "$CODE/run_marmousi_full.py" baseline --out "$ROOT/runs/$run" --n-jobs 16 \
            --wells --well-traces $W --well-weight "$weight"
        mv "$ROOT/runs/$run/baseline_wells2" "$ROOT/runs/$run/baseline_wells2_w$weight"
    done
done
echo "job finished $(date -Is)"
