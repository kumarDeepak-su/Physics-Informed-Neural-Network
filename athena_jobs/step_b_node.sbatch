#!/bin/bash -l
#SBATCH --job-name=step_b
#SBATCH --account=plgadmmpinn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=06:00:00
#SBATCH --output=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn/jobs/logs/%x-%j.out

# Step B of the revision on one whole node: the three supervision variants
# (dense labels, label-free, pseudo-wells) x 5 seeds x U-Net/ResCNN x
# clean/noisy, with the reweighting floor epsilon = 1e-3, on
#   1. the two crop windows of the paper (400 x 300 each, run through the
#      same pipeline: the forward model, noise, starting model and patch grid
#      of the crop are those of the published crop experiments), and
#   2. the complete model (only the pseudo-well variant is new; the finished
#      dense and label-free trainings of RUN_FULL are reused).
# A check run reproduces the published fault-crop protocol (epsilon = 1e-4,
# dense labels, seed 42) through the pipeline for comparison with M1.
#
#   RUN_FULL   existing full-model run folder            (default full_all_e3_fault)
#   SEEDS      space-separated                           (default "42 123 7 2024 2025")
#   EPSILON    reweighting floor                         (default 1e-3)

set -eo pipefail

ROOT=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn
PY=$ROOT/env/py311_torch210/bin/python
CODE=$ROOT/code
RUN_FULL=$ROOT/runs/${RUN_FULL:-full_all_e3_fault}
SEEDS=${SEEDS:-"42 123 7 2024 2025"}
EPSILON=${EPSILON:-1e-3}
NGPU=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
NCPU=$SLURM_CPUS_PER_TASK
RUNNER="$PY $CODE/run_marmousi_full.py"

echo "job $SLURM_JOB_ID on $(hostname): $NGPU GPUs, $NCPU cores, started $(date -Is)"
"$PY" -c "import threadpoolctl, torch; assert torch.cuda.is_available(), 'no CUDA'"

# ---- 1. crops: fault window 8100:8500 x 1500:1800, second window 4500:4900 x 1000:1300
crop_run () {   # $1 = run name, $2 = trace range, $3 = sample range, $4 = epsilon, $5 = supervisions, $6 = seeds, $7 = wells
    local run=$ROOT/runs/$1
    if [ ! -f "$run/prepared/prepared.json" ]; then
        OMP_NUM_THREADS=$NCPU $RUNNER prepare --out "$run" --marmousi-dir "$ROOT/data/marmousi2" \
            --trace-range "$2" --sample-range "$3" --epsilon "$4" --n-jobs 32
    fi
    [ -f "$run/baseline/baseline.json" ] || OMP_NUM_THREADS=$NCPU $RUNNER baseline --out "$run" --n-jobs 32
    OMP_NUM_THREADS=2 $RUNNER sweep --out "$run" --networks unet rescnn --datas clean noisy \
        --supervisions $5 --seeds $6 --train-set all --epochs 80 --transfer-dir "$ROOT/transfer" \
        --device cuda --gpus "$NGPU" --per-gpu 4 --threads 2 --well-traces $7
    OMP_NUM_THREADS=$NCPU $RUNNER report --out "$run"
}
# published-protocol check (epsilon 1e-4, dense, seed 42): to be compared with M1 / the paper
crop_run crop_fault_e4_check 8100:8500 1500:1800 1e-4 "dense" "42" "8160 8440"
# Step B proper: wells at traces 60 and 340 of each crop (two-well layout)
crop_run crop_fault_e3 8100:8500 1500:1800 "$EPSILON" "dense none wells" "$SEEDS" "8160 8440"
crop_run crop_second_e3 4500:4900 1000:1300 "$EPSILON" "dense none wells" "$SEEDS" "4560 4840"

# ---- 2. complete model: pseudo-well variant added to the existing run
OMP_NUM_THREADS=2 $RUNNER sweep --out "$RUN_FULL" --networks unet rescnn --datas clean noisy \
    --supervisions wells --seeds $SEEDS --train-set all --epochs 80 --transfer-dir "$ROOT/transfer" \
    --device cuda --gpus "$NGPU" --per-gpu 2 --threads 2
OMP_NUM_THREADS=$NCPU $RUNNER report --out "$RUN_FULL"

echo "job finished $(date -Is)"
