#!/bin/bash -l
#SBATCH --job-name=sm_sweep_tv
#SBATCH --account=plgadmmpinn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=10:00:00
#SBATCH --output=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn/jobs/logs/%x-%j.out

# Starting-model sensitivity of the practical variant (five simulated wells,
# 2-D coupled result as the network input, no low-band anchor) on the seven
# starting models of the sweep that still lack it; the headline run (21.9 dB)
# has it with five seeds.  One seed, clean and noisy data, U-Net and ResCNN,
# same schedule and warm start as every other run; the stored 2-D baseline of
# each run folder is the input prior and the noise realization of the folder is
# reused.  Two groups of folders run side by side on four GPUs each, and the
# report of every folder is regenerated after its trainings.
#
#   GROUP_A, GROUP_B  run folders of the two groups (defaults below)
#   SEEDS             (default 42)

set -eo pipefail

ROOT=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn
PY=$ROOT/env/py311_torch210/bin/python
CODE=$ROOT/code
RUNNER="$PY $CODE/run_marmousi_full.py"
GROUP_A=${GROUP_A:-"full_all_e3_fault full_sm_g16x24 full_sm_g32x48 full_sm_w48"}
GROUP_B=${GROUP_B:-"full_sm_g136x56 full_sm_g64x96 full_sm_g272x112"}
SEEDS=${SEEDS:-42}
NCPU=$SLURM_CPUS_PER_TASK

echo "job $SLURM_JOB_ID on $(hostname): $(nvidia-smi -L | wc -l) GPUs, $NCPU cores, started $(date -Is)"
echo "group A [$GROUP_A]  group B [$GROUP_B]  seeds [$SEEDS]  prior tv"
"$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
for m in $GROUP_A $GROUP_B; do
    [ -f "$ROOT/runs/$m/baseline/baseline.json" ] || { echo "no 2-D baseline in runs/$m"; exit 1; }
done

run_group() {
    local gpus=$1; shift
    for m in "$@"; do
        local RUN=$ROOT/runs/$m
        echo "=== $m  started $(date -Is)"
        CUDA_VISIBLE_DEVICES=$gpus OMP_NUM_THREADS=2 $RUNNER sweep --out "$RUN" \
            --networks unet rescnn --datas clean noisy --supervisions wells --n-wells 5 \
            --seeds $SEEDS --train-set all --epochs 80 --transfer-dir "$ROOT/transfer" --device cuda \
            --gpus 4 --per-gpu 1 --threads 2 --prior tv --lowband 0
        OMP_NUM_THREADS=32 $RUNNER report --out "$RUN"
        echo "=== $m  reported $(date -Is)"
    done
}

run_group 0,1,2,3 $GROUP_A > "$ROOT/jobs/logs/sm_sweep_tv-$SLURM_JOB_ID-A.log" 2>&1 &
PID_A=$!
run_group 4,5,6,7 $GROUP_B > "$ROOT/jobs/logs/sm_sweep_tv-$SLURM_JOB_ID-B.log" 2>&1 &
PID_B=$!
wait $PID_A || echo "group A returned an error"
wait $PID_B || echo "group B returned an error"
echo "job finished $(date -Is)"
