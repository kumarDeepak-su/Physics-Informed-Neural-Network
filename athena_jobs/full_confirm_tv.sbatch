#!/bin/bash -l
#SBATCH --job-name=full_confirm_tv
#SBATCH --account=plgadmmpinn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=08:00:00
#SBATCH --output=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn/jobs/logs/%x-%j.out

# Confirmation on the complete model of the configuration selected on the
# second crop (2-D coupled result as the network input prior, no low-band
# anchor), in the headline run folder:
#   trainings  dense / label-free / five wells with five seeds, and 2 / 10 / 20
#              wells with seed 42, U-Net and ResCNN, clean and noisy data,
#              input prior = 2-D coupled result ("+tv" labels);
#   comparator the 2-D coupled baseline conditioned on the same wells
#              (baseline_wells<n>/ for n = 2, 5, 10, 20), on the cores the
#              trainings leave free;
#   report     regenerated at the end (all groups of the run folder).
#
#   RUN_NAME     run folder under runs/ (default full_s22x33_e3_fault)
#   WELL_WEIGHT  weight of the well penalty in the comparator (default 1)
#   SEEDS        seeds of the five-well / dense / label-free trainings
#   LABEL_COUNTS extra well counts with LC_SEEDS (default "2 10 20", seed 42)

set -eo pipefail

ROOT=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn
PY=$ROOT/env/py311_torch210/bin/python
CODE=$ROOT/code
RUNNER="$PY $CODE/run_marmousi_full.py"
RUN=$ROOT/runs/${RUN_NAME:-full_s22x33_e3_fault}
WELL_WEIGHT=${WELL_WEIGHT:-1}
SEEDS=${SEEDS:-"42 123 7 2024 2025"}
LABEL_COUNTS=${LABEL_COUNTS:-"2 10 20"}
LC_SEEDS=${LC_SEEDS:-42}
PER_GPU=${PER_GPU:-2}
NGPU=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
NCPU=$SLURM_CPUS_PER_TASK

echo "job $SLURM_JOB_ID on $(hostname): $NGPU GPUs, $NCPU cores, started $(date -Is)"
echo "run $RUN  prior tv  well weight $WELL_WEIGHT  seeds [$SEEDS]  label counts [$LABEL_COUNTS]"
[ -f "$RUN/prepared/prepared.json" ] && [ -f "$RUN/baseline/baseline.json" ] \
    || { echo "prepared data and the 2-D baseline of $RUN are required"; exit 1; }
"$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

# 1. well-conditioned 2-D baselines on the free cores (trainings use 2 threads each)
FREE=$(( NCPU - NGPU * PER_GPU * 2 ))
[ "$FREE" -lt 16 ] && FREE=16
(
    export OMP_NUM_THREADS=1
    for k in 2 5 $LABEL_COUNTS; do
        if [ -f "$RUN/baseline_wells$k/baseline.json" ]; then
            echo "baseline_wells$k exists, reused"
            continue
        fi
        echo "=== well-conditioned baseline, $k wells, weight $WELL_WEIGHT  $(date -Is)"
        $RUNNER baseline --out "$RUN" --wells --n-wells "$k" --well-weight "$WELL_WEIGHT" --n-jobs "$FREE"
    done
    echo "well-conditioned baselines finished $(date -Is)"
) > "$RUN/baseline_wells.log" 2>&1 &
BASE_PID=$!

# 2. trainings with the 2-D coupled result as the input prior
export OMP_NUM_THREADS=2
$RUNNER sweep --out "$RUN" \
    --networks unet rescnn --datas clean noisy --supervisions dense none wells --n-wells 5 \
    --seeds $SEEDS --train-set all --epochs 80 --transfer-dir "$ROOT/transfer" --device cuda \
    --gpus "$NGPU" --per-gpu "$PER_GPU" --threads 2 --prior tv --lowband 0
for k in $LABEL_COUNTS; do
    $RUNNER sweep --out "$RUN" \
        --networks unet rescnn --datas clean noisy --supervisions wells --n-wells "$k" \
        --seeds $LC_SEEDS --train-set all --epochs 80 --transfer-dir "$ROOT/transfer" --device cuda \
        --gpus "$NGPU" --per-gpu "$PER_GPU" --threads 2 --prior tv --lowband 0
done

# 3. comparators must be in before the report
wait $BASE_PID || echo "well-conditioned baseline step returned an error (see $RUN/baseline_wells.log)"
cat "$RUN/baseline_wells.log"

# 4. tables and figures
export OMP_NUM_THREADS=$NCPU
$RUNNER report --out "$RUN"

echo "job finished $(date -Is)"
