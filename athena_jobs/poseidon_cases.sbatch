#!/bin/bash -l
#SBATCH --job-name=poseidon
#SBATCH --account=plgadmmpinn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn/jobs/logs/%x-%j.out

# Poseidon field example, protocol v1: the five leave-one-well-out cases.
#   1. phase A prior (classical inversions of the ten lines per case) on the cores, cases in sequence;
#   2. phase A trainings packed over the GPUs (networks x inputs x seeds x cases, two per GPU);
#   3. seal, phase B score, report.
#   SEEDS     (default "42 123 7 2024 2025")   INPUTS (default "tv rw")   NETWORKS (default "unet rescnn")
#   RUN_NAME  run folder under runs/ (default poseidon_v1)

set -eo pipefail
ROOT=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn
PY=$ROOT/env/py311_torch210/bin/python
RUNNER="$PY $ROOT/code/poseidon/poseidon_runner.py"
PREP=$ROOT/field/poseidon/prepared
RUN=$ROOT/runs/${RUN_NAME:-poseidon_v1}
SEEDS=${SEEDS:-"42 123 7 2024 2025"}
INPUTS=${INPUTS:-"tv rw"}
NETWORKS=${NETWORKS:-"unet rescnn"}
PER_GPU=${PER_GPU:-2}
PRIOR_JOBS=${PRIOR_JOBS:-16}
NGPU=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
NCPU=$SLURM_CPUS_PER_TASK
CASES=("Proteus 1" "Pharos 1" "Poseidon 2" "Kronos 1" "Torosa 1")
mkdir -p "$RUN"
echo "job $SLURM_JOB_ID on $(hostname): $NGPU GPUs, $NCPU cores, started $(date -Is); run $RUN"
"$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

# 1. phase A priors
export OMP_NUM_THREADS=1
for c in "${CASES[@]}"; do
    if [ -f "$RUN/cases/${c// /_}/phaseA/prior_record.json" ]; then echo "prior of $c exists"; continue; fi
    $RUNNER prior --prepared "$PREP" --out "$RUN" --case "$c" --n-jobs "$PRIOR_JOBS"
done

# 2. phase A trainings: one task line per training, packed with xargs over GPU slots
export OMP_NUM_THREADS=2
TASKS=$RUN/train_tasks.txt; : > "$TASKS"
i=0
for c in "${CASES[@]}"; do for n in $NETWORKS; do for inp in $INPUTS; do for s in $SEEDS; do
    if [ -f "$RUN/cases/${c// /_}/phaseA/net_${n}_${inp}_s${s}/train_record.json" ]; then continue; fi
    gpu=$(( i % NGPU )); i=$(( i + 1 ))
    echo "$gpu|$c|$n|$inp|$s" >> "$TASKS"
done; done; done; done
echo "$(wc -l < "$TASKS") trainings over $(( NGPU * PER_GPU )) slots"
run_one() {
    IFS='|' read -r gpu c n inp s <<< "$1"
    CUDA_VISIBLE_DEVICES=$gpu $RUNNER train --prepared "$PREP" --out "$RUN" --case "$c" --network "$n" --input "$inp" \
        --seed "$s" --transfer-dir "$ROOT/transfer" --device cuda > "$RUN/train_${c// /_}_${n}_${inp}_s${s}.log" 2>&1 \
        && echo "ok   $c $n $inp s$s" || echo "FAIL $c $n $inp s$s"
}
export -f run_one; export RUNNER PREP RUN ROOT
xargs -a "$TASKS" -d '\n' -P $(( NGPU * PER_GPU )) -I{} bash -c 'run_one "$@"' _ {}

# 3. seal, score, report
export OMP_NUM_THREADS=$NCPU
for c in "${CASES[@]}"; do
    $RUNNER seal --prepared "$PREP" --out "$RUN" --case "$c"
    $RUNNER score --prepared "$PREP" --out "$RUN" --case "$c"
done
$RUNNER report --prepared "$PREP" --out "$RUN"
echo "job finished $(date -Is)"
