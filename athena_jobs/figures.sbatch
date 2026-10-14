#!/bin/bash -l
#SBATCH --job-name=figures
#SBATCH --account=plgadmmpinn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn/jobs/logs/%x-%j.out

# Manuscript figures of the complete-model section from the stored outputs
# (figures_full_marmousi.py), written next to the run: <run>/figures_manuscript/.
#   RUN_NAME  headline run folder (default full_s22x33_e3_fault)
#   SWEEP     sweep run folders, space separated (default: the eight starting models)
#   DATA      clean | noisy (default noisy)

set -eo pipefail
ROOT=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn
PY=$ROOT/env/py311_torch210/bin/python
RUN=$ROOT/runs/${RUN_NAME:-full_s22x33_e3_fault}
SWEEP=${SWEEP:-"full_all_e3_fault full_sm_g16x24 full_sm_g32x48 full_sm_g136x56 full_sm_g64x96 full_sm_g272x112 full_sm_w48"}
DATA=${DATA:-noisy}
SWEEP_DIRS=""
for s in $SWEEP; do SWEEP_DIRS="$SWEEP_DIRS $ROOT/runs/$s"; done
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPLBACKEND=Agg
echo "job $SLURM_JOB_ID on $(hostname), started $(date -Is): run $RUN, data $DATA"
$PY $ROOT/code/figures_full_marmousi.py --run "$RUN" --out "$RUN/figures_manuscript" --data "$DATA" --sweep $SWEEP_DIRS
echo "job finished $(date -Is)"
