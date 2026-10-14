#!/bin/bash -l
#SBATCH --job-name=ensemble
#SBATCH --account=plgadmmpinn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn/jobs/logs/%x-%j.out
set -eo pipefail
ROOT=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn
$ROOT/env/py311_torch210/bin/python $ROOT/code/ensemble_full_marmousi.py --run $ROOT/runs/${RUN_NAME:-full_s22x33_e3_fault}
echo "job finished $(date -Is)"
