#!/bin/bash -l
#SBATCH --job-name=poseidon_fig
#SBATCH --account=plgadmmpinn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn/jobs/logs/%x-%j.out
# Figures of the Poseidon field example from the sealed run outputs (figures_poseidon.py).
set -eo pipefail
ROOT=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn
PY=$ROOT/env/py311_torch210/bin/python
RUN=$ROOT/runs/${RUN_NAME:-poseidon_v1}
export MPLBACKEND=Agg OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
$PY $ROOT/code/poseidon/figures_poseidon.py --prepared $ROOT/field/poseidon/prepared --run "$RUN" --out "$RUN/figures_manuscript" --section-line ${SECTION_LINE:-Proteus_1_inline} --all-sections
echo "job finished $(date -Is)"
