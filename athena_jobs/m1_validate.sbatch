#!/bin/bash -l
#SBATCH --job-name=m1_validate
#SBATCH --account=plgadmmpinn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn/jobs/logs/%x-%j.out

# Milestone M1: build the software environment (once), run the published
# fault crop on the A100 and on the CPU cores of the same node, and compare
# both runs with the published result.

set -eo pipefail

ROOT=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn
ENV=$ROOT/env/py311_torch210
PY=$ENV/bin/python
CODE=$ROOT/code
MAIN=2D_ADMM_PINN_ResCNN_Attention.py

echo "job $SLURM_JOB_ID on $(hostname), started $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ---------------------------------------------------------------- environment
if [ ! -x "$PY" ]; then
    echo "[env] building $ENV"
    module load Miniconda3/25.7.0-2
    export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs
    mkdir -p "$ROOT/env"
    conda create -y -q -p "$ENV" -c conda-forge --override-channels python=3.11 pip
    "$PY" -m pip install --no-cache-dir torch==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu126
    "$PY" -m pip install --no-cache-dir numpy==2.4.6 scipy==1.17.1 matplotlib==3.10.9
    "$PY" -m pip freeze > "$ROOT/env/py311_torch210.freeze.txt"
fi

"$PY" - <<'PYEOF'
import sys, numpy, scipy, matplotlib, torch
print("[env] python", sys.version.split()[0], "| numpy", numpy.__version__,
      "| scipy", scipy.__version__, "| matplotlib", matplotlib.__version__,
      "| torch", torch.__version__)
print("[env] cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA is not available to PyTorch")
print("[env] device:", torch.cuda.get_device_name(0))
PYEOF

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

COMMON="--benchmark marmousi-crop --marmousi-dir $ROOT/data/marmousi2 \
--transfer-dir $ROOT/transfer --epochs 80"

run_case () {   # $1 = device, $2 = run tag
    local out=$ROOT/runs/$2
    if [ -e "$out/2D_results.npz" ]; then
        echo "[run] $2 already has results; refusing to overwrite"; exit 1
    fi
    mkdir -p "$out"
    echo "[run] $2 on $1, started $(date -Is)"
    if [ "$1" = "cuda" ]; then
        nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
            --format=csv,noheader -l 2 > "$out/gpu_trace.csv" &
        local smi=$!
    fi
    local t0=$SECONDS
    ( cd "$CODE" && "$PY" $MAIN $COMMON --device "$1" --run-tag "$2" \
        --results-dir "$out" ) > "$out/stdout.log" 2>&1
    echo "[run] $2 finished in $((SECONDS - t0)) s"
    if [ "$1" = "cuda" ]; then kill $smi 2>/dev/null || true; fi
    "$PY" "$ROOT/tools/run_record.py" --run-dir "$out" --device "$1" \
        --command "python $MAIN $COMMON --device $1 --run-tag $2 --results-dir $out" \
        --inputs "$CODE/$MAIN" "$ROOT"/data/marmousi2/*.segy "$ROOT"/transfer/*.pt
}

TAG=${M1_TAG:-m1_fault}
run_case cuda ${TAG}_cuda
run_case cpu  ${TAG}_cpu

"$PY" "$ROOT/tools/compare_runs.py" \
    --published "$ROOT/reference/marmousi_fault_crop_400x300" \
    --multiseed "$ROOT/reference/multiseed/marmousi_fault_crop_400x300" \
    --cpu  "$ROOT/runs/${TAG}_cpu" \
    --cuda "$ROOT/runs/${TAG}_cuda" \
    --out  "$ROOT/runs/${TAG}_report"

echo "job finished $(date -Is)"
