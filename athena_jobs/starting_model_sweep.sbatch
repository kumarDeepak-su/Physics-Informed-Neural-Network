#!/bin/bash -l
#SBATCH --job-name=sm_sweep
#SBATCH --account=plgadmmpinn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=10:00:00
#SBATCH --output=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn/jobs/logs/%x-%j.out

# Sensitivity to the low-frequency starting model (reviewer comment 2) on the
# complete Marmousi-2 model: one run folder per starting model, each with its
# own ADMM prior, 2-D TV baseline and the two practical network variants
# (label-free and five pseudo-wells), one seed.  The paper's crop rule
# (8 x 12, run full_all_e3_fault) and the scaled rule (272 x 112, run
# full_blur_e3_fault) complete the series.
#
#   MODELS   space-separated items  g<sx>x<st>  (Gaussian)  or  w<lowpass>  (five wells)
#            (default "g16x24 g32x48 g64x96 g136x56 w48")
#   SEEDS    (default 42)      EPSILON (default 1e-3)      NOISE_REF (default fault)

set -eo pipefail

ROOT=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn
PY=$ROOT/env/py311_torch210/bin/python
CODE=$ROOT/code
RUNNER="$PY $CODE/run_marmousi_full.py"
MODELS=${MODELS:-"g16x24 g32x48 g64x96 g136x56 w48"}
SEEDS=${SEEDS:-42}
EPSILON=${EPSILON:-1e-3}
NOISE_REF=${NOISE_REF:-fault}
NGPU=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
NCPU=$SLURM_CPUS_PER_TASK

echo "job $SLURM_JOB_ID on $(hostname): $NGPU GPUs, $NCPU cores, started $(date -Is)"
echo "starting models [$MODELS]  seeds [$SEEDS]  epsilon $EPSILON  noise $NOISE_REF"
"$PY" -c "import threadpoolctl, torch; assert torch.cuda.is_available(), 'no CUDA'"

for m in $MODELS; do
    RUN=$ROOT/runs/full_sm_$m
    mkdir -p "$RUN"
    case $m in
        g*) sx=${m#g}; sx=${sx%x*}; st=${m#*x}
            PREP=(--starting-model gaussian --smooth-sigma-x "$sx" --smooth-sigma-t "$st") ;;
        w*) PREP=(--starting-model wells --well-lowpass "${m#w}") ;;
        *)  echo "unknown model $m"; exit 1 ;;
    esac
    echo "=== starting model $m -> $RUN  $(date -Is)"
    if [ ! -f "$RUN/prepared/prepared.json" ]; then
        OMP_NUM_THREADS=$NCPU $RUNNER prepare --out "$RUN" --marmousi-dir "$ROOT/data/marmousi2" \
            "${PREP[@]}" --noise-reference "$NOISE_REF" --epsilon "$EPSILON" --n-jobs "$NCPU"
    fi
    BASE_ARGS=()
    [ -f "$RUN/baseline/baseline.json" ] || BASE_ARGS=(--with-baseline --baseline-jobs 96)
    OMP_NUM_THREADS=2 $RUNNER sweep --out "$RUN" --networks unet rescnn --datas clean noisy \
        --supervisions none wells --n-wells 5 --seeds $SEEDS --train-set all --epochs 80 \
        --transfer-dir "$ROOT/transfer" --device cuda --gpus "$NGPU" --per-gpu 2 --threads 2 \
        "${BASE_ARGS[@]}"
    OMP_NUM_THREADS=$NCPU $RUNNER report --out "$RUN"
done

echo "job finished $(date -Is)"
