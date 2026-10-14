#!/bin/bash -l
#SBATCH --job-name=marmousi_full
#SBATCH --account=plgadmmpinn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=10:00:00
#SBATCH --output=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn/jobs/logs/%x-%j.out

# Complete Marmousi-2 benchmark on one whole Athena node: the classical
# solvers use all 128 cores, the trainings are packed over the 8 A100s.
#
# Settings come from the environment, e.g.
#   sbatch --export=ALL,RUN_NAME=full_blur_e3_fault,SIGMA_X=272,SIGMA_T=112,EPSILON=1e-3,NOISE_REF=fault,LABEL_COUNTS="2 10 20" jobs/marmousi_full_node.sbatch
#   sbatch --export=ALL,RUN_NAME=full_sm_wells48,STARTING_MODEL=wells,WELL_LOWPASS=48,SUPERVISIONS="none wells",SEEDS=42 jobs/marmousi_full_node.sbatch
# Cost from the measured A100 rate (about 2.3 ms per patch and step, two
# trainings per GPU): one all-patch training = 60 min, 16 run at a time, so
# 60 trainings need four waves (4 h); the ADMM prior takes 2 min and the 2-D
# TV baseline 50 min per data set on the free cores.
#
#   RUN_NAME        run folder under runs/                     (default full_all)
#   STARTING_MODEL  gaussian | wells                           (default gaussian)
#   SIGMA_X SIGMA_T Gaussian widths, traces x samples          (default 8 x 12, the paper's crop rule)
#   WELL_LOWPASS    time low-pass of the well logs, samples    (default 48; wells model only)
#   NOISE_REF       whole | fault | second                     (default whole)
#   EPSILON         reweighting floor of the ADMM prior        (default 1e-4, the paper's value)
#   TRAIN_SET       all | subset                               (default all: every patch of the model)
#   N_RANDOM        random patches added to the crops          (subset only, default 2000)
#   NETWORKS        space-separated                            (default "unet rescnn")
#   SUPERVISIONS    dense none wells                           (default "dense none wells")
#   N_WELLS         pseudo-wells of the wells variant          (default 5)
#   PRIOR           rw | tv  network input prior               (default rw, the paper; tv = 2-D coupled result)
#   LOWBAND         weight of the low-band anchor              (default 0 = off, the paper)
#   LOWBAND_SIGMA   low-pass width of that anchor, samples     (default 12)
#   SEEDS           space-separated                            (default "42 123 7 2024 2025")
#   LABEL_COUNTS    extra pseudo-well counts, one seed each    (default none, e.g. "2 10 20")
#   LC_SEEDS        seeds of the label-count runs              (default 42)
#   EPOCHS          training epochs                            (default 80)
#   PER_GPU         concurrent trainings per GPU               (default 2)
#   BASELINE        1 = run the 2-D TV baseline alongside      (default 1)

set -eo pipefail

ROOT=/net/pr2/projects/plgrid/plggseismicresearch/admm_pinn
PY=$ROOT/env/py311_torch210/bin/python
CODE=$ROOT/code
RUNNER="$PY $CODE/run_marmousi_full.py"
RUN=$ROOT/runs/${RUN_NAME:-full_all}
STARTING_MODEL=${STARTING_MODEL:-gaussian}
SIGMA_X=${SIGMA_X:-8}
SIGMA_T=${SIGMA_T:-12}
WELL_LOWPASS=${WELL_LOWPASS:-48}
NOISE_REF=${NOISE_REF:-whole}
EPSILON=${EPSILON:-1e-4}
TRAIN_SET=${TRAIN_SET:-all}
N_RANDOM=${N_RANDOM:-2000}
NETWORKS=${NETWORKS:-"unet rescnn"}
SUPERVISIONS=${SUPERVISIONS:-"dense none wells"}
N_WELLS=${N_WELLS:-5}
PRIOR=${PRIOR:-rw}
LOWBAND=${LOWBAND:-0}
LOWBAND_SIGMA=${LOWBAND_SIGMA:-12}
SEEDS=${SEEDS:-"42 123 7 2024 2025"}
LABEL_COUNTS=${LABEL_COUNTS:-""}
LC_SEEDS=${LC_SEEDS:-42}
EPOCHS=${EPOCHS:-80}
PER_GPU=${PER_GPU:-2}
BASELINE=${BASELINE:-1}
NGPU=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
NCPU=$SLURM_CPUS_PER_TASK

echo "job $SLURM_JOB_ID on $(hostname): $NGPU GPUs, $NCPU cores, started $(date -Is)"
echo "run $RUN  starting model $STARTING_MODEL ($SIGMA_X x $SIGMA_T / low-pass $WELL_LOWPASS)  epsilon $EPSILON  noise $NOISE_REF"
echo "train-set $TRAIN_SET  networks [$NETWORKS]  supervisions [$SUPERVISIONS]  wells $N_WELLS  seeds [$SEEDS]  label counts [$LABEL_COUNTS]"
echo "prior $PRIOR  low-band anchor $LOWBAND (sigma $LOWBAND_SIGMA)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PY" -c "import threadpoolctl, torch; assert torch.cuda.is_available(), 'no CUDA'"
mkdir -p "$RUN"

# 1. model, data, starting model and the trace-wise ADMM prior on all cores
if [ ! -f "$RUN/prepared/prepared.json" ]; then
    export OMP_NUM_THREADS=$NCPU
    $RUNNER prepare --out "$RUN" --marmousi-dir "$ROOT/data/marmousi2" \
        --starting-model "$STARTING_MODEL" --smooth-sigma-x "$SIGMA_X" --smooth-sigma-t "$SIGMA_T" \
        --well-lowpass "$WELL_LOWPASS" --noise-reference "$NOISE_REF" --epsilon "$EPSILON" \
        --n-jobs "$NCPU"
else
    echo "prepared data found, reused"
fi

# 2. trainings packed over the GPUs; the 2-D TV baseline runs on the cores
#    that the trainings leave free (at least 16)
BASE_ARGS=()
if [ "$PRIOR" = "tv" ] && [ ! -f "$RUN/baseline/baseline.json" ]; then
    # the 2-D coupled result is the network input: it has to exist before training
    export OMP_NUM_THREADS=$NCPU
    $RUNNER baseline --out "$RUN" --n-jobs "$NCPU"
elif [ "$BASELINE" = "1" ] && [ ! -f "$RUN/baseline/baseline.json" ]; then
    FREE=$(( NCPU - NGPU * PER_GPU * 2 ))
    [ "$FREE" -lt 16 ] && FREE=16
    BASE_ARGS=(--with-baseline --baseline-jobs "$FREE")
fi
VARIANT_ARGS=(--prior "$PRIOR" --lowband "$LOWBAND" --lowband-sigma "$LOWBAND_SIGMA")
export OMP_NUM_THREADS=2
$RUNNER sweep --out "$RUN" \
    --networks $NETWORKS --datas clean noisy --supervisions $SUPERVISIONS --n-wells "$N_WELLS" \
    --seeds $SEEDS --train-set "$TRAIN_SET" --n-random "$N_RANDOM" \
    --epochs "$EPOCHS" --transfer-dir "$ROOT/transfer" --device cuda \
    --gpus "$NGPU" --per-gpu "$PER_GPU" --threads 2 "${VARIANT_ARGS[@]}" "${BASE_ARGS[@]}"

# 3. label-count runs: the pseudo-well variant with other numbers of wells
for k in $LABEL_COUNTS; do
    $RUNNER sweep --out "$RUN" \
        --networks $NETWORKS --datas clean noisy --supervisions wells --n-wells "$k" \
        --seeds $LC_SEEDS --train-set "$TRAIN_SET" --n-random "$N_RANDOM" \
        --epochs "$EPOCHS" --transfer-dir "$ROOT/transfer" --device cuda \
        --gpus "$NGPU" --per-gpu "$PER_GPU" --threads 2 "${VARIANT_ARGS[@]}"
done

# 4. tables and figures
export OMP_NUM_THREADS=$NCPU
$RUNNER report --out "$RUN"

echo "job finished $(date -Is)"
