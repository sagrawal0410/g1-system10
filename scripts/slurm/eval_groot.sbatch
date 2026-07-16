#!/usr/bin/env bash
#SBATCH --job-name=groot-eval-openloop
#SBATCH --partition=defq
#SBATCH --account=liquidai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=01:00:00
#SBATCH --output=/lambdafs/shaurya/g1_sonic_system1/logs/%x-%j.out
set -euo pipefail

export HF_HUB_CACHE=/lambdafs/shaurya/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
conda activate groot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

DATASET="${DATASET:-/lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic}"
OUT_CSV="${OUT_CSV:-/lambdafs/shaurya/g1_sonic_system1/results/openloop_metrics.csv}"
PLOT_DIR="${PLOT_DIR:-/lambdafs/shaurya/g1_sonic_system1/results/openloop_plots}"
STEPS="${STEPS:-400}"
CKPTS="${CKPTS:?Set CKPTS='label=path ...'}"

CK_ARGS=()
for kv in $CKPTS; do CK_ARGS+=(--checkpoint "$kv"); done

echo "[eval] node=$(hostname) job=$SLURM_JOB_ID dataset=$DATASET ckpts=$CKPTS"
cd ~/g1_sonic_system1/repos/Isaac-GR00T
exec python ~/g1_sonic_system1/scripts/eval_openloop.py \
    --action-layout /home/shaurya/g1_sonic_system1/results/action_layout.json \
    --eval-split /home/shaurya/g1_sonic_system1/results/eval_split.json \
    --dataset-path "$DATASET" \
    --embodiment-tag UNITREE_G1_SONIC \
    --steps "$STEPS" \
    --output-csv "$OUT_CSV" \
    --plot-dir "$PLOT_DIR" \
    --device cuda:0 \
    "${CK_ARGS[@]}"
