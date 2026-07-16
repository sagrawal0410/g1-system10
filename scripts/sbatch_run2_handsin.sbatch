#!/usr/bin/env bash
#SBATCH --job-name=grootN17-sft-handsin-taskcond
#SBATCH --partition=defq
#SBATCH --account=liquidai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-gpu=12
#SBATCH --time=08:00:00
#SBATCH --output=/lambdafs/shaurya/g1_sonic_system1/logs/%x-%j.out
# RUN 2 — GR00T-B "hands in, normalized + task-conditioned"
# (per-task-normalized hand_cmd_pico in the 14 hand dims + per-task masks).
# Submit with:  DATASET=<train-only dir>  sbatch sbatch_run2_handsin.sbatch
# (defaults to g1_encoded_sonic_handsB_train if DATASET unset)
set -euo pipefail

DATASET="${DATASET:-/lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic_handsB_train}"

export HF_HUB_CACHE=/lambdafs/shaurya/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
conda activate groot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export ACTION_LAYOUT_PATH=/home/shaurya/g1_sonic_system1/results/action_layout.json

export NUM_GPUS=3
export TORCHRUN_STANDALONE=1
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-30}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export USE_WANDB=1
export WANDB_PROJECT=g1-sonic-vla
export WANDB__SERVICE_WAIT=300

echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID gpus=$SLURM_GPUS_PER_NODE dataset=$DATASET"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || true

cd ~/g1_sonic_system1/repos/Isaac-GR00T
exec bash ~/g1_sonic_system1/scripts/finetune_with_block_loss.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path "$DATASET" \
    --embodiment-tag UNITREE_G1_SONIC \
    --output-dir /lambdafs/shaurya/g1_sonic_system1/outputs/grootN17-sft-handsin-taskcond \
    --experiment-name grootN17-sft-handsin-taskcond \
    --wandb-project g1-sonic-vla
