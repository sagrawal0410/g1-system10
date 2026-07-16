#!/usr/bin/env bash
#SBATCH --job-name=groot-baseproxy
#SBATCH --partition=defq
#SBATCH --account=liquidai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=00:40:00
#SBATCH --output=/lambdafs/shaurya/g1_sonic_system1/logs/%x-%j.out
# Near-zero-step BASE PROXY for GATE A row (a): base N1.7 adapted to the
# UNITREE_G1_SONIC embodiment head with essentially no training (max_steps=10).
# Represents "base model on this embodiment before fine-tuning" (base can't run
# this posttrain tag zero-shot). Submit with DATASET/OUTPUT_DIR/EXP env.
set -euo pipefail

DATASET="${DATASET:?set DATASET}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
EXP="${EXP:?set EXP}"

export HF_HUB_CACHE=/lambdafs/shaurya/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
conda activate groot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export ACTION_LAYOUT_PATH=/home/shaurya/g1_sonic_system1/results/action_layout.json

# 1 GPU (Slurm sets visibility). Small batch to fit one 80GB GPU. No DDP/deepspeed.
export NUM_GPUS=1
export SAVE_STEPS=10
export MAX_STEPS=10
export GLOBAL_BATCH_SIZE=8
export DATALOADER_NUM_WORKERS=4
export USE_WANDB=0

echo "[baseproxy] node=$(hostname) job=$SLURM_JOB_ID dataset=$DATASET out=$OUTPUT_DIR exp=$EXP"
cd ~/g1_sonic_system1/repos/Isaac-GR00T
exec bash ~/g1_sonic_system1/scripts/finetune_with_block_loss.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path "$DATASET" \
    --embodiment-tag UNITREE_G1_SONIC \
    --output-dir "$OUTPUT_DIR" \
    --experiment-name "$EXP"
