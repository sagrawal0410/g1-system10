#!/usr/bin/env bash
#SBATCH --job-name=groot-sft
#SBATCH --partition=defq
#SBATCH --account=liquidai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-gpu=12
#SBATCH --time=08:00:00
#SBATCH --output=/lambdafs/shaurya/g1_sonic_system1/logs/%x-%j.out
set -euo pipefail
# handsout: defaults | handsin: override DATASET/OUTPUT_DIR/EXP
# baseproxy: NUM_GPUS=1 SAVE_STEPS=10 MAX_STEPS=10 GLOBAL_BATCH_SIZE=8 USE_WANDB=0 (sbatch --gpus-per-node=1)
DATASET="${DATASET:-/lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic_train}"
OUTPUT_DIR="${OUTPUT_DIR:-/lambdafs/shaurya/g1_sonic_system1/outputs/grootN17-sft-handsout}"
EXP="${EXP:-grootN17-sft-handsout}"
export HF_HUB_CACHE=/lambdafs/shaurya/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
conda activate groot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export ACTION_LAYOUT_PATH=/home/shaurya/g1_sonic_system1/results/action_layout.json
export NUM_GPUS="${NUM_GPUS:-3}"
export TORCHRUN_STANDALONE=1
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-30}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT=g1-sonic-vla
export WANDB__SERVICE_WAIT=300
echo "[train_groot] job=${SLURM_JOB_ID:-NA} dataset=$DATASET out=$OUTPUT_DIR exp=$EXP"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || true
cd ~/g1_sonic_system1/repos/Isaac-GR00T
exec bash ~/g1_sonic_system1/scripts/slurm/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B --dataset-path "$DATASET" \
    --embodiment-tag UNITREE_G1_SONIC --output-dir "$OUTPUT_DIR" \
    --experiment-name "$EXP" --wandb-project g1-sonic-vla
