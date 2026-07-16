#!/usr/bin/env bash
#SBATCH --job-name=d3-T3-rabc
#SBATCH --partition=defq
#SBATCH --account=liquidai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=12
#SBATCH --time=04:00:00
#SBATCH --output=/lambdafs/shaurya/g1_sonic_system1/logs/%x-%j.out
# Phase D3 (T3) — TOPReward RA-BC weighted SHORT continuation of GR00T-A.
# Continuation init = GR00T-A checkpoint-20000 (== T1 eval point).
# Env overrides:
#   RABC_WEIGHTS_JSON   (required) per-frame RA-BC weights
#   EXPERIMENT_NAME     (default grootN17-sft-T3-rabc)
#   MAX_STEPS           (default 1500)  LR (default 2e-5)  GLOBAL_BATCH_SIZE (default 20)
set -x -euo pipefail

export HF_HUB_CACHE=/lambdafs/shaurya/hf_cache/hub
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate groot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export ACTION_LAYOUT_PATH=/home/shaurya/g1_sonic_system1/results/action_layout.json

export RABC_WEIGHTS_JSON="${RABC_WEIGHTS_JSON:?set RABC_WEIGHTS_JSON}"
export RABC_FALLBACK_WEIGHT="${RABC_FALLBACK_WEIGHT:-1.0}"

BASE_CKPT="${BASE_CKPT:-/lambdafs/shaurya/g1_sonic_system1/outputs/grootN17-sft-handsout/grootN17-sft-handsout/checkpoint-20000}"
DATASET="${DATASET:-/lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic_train}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grootN17-sft-T3-rabc}"
OUTDIR="/lambdafs/shaurya/g1_sonic_system1/outputs/${EXPERIMENT_NAME}"

NUM_GPUS="${NUM_GPUS:-2}"
MAX_STEPS="${MAX_STEPS:-1500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-20}"
LR="${LR:-2e-5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

export WANDB_PROJECT=g1-sonic-vla
export WANDB__SERVICE_WAIT=300

echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID gpus=${SLURM_GPUS_PER_NODE:-?} base=$BASE_CKPT weights=$RABC_WEIGHTS_JSON"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || true

cd ~/g1_sonic_system1/repos/Isaac-GR00T

LAUNCH=(
    /home/shaurya/g1_sonic_system1/scripts/d3_launch_sft_rabc.py
    --base_model_path "$BASE_CKPT"
    --dataset_path "$DATASET"
    --embodiment_tag UNITREE_G1_SONIC
    --num_gpus "$NUM_GPUS"
    --output_dir "$OUTDIR"
    --experiment_name "$EXPERIMENT_NAME"
    --wandb_project "$WANDB_PROJECT"
    --save_steps "$SAVE_STEPS"
    --save_total_limit 5
    --max_steps "$MAX_STEPS"
    --warmup_ratio 0.05
    --weight_decay 1e-5
    --learning_rate "$LR"
    --use_wandb
    --global_batch_size "$GLOBAL_BATCH_SIZE"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --shard_size 1024
    --num_shards_per_epoch 100000
    --episode_sampling_rate 0.1
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
)

exec torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" "${LAUNCH[@]}"
