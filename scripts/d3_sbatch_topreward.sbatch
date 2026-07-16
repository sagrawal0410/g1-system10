#!/usr/bin/env bash
#SBATCH --job-name=d3-topreward
#SBATCH --partition=defq
#SBATCH --account=liquidai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=03:00:00
#SBATCH --output=/lambdafs/shaurya/g1_sonic_system1/logs/%x-%j.out
# Phase D3 — TOPReward zero-shot progress over the 19 train demos.
set -euo pipefail

export HF_HUB_CACHE=/lambdafs/shaurya/hf_cache/hub
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
# lerobot env has no ffmpeg; torchcodec video decode borrows the groot env's
# ffmpeg-7 shared libs (proven to decode these mp4s).
export LD_LIBRARY_PATH="/home/shaurya/miniconda3/envs/groot/lib:${LD_LIBRARY_PATH:-}"

OUTDIR=/lambdafs/shaurya/g1_sonic_system1/results/phase_d3
mkdir -p "$OUTDIR"

echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID gpus=${SLURM_GPUS_PER_NODE:-?}"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || true

cd ~/g1_sonic_system1/repos/lerobot
exec python /home/shaurya/g1_sonic_system1/scripts/d3_topreward_compute.py \
    --dataset-root /lambdafs/shaurya/g1_sonic_system1/data/g1_act_lerobot \
    --image-key observation.images.head_cam \
    --vlm-name Qwen/Qwen3-VL-8B-Instruct \
    --num-samples 15 \
    --fps 2.0 \
    --device cuda \
    --output-path "$OUTDIR/topreward_progress_video.parquet" \
    --raw-json "$OUTDIR/topreward_raw_anchors.json"
