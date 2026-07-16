#!/usr/bin/env bash
#SBATCH --job-name=d3
#SBATCH --partition=defq
#SBATCH --account=liquidai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=04:00:00
#SBATCH --output=/lambdafs/shaurya/g1_sonic_system1/logs/%x-%j.out
set -euo pipefail
# STAGE=reward (1 GPU, lerobot) | STAGE=train (2 GPUs, groot; sbatch --gpus-per-node=2; needs RABC_WEIGHTS_JSON)
STAGE="${STAGE:-reward}"
export HF_HUB_CACHE=/lambdafs/shaurya/hf_cache/hub
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
source ~/miniconda3/etc/profile.d/conda.sh
if [ "$STAGE" = "reward" ]; then
    conda activate lerobot
    export LD_LIBRARY_PATH="/home/shaurya/miniconda3/envs/groot/lib:${LD_LIBRARY_PATH:-}"
    OUTDIR=/lambdafs/shaurya/g1_sonic_system1/results/phase_d3; mkdir -p "$OUTDIR"
    cd ~/g1_sonic_system1/repos/lerobot
    exec python /home/shaurya/g1_sonic_system1/scripts/d3_topreward_compute.py \
        --dataset-root /lambdafs/shaurya/g1_sonic_system1/data/g1_act_lerobot \
        --image-key observation.images.head_cam --vlm-name Qwen/Qwen3-VL-8B-Instruct \
        --num-samples 15 --fps 2.0 --device cuda \
        --output-path "$OUTDIR/topreward_progress_video.parquet" \
        --raw-json "$OUTDIR/topreward_raw_anchors.json"
fi
conda activate groot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export ACTION_LAYOUT_PATH=/home/shaurya/g1_sonic_system1/results/action_layout.json
export RABC_WEIGHTS_JSON="${RABC_WEIGHTS_JSON:?set RABC_WEIGHTS_JSON}"
export RABC_FALLBACK_WEIGHT="${RABC_FALLBACK_WEIGHT:-1.0}"
export WANDB_PROJECT=g1-sonic-vla; export WANDB__SERVICE_WAIT=300
BASE_CKPT="${BASE_CKPT:-/lambdafs/shaurya/g1_sonic_system1/outputs/grootN17-sft-handsout/grootN17-sft-handsout/checkpoint-20000}"
DATASET="${DATASET:-/lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic_train}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grootN17-sft-T3-rabc}"
OUTDIR="/lambdafs/shaurya/g1_sonic_system1/outputs/${EXPERIMENT_NAME}"
NUM_GPUS="${NUM_GPUS:-2}"
cd ~/g1_sonic_system1/repos/Isaac-GR00T
exec torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    /home/shaurya/g1_sonic_system1/scripts/d3_launch_sft_rabc.py \
    --base_model_path "$BASE_CKPT" --dataset_path "$DATASET" \
    --embodiment_tag UNITREE_G1_SONIC --num_gpus "$NUM_GPUS" --output_dir "$OUTDIR" \
    --experiment_name "$EXPERIMENT_NAME" --wandb_project "$WANDB_PROJECT" \
    --save_steps "${SAVE_STEPS:-500}" --save_total_limit 5 --max_steps "${MAX_STEPS:-1500}" \
    --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate "${LR:-2e-5}" --use_wandb \
    --global_batch_size "${GLOBAL_BATCH_SIZE:-20}" --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
    --shard_size 1024 --num_shards_per_epoch 100000 --episode_sampling_rate 0.1 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
