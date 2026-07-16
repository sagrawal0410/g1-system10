#!/usr/bin/env bash
#SBATCH --job-name=act-eval
#SBATCH --partition=defq
#SBATCH --account=liquidai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=01:00:00
#SBATCH --output=/lambdafs/shaurya/g1_sonic_system1/logs/%x-%j.out
set -euo pipefail
# MODE=handsout|handsin ; mints near-zero-step proxy if absent, then rolls out trained + proxy
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export HF_HUB_CACHE=/lambdafs/shaurya/hf_cache/hub
export HF_HUB_OFFLINE=0
MODE="${MODE:-handsout}"
ROLLOUT=$HOME/g1_sonic_system1/scripts/act_rollout.py
if [ "$MODE" = "handsin" ]; then
    DATA="${DATA:-/lambdafs/shaurya/g1_sonic_system1/data/g1_act_lerobot_handsB}"
    RA_CKPT="${RA_CKPT:-/lambdafs/shaurya/g1_sonic_system1/outputs/act_handsin_taskcond/checkpoints/050000/pretrained_model}"
    PROXY_OUT="${PROXY_OUT:-/lambdafs/shaurya/g1_sonic_system1/outputs/act_base_proxy_handsin}"
    TRAINED_LABEL=RA2_handsin; PROXY_LABEL=R0_base_ACT_handsin; REPO=shaurya/g1_sonic_act_handsin
    HANDS_ARGS=(--dataset-root "$DATA" --repo-id "$REPO" --hands-in)
    MINT_MASK=(--policy.hand_active_mask=true --policy.hand_mask_table="$DATA/hand_mask_table.json")
else
    DATA="${DATA:-/lambdafs/shaurya/g1_sonic_system1/data/g1_act_lerobot}"
    RA_CKPT="${RA_CKPT:-/lambdafs/shaurya/g1_sonic_system1/outputs/act_baseline_latent/checkpoints/050000/pretrained_model}"
    PROXY_OUT="${PROXY_OUT:-/lambdafs/shaurya/g1_sonic_system1/outputs/act_base_proxy_handsout}"
    TRAINED_LABEL=RA_handsout; PROXY_LABEL=R0_base_ACT_handsout; REPO=shaurya/g1_sonic_act
    HANDS_ARGS=(); MINT_MASK=()
fi
echo "[act-eval] mode=$MODE job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || true
PROXY_CKPT=$(ls -d "$PROXY_OUT"/checkpoints/[0-9]*/pretrained_model 2>/dev/null | sort | tail -1 || true)
if [ -z "${PROXY_CKPT:-}" ] || [ ! -f "${PROXY_CKPT:-}/model.safetensors" ]; then
    rm -rf "$PROXY_OUT"
    lerobot-train \
      --dataset.repo_id="$REPO" --dataset.root="$DATA" --dataset.video_backend=pyav \
      --dataset.episodes='[1,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21]' \
      --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
      --policy.discrete_latent=true --policy.latent_block='[0,64]' \
      --policy.latent_num_classes=32 --policy.latent_grid_scale=16.0 "${MINT_MASK[@]}" \
      --policy.chunk_size=40 --policy.n_action_steps=40 --policy.vision_backbone=resnet18 \
      --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
      --output_dir="$PROXY_OUT" --job_name=act-base-proxy-$MODE --batch_size=8 \
      --steps=10 --save_freq=10 --log_freq=1 --num_workers=8 --env_eval_freq=0 --wandb.enable=false
    PROXY_CKPT=$(ls -d "$PROXY_OUT"/checkpoints/[0-9]*/pretrained_model 2>/dev/null | sort | tail -1 || true)
fi
python "$ROLLOUT" --label "$TRAINED_LABEL" --checkpoint "$RA_CKPT" "${HANDS_ARGS[@]}"
python "$ROLLOUT" --label "$PROXY_LABEL" --checkpoint "$PROXY_CKPT" "${HANDS_ARGS[@]}"
echo "[act-eval] DONE"
