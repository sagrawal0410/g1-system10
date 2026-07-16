#!/usr/bin/env bash
#SBATCH --job-name=actA-ra2final
#SBATCH --partition=defq
#SBATCH --account=liquidai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=00:30:00
#SBATCH --output=/lambdafs/shaurya/g1_sonic_system1/logs/%x-%j.out
# Phase A (hands-in) RA2 reconciliation: re-run the TRAINED RA2_handsin rollout
# on the FINAL checkpoint (050000) and OVERWRITE the 3 RA2_handsin npz. Proxy
# (R0_base_ACT_handsin) is checkpoint-agnostic and left untouched.
# Slurm assigns the GPU -> do NOT set CUDA_VISIBLE_DEVICES.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export HF_HUB_CACHE=/lambdafs/shaurya/hf_cache/hub
export HF_HUB_OFFLINE=0

DATA=/lambdafs/shaurya/g1_sonic_system1/data/g1_act_lerobot_handsB
RA2_CKPT=/lambdafs/shaurya/g1_sonic_system1/outputs/act_handsin_taskcond/checkpoints/050000/pretrained_model
ROLLOUT=$HOME/g1_sonic_system1/scripts/act_rollout.py

echo "[ra2final] node=$(hostname) job=${SLURM_JOB_ID:-NA} ckpt=$RA2_CKPT"
test -f "$RA2_CKPT/model.safetensors" || { echo "RA2 final checkpoint missing" >&2; exit 1; }

python "$ROLLOUT" --label RA2_handsin --checkpoint "$RA2_CKPT" \
    --dataset-root "$DATA" --repo-id shaurya/g1_sonic_act_handsin --hands-in

echo "[ra2final] DONE."
ls -la /lambdafs/shaurya/g1_sonic_system1/results/eval_npz/ | grep 'RA2_handsin' || true
