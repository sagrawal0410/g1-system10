#!/usr/bin/env bash
#SBATCH --job-name=phaseD-eval
#SBATCH --partition=defq
#SBATCH --account=liquidai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=03:00:00
#SBATCH --output=/lambdafs/shaurya/g1_sonic_system1/logs/%x-%j.out
# Phase D OPEN-LOOP NPZ rollouts (D1 stitching + D2 best-of-N+FSQ) on a
# fine-tuned GR00T checkpoint. 1 GPU; does NOT touch the training GPUs.
# Submit with:
#   CKPT=/path/to/checkpoint-20000 STEP=20000 NPZ_POLICY=R1_handsout sbatch sbatch_phase_d_eval.sbatch
set -euo pipefail

CKPT="${CKPT:?set CKPT=/path/to/checkpoint-N}"
STEP="${STEP:-unknown}"
POLICY="${POLICY:-GR00T-A}"
POLICY_TYPE="${POLICY_TYPE:-groot}"
NPZ_POLICY="${NPZ_POLICY:-$POLICY}"
STEPS_CAP="${STEPS_CAP:-3000}"
EPISODES="${EPISODES:-0,2,16}"
DATASET="${DATASET:-/lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic}"
OUTDIR="${OUTDIR:-/home/shaurya/g1_sonic_system1/results/phase_d/parts/$POLICY}"
NPZDIR="${NPZDIR:-/lambdafs/shaurya/g1_sonic_system1/results/eval_npz}"

export HF_HUB_CACHE=/lambdafs/shaurya/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
conda activate groot
# torchcodec video decode needs conda's ffmpeg-7 libs on the loader path.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export ACTION_LAYOUT_PATH=/home/shaurya/g1_sonic_system1/results/action_layout.json
export ISAAC_GR00T_REPO=/home/shaurya/g1_sonic_system1/repos/Isaac-GR00T

echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID gpus=${SLURM_GPUS_PER_NODE:-?} ckpt=$CKPT step=$STEP"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || true

cd ~/g1_sonic_system1/scripts
python -m phase_d.run_phase_d_experiments \
    --checkpoint "$CKPT" \
    --checkpoint-step "$STEP" \
    --policy "$POLICY" \
    --policy-type "$POLICY_TYPE" \
    --npz-policy "$NPZ_POLICY" \
    --dataset-path "$DATASET" \
    --episodes "$EPISODES" \
    --steps "$STEPS_CAP" \
    --output-dir "$OUTDIR" \
    --npz-dir "$NPZDIR" \
    --device cuda:0

echo "[sbatch] DONE"
