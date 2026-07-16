#!/usr/bin/env bash
#SBATCH --job-name=phaseD-eval-act
#SBATCH --partition=defq
#SBATCH --account=liquidai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=03:00:00
#SBATCH --output=/lambdafs/shaurya/g1_sonic_system1/logs/%x-%j.out
# Phase D OPEN-LOOP NPZ rollouts for an ACT (LeRobot CVAE) policy. 1 GPU.
# Submit with:  CKPT=/path/to/checkpoints/NNNNNN STEP=NNNNNN NPZ_POLICY=RA_handsout \
#               DATASET=/path/to/g1_act_lerobot [ABLATE_DIMS=36] sbatch sbatch_phase_d_eval_act.sbatch
set -euo pipefail

CKPT="${CKPT:?set CKPT=/path/to/act/checkpoints/NNNNNN}"
STEP="${STEP:-unknown}"
POLICY="${POLICY:-ACT-A}"
NPZ_POLICY="${NPZ_POLICY:-$POLICY}"
ABLATE_DIMS="${ABLATE_DIMS:--1}"
STEPS_CAP="${STEPS_CAP:-3000}"
EPISODES="${EPISODES:-0,2,16}"
DATASET="${DATASET:-/lambdafs/shaurya/g1_sonic_system1/data/g1_act_lerobot}"
OUTDIR="${OUTDIR:-/home/shaurya/g1_sonic_system1/results/phase_d/parts/$POLICY}"
NPZDIR="${NPZDIR:-/lambdafs/shaurya/g1_sonic_system1/results/eval_npz}"

export HF_HUB_CACHE=/lambdafs/shaurya/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/shaurya/g1_sonic_system1/repos/lerobot/src:${PYTHONPATH:-}"

echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID gpus=${SLURM_GPUS_PER_NODE:-?} ckpt=$CKPT step=$STEP policy=$POLICY"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || true

cd ~/g1_sonic_system1/scripts
python -m phase_d.run_phase_d_experiments \
    --checkpoint "$CKPT" \
    --checkpoint-step "$STEP" \
    --policy "$POLICY" \
    --policy-type act \
    --npz-policy "$NPZ_POLICY" \
    --ablate-proprio-dims "$ABLATE_DIMS" \
    --dataset-path "$DATASET" \
    --episodes "$EPISODES" \
    --steps "$STEPS_CAP" \
    --output-dir "$OUTDIR" \
    --npz-dir "$NPZDIR" \
    --device cuda:0

echo "[sbatch] DONE"
