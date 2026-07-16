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
set -euo pipefail
# POLICY_TYPE=groot|act ; set CKPT, POLICY, STEP. act also uses ABLATE_DIMS.
CKPT="${CKPT:?set CKPT}"
STEP="${STEP:-unknown}"
POLICY="${POLICY:-GR00T-A}"
POLICY_TYPE="${POLICY_TYPE:-groot}"
NPZ_POLICY="${NPZ_POLICY:-$POLICY}"
STEPS_CAP="${STEPS_CAP:-3000}"
EPISODES="${EPISODES:-0,2,16}"
ABLATE_DIMS="${ABLATE_DIMS:--1}"
OUTDIR="${OUTDIR:-/home/shaurya/g1_sonic_system1/results/phase_d/parts/$POLICY}"
NPZDIR="${NPZDIR:-/lambdafs/shaurya/g1_sonic_system1/results/eval_npz}"
export HF_HUB_CACHE=/lambdafs/shaurya/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
if [ "$POLICY_TYPE" = "act" ]; then
    conda activate lerobot
    export PYTHONPATH="/home/shaurya/g1_sonic_system1/repos/lerobot/src:${PYTHONPATH:-}"
    DATASET="${DATASET:-/lambdafs/shaurya/g1_sonic_system1/data/g1_act_lerobot}"
else
    conda activate groot
    export ISAAC_GR00T_REPO=/home/shaurya/g1_sonic_system1/repos/Isaac-GR00T
    DATASET="${DATASET:-/lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic}"
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export ACTION_LAYOUT_PATH=/home/shaurya/g1_sonic_system1/results/action_layout.json
echo "[phaseD-eval] job=${SLURM_JOB_ID:-NA} type=$POLICY_TYPE ckpt=$CKPT step=$STEP"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || true
cd ~/g1_sonic_system1/scripts
EXTRA=()
[ "$POLICY_TYPE" = "act" ] && EXTRA=(--ablate-proprio-dims "$ABLATE_DIMS")
python -m phase_d.run_phase_d_experiments \
    --checkpoint "$CKPT" --checkpoint-step "$STEP" --policy "$POLICY" \
    --policy-type "$POLICY_TYPE" --npz-policy "$NPZ_POLICY" "${EXTRA[@]}" \
    --dataset-path "$DATASET" --episodes "$EPISODES" --steps "$STEPS_CAP" \
    --output-dir "$OUTDIR" --npz-dir "$NPZDIR" --device cuda:0
echo "[phaseD-eval] DONE"
