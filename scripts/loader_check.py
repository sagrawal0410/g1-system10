#!/usr/bin/env python3
"""Pre-launch: verify GR00T's LeRobotEpisodeLoader reads the encoded dataset
through its real torchcodec video path. Reads one TRAIN episode and prints
video/state/action/language shapes."""
import sys
from pathlib import Path

DATASET = sys.argv[1] if len(sys.argv) > 1 else "/lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic"
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # a train episode

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

modality = MODALITY_CONFIGS["unitree_g1_sonic"]
print("modality keys:", {k: v.modality_keys for k, v in modality.items()})

loader = LeRobotEpisodeLoader(dataset_path=DATASET, modality_configs=modality)
print("num episodes in loader:", len(loader))

traj = loader[EP]
print(f"episode {EP}: {len(traj)} steps; columns: {list(traj.columns)}")
import numpy as np
for col in traj.columns:
    v = traj[col].iloc[0]
    try:
        arr = np.asarray(v)
        print(f"  {col}: first-elem shape {arr.shape} dtype {arr.dtype}")
    except Exception:
        print(f"  {col}: {type(v)} value={str(v)[:80]}")
print("LOADER READ OK")
