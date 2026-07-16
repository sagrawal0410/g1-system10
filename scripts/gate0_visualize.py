#!/usr/bin/env python3
"""GATE 0 sanity check: load one episode via LeRobotDataset, decode+save one
image sample, and print the decoded action for that same frame."""
import sys
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = sys.argv[1]
out_png = sys.argv[2]
frame_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 500

ds = LeRobotDataset(repo_id="gate0_check", root=root, video_backend="pyav")
print(f"num_episodes={ds.num_episodes} num_frames={len(ds)} fps={ds.fps}")
print(f"camera_keys={ds.meta.camera_keys}")

sample = ds[frame_idx]
print(f"\n--- Sample at frame_idx={frame_idx} ---")
print(f"task: {sample.get('task', 'n/a')}")
print(f"timestamp: {float(sample['timestamp'])}")

img_t = sample["observation.images.head_cam"]  # CHW float32 in [0,1]
img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
Image.fromarray(img_np).save(out_png)
print(f"saved image: {out_png} shape={img_np.shape}")

for key in ["action.ee_action", "action.robot_q_desired", "action.planner_cmd", "action.token_state"]:
    v = sample[key].numpy()
    print(f"{key}: shape={v.shape} values(first 6)={np.round(v[:6], 4)}")

state = sample["observation.state.robot_q_current"].numpy()
print(f"observation.state.robot_q_current: shape={state.shape} root_xyz={np.round(state[:3],4)}")
