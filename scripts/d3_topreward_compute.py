#!/usr/bin/env python3
"""Phase D3 — TOPReward zero-shot progress over the G1 3-task demos.

Runs LeRobot's TOPReward (frozen Qwen3-VL-8B-Instruct) over the *video*
LeRobot dataset (g1_act_lerobot, v3.0) restricted to the 19 TRAIN episodes,
producing a per-frame task-progress curve (sparse-dense, 15 anchors/episode,
min-max normalised per episode) — exactly LeRobot's
`lerobot.rewards.topreward.compute_rabc_weights` recipe, but:
  * loads a LOCAL dataset via LeRobotDataset(repo_id="local", root=...),
  * lets us set `image_key` (the packaged CLI hard-codes .top and can't take
    a local path).

Output parquet schema matches upstream: index / episode_index / frame_index /
progress_sparse, with episode_index in VIDEO-dataset space. A separate
post-processing step remaps to encoded_sonic_train space by session.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset
from lerobot.rewards.topreward.configuration_topreward import TOPRewardConfig
from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel
from lerobot.rewards.topreward.processor_topreward import TOPRewardEncoderProcessorStep
from lerobot.rewards.topreward.compute_rabc_weights import (
    compute_instruction_rewards_for_prefixes,
    _resolve_task,
)

# 19 TRAIN episodes in VIDEO-dataset (g1_act_lerobot) index space
# = all 22 minus held-out {0, 2, 16}. Matches results/eval_split.json.
TRAIN_EPS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--image-key", default="observation.images.head_cam")
    ap.add_argument("--vlm-name", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--num-samples", type=int, default=15)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--raw-json", required=True,
                    help="Per-episode raw (unnormalised) anchor log-probs, for distribution analysis.")
    ap.add_argument("--episodes", type=int, nargs="+", default=None,
                    help="Override the default 19 train episodes.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = TOPRewardConfig(vlm_name=args.vlm_name, image_key=args.image_key, fps=args.fps)
    cfg.device = args.device

    logging.info(f"Loading VLM backbone: {cfg.vlm_name}")
    model = TOPRewardModel(cfg).to(args.device).eval()

    encoder = TOPRewardEncoderProcessorStep(
        vlm_name=cfg.vlm_name,
        image_key=cfg.image_key,
        task_key=cfg.task_key,
        default_task=cfg.default_task or "perform the manipulation task",
        max_frames=None,  # prefix length is controlled explicitly
        fps=cfg.fps,
        prompt_prefix=cfg.prompt_prefix,
        prompt_suffix_template=cfg.prompt_suffix_template,
        add_chat_template=cfg.add_chat_template,
        max_length=cfg.max_input_length,
    )

    logging.info(f"Loading dataset: {args.dataset_root}")
    dataset = LeRobotDataset(repo_id="local", root=args.dataset_root, download_videos=False)
    logging.info(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    eps = TRAIN_EPS if args.episodes is None else args.episodes

    all_index, all_episode, all_frame, all_progress = [], [], [], []
    raw = {}

    # Patch compute_instruction_rewards_for_prefixes to also hand back raw
    # anchor log-probs. We reimplement its inner loop here to capture raw.
    for episode_idx in tqdm(eps, desc="Episodes"):
        ep = dataset.meta.episodes[episode_idx]
        ep_start = int(ep["dataset_from_index"])
        ep_end = int(ep["dataset_to_index"])
        num_frames = ep_end - ep_start
        if num_frames <= 0:
            continue

        first_sample = dataset[ep_start]
        task = _resolve_task(first_sample, default=cfg.default_task or "perform the manipulation task")

        # ---- prefix sweep (mirrors compute_instruction_rewards_for_prefixes) ----
        num_samples = args.num_samples
        if num_samples is None or num_samples >= num_frames:
            prefix_lengths = np.arange(1, num_frames + 1, dtype=np.int64)
        else:
            prefix_lengths = np.unique(
                np.linspace(1, num_frames, num_samples).round().astype(np.int64)
            )

        from lerobot.types import TransitionKey
        episode_frames = torch.stack([dataset[ep_start + i][cfg.image_key] for i in range(num_frames)])
        raw_rewards = []
        for length in prefix_lengths:
            frames = episode_frames[: int(length)].unsqueeze(0)
            transition = {
                TransitionKey.OBSERVATION: {cfg.image_key: frames},
                TransitionKey.COMPLEMENTARY_DATA: {"task": task},
            }
            encoded = encoder(transition)
            obs = encoded[TransitionKey.OBSERVATION]
            batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}
            with torch.no_grad():
                r = model.compute_reward(batch)
            raw_rewards.append(float(r.item()))

        raw_arr = np.asarray(raw_rewards, dtype=np.float64)
        r_min, r_max = raw_arr.min(), raw_arr.max()
        if r_max == r_min:
            norm = np.ones_like(raw_arr, dtype=np.float32)
        else:
            norm = ((raw_arr - r_min) / (r_max - r_min)).astype(np.float32)

        if prefix_lengths.shape[0] == num_frames:
            per_frame = norm
        else:
            per_frame = np.interp(
                np.arange(1, num_frames + 1, dtype=np.float64),
                prefix_lengths.astype(np.float64),
                norm.astype(np.float64),
            ).astype(np.float32)

        raw[str(episode_idx)] = {
            "task": task,
            "num_frames": int(num_frames),
            "anchor_prefix_lengths": prefix_lengths.tolist(),
            "anchor_raw_logprob": raw_rewards,
            "raw_min": float(r_min),
            "raw_max": float(r_max),
            "raw_range": float(r_max - r_min),
            "raw_monotonic_frac": float(np.mean(np.diff(raw_arr) >= 0)) if raw_arr.size > 1 else 1.0,
        }

        for local in range(num_frames):
            all_index.append(ep_start + local)
            all_episode.append(episode_idx)
            all_frame.append(local)
            all_progress.append(float(per_frame[local]))

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    table = pa.table({
        "index": np.asarray(all_index, dtype=np.int64),
        "episode_index": np.asarray(all_episode, dtype=np.int64),
        "frame_index": np.asarray(all_frame, dtype=np.int64),
        "progress_sparse": np.asarray(all_progress, dtype=np.float32),
    }).replace_schema_metadata({b"vlm_name": cfg.vlm_name.encode(),
                                b"note": b"episode_index in g1_act_lerobot (video) space"})

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    logging.info(f"Saved {len(table)} frame rows -> {out}")

    Path(args.raw_json).write_text(json.dumps(raw, indent=2))
    logging.info(f"Saved raw anchor log-probs -> {args.raw_json}")

    p = np.asarray(all_progress, dtype=np.float32)
    if p.size:
        logging.info(f"progress_sparse: mean={p.mean():.4f} std={p.std():.4f} "
                     f"min={p.min():.4f} max={p.max():.4f}")


if __name__ == "__main__":
    main()
