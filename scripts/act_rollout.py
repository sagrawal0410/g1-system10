#!/usr/bin/env python3
"""
Phase A -- Open-loop ACT rollout -> npz producer.

Rolls out a LeRobot ACT (discrete-FSQ-latent head) policy on held-out episodes
of the g1_act_lerobot dataset, teacher-forcing observations from the dataset and
re-planning every execution_horizon=40 steps (execute all 40 predicted steps of
each chunk). Writes one npz per (label, episode) in the exact contract expected
by scripts/eval_openloop.py --from-npz.

Also runs a SECOND, proprio-ablated pass (observation.state held at its t=0
value for ALL steps; camera images still teacher-forced) -- the section-1.4
leakage probe.

Run inside the `lerobot` conda env on a GPU (Slurm). Imports lerobot from
~/g1_sonic_system1/repos/lerobot (discrete-FSQ head patch already applied there).

Usage:
    python act_rollout.py --label RA_handsout \
        --checkpoint .../act_baseline_latent/checkpoints/020000/pretrained_model
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("act_rollout")

CAMS = ["head_cam", "left_wrist_cam", "right_wrist_cam"]
GRID_SCALE = 16.0
LATENT_DIM = 64
ACTION_DIM = 78
STATE_DIM = 36
EXEC_HORIZON = 40
NUM_CLASSES = 32

GRID_MIN = -(NUM_CLASSES // 2) / GRID_SCALE
GRID_MAX = (NUM_CLASSES - 1 - NUM_CLASSES // 2) / GRID_SCALE

TASK_ONEHOT_IDX = {
    "bottle_cupnoodles_shelf": 0,
    "cup_wipe_sponge_dryingrack": 1,
    "floor_box_table": 2,
}

HANDSIN_ACTIVE = {
    "bottle_cupnoodles_shelf": {
        "left_hand_joints": [],
        "right_hand_joints": [0, 1, 2, 3, 4],
        "all": list(range(64)) + [71, 72, 73, 74, 75],
    },
    "cup_wipe_sponge_dryingrack": {
        "left_hand_joints": [0],
        "right_hand_joints": [0],
        "all": list(range(64)) + [64, 71],
    },
    "floor_box_table": {
        "left_hand_joints": [0],
        "right_hand_joints": [0],
        "all": list(range(64)) + [64, 71],
    },
}

def load_policy_and_processors(ckpt: str, device: str):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    log.info("Loading ACTPolicy from %s", ckpt)
    policy = ACTPolicy.from_pretrained(ckpt)
    policy.to(device)
    policy.eval()
    if not getattr(policy.config, "discrete_latent", False):
        raise SystemExit("[VALIDATION FAIL] loaded policy has discrete_latent=False -- wrong repo/patch.")
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post

def assert_grid_snapped(latent: np.ndarray, tol: float = 1e-3):
    scaled = latent * GRID_SCALE
    off = np.abs(scaled - np.round(scaled))
    if off.max() > tol:
        raise SystemExit(
            f"[VALIDATION FAIL] latent block not grid-snapped: max off-grid={off.max():.5f} "
            f"(expected integer multiples of 1/{GRID_SCALE:g}). Mis-loaded patched model?"
        )
    if latent.min() < GRID_MIN - tol or latent.max() > GRID_MAX + tol:
        raise SystemExit(
            f"[VALIDATION FAIL] latent out of head range [{GRID_MIN},{GRID_MAX}]: "
            f"min={latent.min():.4f} max={latent.max():.4f}"
        )

def build_obs(img_dict, state_vec):
    """Unbatched CPU observation dict. The preprocessor pipeline adds the batch
    dim (to_batch_processor) and moves to the target device (device_processor)."""
    obs = {}
    for cam in CAMS:
        img = img_dict[cam]
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        obs[f"observation.images.{cam}"] = img.float()
    obs["observation.state"] = torch.as_tensor(np.asarray(state_vec, dtype=np.float32))
    return obs

@torch.no_grad()
def infer_chunk(policy, pre, img_dict, state_vec):
    obs = build_obs(img_dict, state_vec)
    proc = pre(obs)
    chunk = policy.predict_action_chunk(proc)
    return chunk[0].float().cpu().numpy()

def rollout(ds, g0, T, state_joints, policy, pre, ablate_dim):
    """Teacher-forced open-loop rollout, re-planning every EXEC_HORIZON steps.
    The ablated pass holds state[:ablate_dim] at its t=0 value while keeping
    state[ablate_dim:] at the per-step value (for hands-in: proprio [0:36] held,
    task-id one-hot [36:39] kept intact). Cameras are always teacher-forced.
    Returns (pred_action[T,78], pred_action_ablated[T,78])."""
    S = state_joints.shape[1]
    if ablate_dim < 0 or ablate_dim > S:
        ablate_dim = S
    state_t0 = state_joints[0].copy()
    pred = np.zeros((T, ACTION_DIM), np.float32)
    pred_abl = np.zeros((T, ACTION_DIM), np.float32)
    checked = False
    for s in range(0, T, EXEC_HORIZON):
        frame = ds[g0 + s]
        imgs = {cam: frame[f"observation.images.{cam}"] for cam in CAMS}
        chunk_n = infer_chunk(policy, pre, imgs, state_joints[s])
        state_abl = state_joints[s].copy()
        state_abl[:ablate_dim] = state_t0[:ablate_dim]
        chunk_a = infer_chunk(policy, pre, imgs, state_abl)
        if not checked:
            if chunk_n.shape != (EXEC_HORIZON, ACTION_DIM):
                raise SystemExit(f"[VALIDATION FAIL] chunk shape {chunk_n.shape} != {(EXEC_HORIZON, ACTION_DIM)}")
            assert_grid_snapped(chunk_n[:, :LATENT_DIM])
            log.info("  grid-snap check OK (latent block are integer multiples of 1/%g).", GRID_SCALE)
            checked = True
        L = min(EXEC_HORIZON, T - s)
        pred[s : s + L] = chunk_n[:L]
        pred_abl[s : s + L] = chunk_a[:L]
    return pred, pred_abl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to a lerobot ACT pretrained_model dir")
    ap.add_argument("--label", required=True, help="npz label, e.g. RA_handsout / R0_base_ACT_handsout")
    ap.add_argument("--dataset-root", default="/lambdafs/shaurya/g1_sonic_system1/data/g1_act_lerobot")
    ap.add_argument("--repo-id", default="shaurya/g1_sonic_act")
    ap.add_argument("--eval-split", default="/home/shaurya/g1_sonic_system1/results/eval_split.json")
    ap.add_argument("--out-dir", default="/lambdafs/shaurya/g1_sonic_system1/results/eval_npz")
    ap.add_argument("--cap", type=int, default=3000, help="max steps per episode")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hands-in", action="store_true",
                    help="task-conditioned hands-in mode: 39-dim state ([36:39]=one-hot task-id), "
                         "sanity-check the one-hot, and emit hand_active_local_dims into meta.")
    ap.add_argument("--ablate-proprio-dim", type=int, default=-1,
                    help="ablated pass holds state[:N] at t0, keeps state[N:] intact. "
                         "-1 (default) => hold the ENTIRE state at t0 (hands-out). For hands-in use 36.")
    args = ap.parse_args()

    ablate_dim = args.ablate_proprio_dim
    if args.hands_in and ablate_dim < 0:
        ablate_dim = 36

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with open(args.eval_split) as f:
        eval_eps = json.load(f)["eval"]

    log.info("Opening dataset %s (root=%s)", args.repo_id, args.dataset_root)
    ds = LeRobotDataset(args.repo_id, root=args.dataset_root, video_backend="pyav")
    ei = np.asarray(ds.hf_dataset["episode_index"])

    policy, pre, _post = load_policy_and_processors(args.checkpoint, args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for ep in eval_eps:
        e = int(ep["episode_index"])
        task = ep["task"]
        idx = np.nonzero(ei == e)[0]
        if len(idx) == 0:
            raise SystemExit(f"episode_index {e} not found in dataset")
        g0 = int(idx.min())
        ep_len = int(len(idx))
        if int(idx.max()) - g0 + 1 != ep_len:
            raise SystemExit(f"episode {e} frames are not contiguous in the dataset -- cannot slice")
        T = min(ep_len, args.cap)
        capped = T < ep_len
        log.info("=== Episode %d (%s): global[%d..%d] len=%d T=%d%s ===",
                 e, task, g0, g0 + ep_len - 1, ep_len, T, "  [CAPPED]" if capped else "")

        sub = ds.hf_dataset.select(range(g0, g0 + T))
        gt_action = np.asarray([np.asarray(a, np.float32) for a in sub["action"]], np.float32)
        state_joints = np.asarray([np.asarray(a, np.float32) for a in sub["observation.state"]], np.float32)
        S = state_joints.shape[1]
        if gt_action.shape != (T, ACTION_DIM):
            raise SystemExit(f"gt_action shape {gt_action.shape} != {(T, ACTION_DIM)}")
        if state_joints.shape[0] != T:
            raise SystemExit(f"state_joints rows {state_joints.shape[0]} != T={T}")

        if args.hands_in:
            if S != 39:
                raise SystemExit(f"[VALIDATION FAIL] hands-in expects 39-dim state, got {S}")
            if task not in TASK_ONEHOT_IDX:
                raise SystemExit(f"[VALIDATION FAIL] unknown hands-in task '{task}'")
            oh = state_joints[:, 36:39]
            exp = np.zeros(3, np.float32); exp[TASK_ONEHOT_IDX[task]] = 1.0
            if not np.allclose(oh, exp[None, :], atol=1e-5):
                raise SystemExit(f"[VALIDATION FAIL] one-hot[36:39] for ep{e} ({task}) "
                                 f"not constant {exp.tolist()}; sample={oh[0].tolist()}")
            log.info("  one-hot task-id check OK: [36:39]==%s for all %d steps.", exp.tolist(), T)

        pred, pred_abl = rollout(ds, g0, T, state_joints, policy, pre, ablate_dim)

        latent_mse = float(np.mean((pred[:, :LATENT_DIM] - gt_action[:, :LATENT_DIM]) ** 2))
        lh_mse = float(np.mean((pred[:, 64:71] - gt_action[:, 64:71]) ** 2))
        rh_mse = float(np.mean((pred[:, 71:78] - gt_action[:, 71:78]) ** 2))
        abl_mse = float(np.mean((pred_abl[:, :LATENT_DIM] - gt_action[:, :LATENT_DIM]) ** 2))

        meta = dict(checkpoint=args.label, task=task, episode_id=str(e),
                    plot_rank=0, execution_horizon=EXEC_HORIZON)
        if args.hands_in:
            meta["hand_active_local_dims"] = HANDSIN_ACTIVE[task]
        path = out_dir / f"{args.label}__{task}__ep{e}.npz"
        np.savez(
            path,
            pred_action=pred.astype(np.float32),
            gt_action=gt_action.astype(np.float32),
            pred_action_ablated=pred_abl.astype(np.float32),
            state_joints=state_joints.astype(np.float32),
            meta=meta,
        )

        d = np.load(path, allow_pickle=True)
        for k in ["pred_action", "gt_action", "pred_action_ablated", "state_joints"]:
            if d[k].dtype != np.float32:
                raise SystemExit(f"{k} dtype {d[k].dtype} != float32")
            if np.isnan(d[k]).any():
                raise SystemExit(f"NaN found in {k}")
        assert d["pred_action"].shape == d["gt_action"].shape == d["pred_action_ablated"].shape == (T, ACTION_DIM)
        assert d["state_joints"].shape == (T, S)
        rmeta = d["meta"].item()
        assert "meta" in d and rmeta["checkpoint"] == args.label
        if args.hands_in:
            assert "hand_active_local_dims" in rmeta, "hand_active_local_dims missing from meta"
        log.info("  WROTE %s", path.name)
        log.info("  motion_token MSE=%.6f  left_hand MSE=%.6f  right_hand MSE=%.6f  latent_ablated MSE=%.6f",
                 latent_mse, lh_mse, rh_mse, abl_mse)
        summary.append((task, e, T, capped, latent_mse, abl_mse))

    log.info("========== SUMMARY (label=%s) ==========", args.label)
    for task, e, T, capped, lm, am in summary:
        log.info("  ep%-3d %-28s T=%-5d%s  motion_token_MSE=%.6f  ablated_MSE=%.6f",
                 e, task, T, " CAP" if capped else "   ", lm, am)

if __name__ == "__main__":
    main()
