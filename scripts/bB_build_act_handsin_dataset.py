#!/usr/bin/env python3
"""
Phase B (hands-in): build a lerobot-native v3 ACT dataset whose single unified
`action[78]` = token_state[64] + PER-TASK-NORMALIZED hand_cmd_pico[14], and whose
`observation.state[39]` = robot_q_current[36] + 3-dim one-hot task-id (task
conditioning, since ACT has no language input).

This is the ACT analog of the GR00T-B hands-in run. It does NOT overwrite the
hands-out ACT dataset (g1_act_lerobot). Output -> g1_act_lerobot_handsB.

Design (matches coordinator spec):
  action[0:64]   = action.token_state                         (raw FSQ-grid; ACTION norm forced IDENTITY at train)
  action[64:78]  = normalized hand_cmd_pico                    (per-task min-max on ACTIVE dims; inactive -> 0)
                   pico[0:7]->action[64:71] (left), pico[7:14]->action[71:78] (right)
  observation.state[0:36]  = observation.state.robot_q_current (task-invariant)
  observation.state[36:39] = one-hot(folder): bottle=idx0, cup=idx1, floor=idx2
  observation.images.{head,left_wrist,right_wrist}_cam = native 480x640x3 videos

Reuses the EXACT params from hand_normalization.json (per-task, active dims only,
inactive -> 0; NOT merged across tasks). Emits:
  - episode_mapping.json (episode_index -> folder/session/instruction; state+action layout)
  - hand_mask_table.json (DATASET task_index -> active continuous-hand dims, for
    per-task active-dim loss masking; keyed by the dataset's own task_index, which
    is derived AFTER build because the bottle task has 2 instruction variants).

No GPU. Run inside the `lerobot` conda env.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bB_handsin")

TOKEN_COL = "action.token_state"
STATE_COL = "observation.state.robot_q_current"
PICO_COL = "action.hand_cmd_pico"
CAMS = ["head_cam", "left_wrist_cam", "right_wrist_cam"]
TOKEN_DIM = 64
HAND_DIM = 14
ROBOT_Q_DIM = 36
N_TASKS = 3
STATE_DIM = ROBOT_Q_DIM + N_TASKS
ACTION_DIM = TOKEN_DIM + HAND_DIM
CONT_OFFSET = TOKEN_DIM
IMG_H, IMG_W = 480, 640
FPS = 30
ROBOT_TYPE = "unitree_g1"

FOLDER_ONEHOT = {
    "bottle_cupnoodles_shelf": 0,
    "cup_wipe_sponge_dryingrack": 1,
    "floor_box_table": 2,
}

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": {"axes": [f"q_{i}" for i in range(ROBOT_Q_DIM)]
                  + ["task_bottle", "task_cup", "task_floor"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
        "names": {
            "axes": [f"motion_token_{i}" for i in range(TOKEN_DIM)]
            + [f"left_hand_{i}" for i in range(7)]
            + [f"right_hand_{i}" for i in range(7)]
        },
    },
}
for _cam in CAMS:
    FEATURES[f"observation.images.{_cam}"] = {
        "dtype": "video",
        "shape": (IMG_H, IMG_W, 3),
        "names": ["height", "width", "channels"],
    }

def find_sessions(raw_root: Path):
    out = []
    for task in sorted(os.listdir(raw_root)):
        tdir = raw_root / task
        if not tdir.is_dir():
            continue
        for sess in sorted(os.listdir(tdir)):
            sdir = tdir / sess
            if (sdir / "meta" / "info.json").exists():
                out.append((task, sess, sdir))
    return out

def load_session_parquet(sdir: Path) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(sdir / "data" / "chunk-*" / "file-*.parquet")))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def session_instruction(sdir: Path, df: pd.DataFrame) -> str:
    tp = sdir / "meta" / "tasks.parquet"
    if tp.exists():
        tdf = pd.read_parquet(tp)
        if "task" in tdf.columns and len(tdf) > 0:
            if "task_index" in df.columns and "task_index" in tdf.columns:
                ti = int(df["task_index"].iloc[0])
                row = tdf[tdf["task_index"] == ti]
                if len(row) > 0:
                    return str(row["task"].iloc[0])
            return str(tdf["task"].iloc[0])
    return sdir.parent.name

def normalize_hand(pico_rows: np.ndarray, norm_entry: dict) -> np.ndarray:
    """pico_rows (n,14) -> normalized hand (n,14); active dims min-max to [0,1], inactive->0."""
    n = pico_rows.shape[0]
    h = np.zeros((n, HAND_DIM), dtype=np.float32)
    active = norm_entry["active_action_dims"]
    pico_active = norm_entry["pico_active_dims"]
    mins = norm_entry["min"]
    maxs = norm_entry["max"]
    assert len(active) == len(pico_active) == len(mins) == len(maxs)
    for i, act_dim in enumerate(active):
        pd_idx = pico_active[i]
        assert act_dim - CONT_OFFSET == pd_idx, f"act_dim {act_dim} vs pico {pd_idx}"
        lo, hi = float(mins[i]), float(maxs[i])
        rng = hi - lo
        vals = pico_rows[:, pd_idx].astype(np.float32)
        h[:, pd_idx] = 0.0 if rng <= 0 else np.clip((vals - lo) / rng, 0.0, 1.0)
    return h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="/lambdafs/shaurya/g1_sonic_system1/data/g1_raw_full")
    ap.add_argument("--out-root", default="/lambdafs/shaurya/g1_sonic_system1/data/g1_act_lerobot_handsB")
    ap.add_argument("--hand-norm", default="/lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic_handsB/meta/hand_normalization.json")
    ap.add_argument("--repo-id", default="shaurya/g1_sonic_act_handsin")
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--limit-episodes", type=int, default=0)
    args = ap.parse_args()

    import decord
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"Output {out_root} already exists and is non-empty. Refusing to overwrite.")

    with open(args.hand_norm) as f:
        HN = json.load(f)["tasks"]
    log.info("hand_normalization tasks: %s", list(HN.keys()))

    sessions = find_sessions(raw_root)
    log.info("Found %d session dirs under %s", len(sessions), raw_root)

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=FEATURES,
        root=str(out_root),
        robot_type=ROBOT_TYPE,
        use_videos=True,
        video_backend=args.video_backend,
    )

    mapping = []
    ep_idx = 0
    included_tasks = {}
    skipped = []
    instr_to_folder = {}

    for task, sess, sdir in sessions:
        df = load_session_parquet(sdir)
        if df is None:
            skipped.append((task, sess, "no-parquet")); continue
        if TOKEN_COL not in df.columns:
            skipped.append((task, sess, "no-token_state")); continue
        if STATE_COL not in df.columns:
            skipped.append((task, sess, "no-robot_q_current")); continue
        if PICO_COL not in df.columns:
            skipped.append((task, sess, "no-hand_cmd_pico")); continue
        if task not in HN:
            skipped.append((task, sess, "no-hand-norm-entry")); continue
        if task not in FOLDER_ONEHOT:
            skipped.append((task, sess, "no-onehot-mapping")); continue

        vids = {}
        ok = True
        for cam in CAMS:
            vp = sorted(glob.glob(str(sdir / "videos" / f"observation.images.{cam}" / "chunk-*" / "file-*.mp4")))
            if not vp:
                ok = False; break
            vids[cam] = vp[0]
        if not ok:
            skipped.append((task, sess, "missing-video")); continue

        import decord

        n = len(df)
        tok = np.stack(df[TOKEN_COL].to_numpy()).astype(np.float32)
        rqc = np.stack(df[STATE_COL].to_numpy()).astype(np.float32)
        pico = np.stack(df[PICO_COL].to_numpy()).astype(np.float32)
        assert tok.shape == (n, TOKEN_DIM), f"{sess}: token {tok.shape}"
        assert rqc.shape == (n, ROBOT_Q_DIM), f"{sess}: state {rqc.shape}"
        assert pico.shape == (n, HAND_DIM), f"{sess}: pico {pico.shape}"
        assert np.isfinite(tok).all() and np.isfinite(rqc).all() and np.isfinite(pico).all(), f"{sess}: non-finite"

        hand = normalize_hand(pico, HN[task])
        onehot = np.zeros((N_TASKS,), dtype=np.float32)
        onehot[FOLDER_ONEHOT[task]] = 1.0

        readers = {cam: decord.VideoReader(vids[cam]) for cam in CAMS}
        for cam, vr in readers.items():
            if len(vr) != n:
                raise SystemExit(f"{sess}/{cam}: video frames {len(vr)} != parquet rows {n}")

        instr = session_instruction(sdir, df)
        if instr in instr_to_folder and instr_to_folder[instr] != task:
            raise SystemExit(f"instruction reused across folders: {instr!r} in {instr_to_folder[instr]} and {task}")
        instr_to_folder[instr] = task

        for i in range(n):
            frame = {
                "observation.state": np.concatenate([rqc[i], onehot]).astype(np.float32),
                "action": np.concatenate([tok[i], hand[i]]).astype(np.float32),
                "task": instr,
            }
            for cam in CAMS:
                frame[f"observation.images.{cam}"] = readers[cam][i].asnumpy()
            ds.add_frame(frame)
        ds.save_episode()

        mapping.append({
            "episode_index": ep_idx, "task": task, "source_session": sess,
            "n_frames": int(n), "instruction": instr,
            "onehot_index": FOLDER_ONEHOT[task],
            "active_action_dims": HN[task]["active_action_dims"],
        })
        included_tasks[task] = included_tasks.get(task, 0) + 1
        log.info("[ep %d] %s / %s : %d frames (onehot=%d, hand-active=%s)",
                 ep_idx, task, sess, n, FOLDER_ONEHOT[task], HN[task]["active_action_dims"])
        ep_idx += 1
        del readers
        if args.limit_episodes and ep_idx >= args.limit_episodes:
            log.warning("Stopping early after %d episodes (debug limit).", ep_idx)
            break

    ds.finalize()

    meta = LeRobotDatasetMetadata(args.repo_id, root=str(out_root))
    ti_series = meta.tasks["task_index"]
    mask_table = {}
    ti_to_folder = {}
    for task_str, ti in ti_series.items():
        if task_str not in instr_to_folder:
            raise SystemExit(f"dataset task string not seen during build: {task_str!r}")
        folder = instr_to_folder[task_str]
        rel_active = [int(d) - CONT_OFFSET for d in HN[folder]["active_action_dims"]]
        mask_table[str(int(ti))] = rel_active
        ti_to_folder[str(int(ti))] = folder
    log.info("hand_mask_table (dataset task_index -> active cont dims): %s", mask_table)

    with open(out_root / "hand_mask_table.json", "w") as f:
        json.dump({
            "n_cont": HAND_DIM, "cont_offset": CONT_OFFSET,
            "table": mask_table, "task_index_to_folder": ti_to_folder,
            "note": "Per-task active-dim hand-loss mask. table[task_index] = list of RELATIVE indices "
                    "into the 14-dim continuous hand block (action[64:78]) that carry real signal. "
                    "Absolute action dim = 64 + relative. Used by ACT --policy.hand_active_mask.",
        }, f, indent=2)

    with open(out_root / "episode_mapping.json", "w") as f:
        json.dump({
            "repo_id": args.repo_id, "raw_root": str(raw_root),
            "action_dim": ACTION_DIM,
            "action_layout": {"motion_token": [0, 64], "left_hand_joints": [64, 71], "right_hand_joints": [71, 78]},
            "hands_zeroed": False,
            "hand_source": PICO_COL,
            "hand_normalization": args.hand_norm,
            "state_dim": STATE_DIM,
            "state_layout": {"robot_q_current": [0, 36], "task_onehot": [36, 39]},
            "task_onehot_mapping": FOLDER_ONEHOT,
            "state_source": STATE_COL,
            "fps": FPS, "total_episodes": ep_idx,
            "included_task_counts": included_tasks,
            "task_index_to_folder": ti_to_folder,
            "skipped": [{"task": t, "session": s, "reason": r} for (t, s, r) in skipped],
            "episodes": mapping,
        }, f, indent=2)

    log.info("DONE. episodes=%d included_tasks=%s", ep_idx, included_tasks)
    for t, s, r in skipped:
        log.info("   skip %s/%s (%s)", t, s, r)

    log.info("VALIDATE: total_episodes=%d total_frames=%d action_shape=%s state_shape=%s",
             meta.total_episodes, meta.total_frames,
             meta.features["action"]["shape"], meta.features["observation.state"]["shape"])
    for e in range(meta.total_episodes):
        dp = meta.root / meta.get_data_file_path(e)
        if not dp.exists():
            raise SystemExit(f"missing parquet ep {e}: {dp}")
        for vk in meta.video_keys:
            vp = meta.root / meta.get_video_file_path(e, vk)
            if not vp.exists():
                raise SystemExit(f"missing video ep {e} {vk}: {vp}")
    log.info("VALIDATE OK: all parquet + video present for %d episodes.", meta.total_episodes)

if __name__ == "__main__":
    main()
