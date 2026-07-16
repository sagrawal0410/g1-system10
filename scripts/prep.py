#!/usr/bin/env python3
"""Phase 0 / 0.5 data prep: python prep.py {triage,patch,validate,split,encode} [args...]

  triage   <data-root>                         read-only per-task dataset triage
  patch    <data-root>                         fill LeRobot-v3 meta indices the WBC exporter omits
  validate [data-root]                         check action.token_state is real (not zero placeholder)
  split    <data-root> <out.json>              hold out ~10% episodes per task (seed 42)
  encode   [--hands-mode {zero,pico}] [--out D] [--exclude i,j]   build GR00T-LeRobot v2.1 encoded dataset

encode action(78) = motion_token(64)<-action.token_state | left_hand(7) | right_hand(7).
  zero: hands=0 (body/latent only). pico: per-task per-dim min-max of action.hand_cmd_pico on active dims.
"""
import argparse
import glob
import json
import shutil
import sys
from collections import defaultdict  # noqa: F401
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

FPS_DEFAULT = 30
SEED = 42


# ---------------- triage ----------------
def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_episode_dirs(task_dir):
    return [p for p in sorted(task_dir.iterdir()) if p.is_dir() and (p / "meta" / "info.json").exists()]


def summarize_features(features):
    obs_state, obs_image, obs_depth, actions, other = {}, {}, {}, {}, {}
    for key, spec in features.items():
        names = spec.get("names")
        if isinstance(names, dict):
            names = names.get("axes")
        entry = {"dtype": spec.get("dtype"), "shape": spec.get("shape"), "names": names}
        if key.startswith("observation.images."):
            obs_image[key] = entry
        elif key.startswith("observation.depths."):
            obs_depth[key] = entry
        elif key.startswith("observation."):
            obs_state[key] = entry
        elif key.startswith("action."):
            actions[key] = entry
        else:
            other[key] = entry
    return obs_state, obs_image, obs_depth, actions, other


def _sem_groups(modality, section):
    return {k: (v["start"], v["end"]) for k, v in modality.get(section, {}).items()}


def triage_task(task_dir):
    report = {"task_name": task_dir.name, "num_session_dirs": 0, "total_episodes": 0, "total_frames": 0,
              "codebase_versions": set(), "robot_types": set(), "fps_values": set(), "cameras": {}, "depths": {},
              "state_features": {}, "action_features": {}, "state_semantic_groups": {}, "action_semantic_groups": {},
              "task_descriptions": set(), "errors": []}
    episode_dirs = find_episode_dirs(task_dir)
    report["num_session_dirs"] = len(episode_dirs)
    for ep_dir in episode_dirs:
        try:
            info = load_json(ep_dir / "meta" / "info.json")
        except Exception as e:
            report["errors"].append(f"{ep_dir.name}: info.json ({e})"); continue
        report["codebase_versions"].add(info.get("codebase_version", "unknown"))
        report["robot_types"].add(info.get("robot_type", "unknown"))
        report["fps_values"].add(info.get("fps"))
        report["total_episodes"] += info.get("total_episodes", 0)
        report["total_frames"] += info.get("total_frames", 0)
        obs_state, obs_image, obs_depth, actions, _ = summarize_features(info.get("features", {}))
        report["state_features"].update(obs_state); report["action_features"].update(actions)
        for cam_key, spec in obs_image.items():
            shape = spec["shape"]
            report["cameras"][cam_key] = {"resolution": f"{shape[1]}x{shape[0]}" if shape else "unknown",
                                          "channels": shape[2] if shape and len(shape) > 2 else None, "fps": info.get("fps")}
        for depth_key in obs_depth:
            report["depths"][depth_key] = {"fps": info.get("fps")}
        modality_path = ep_dir / "meta" / "modality.json"
        if modality_path.exists():
            try:
                modality = load_json(modality_path)
                report["state_semantic_groups"].update(_sem_groups(modality, "state"))
                report["action_semantic_groups"].update(_sem_groups(modality, "action"))
            except Exception as e:
                report["errors"].append(f"{ep_dir.name}: modality.json ({e})")
        tasks_parquet = ep_dir / "meta" / "tasks.parquet"
        if tasks_parquet.exists() and pd is not None:
            try:
                tdf = pd.read_parquet(tasks_parquet)
                for col in ("task", "task_description", "language_instruction"):
                    if col in tdf.columns:
                        report["task_descriptions"].update(tdf[col].astype(str).tolist()); break
            except Exception as e:
                report["errors"].append(f"{ep_dir.name}: tasks.parquet ({e})")
    return report


def _print_report(r):
    print(f"\n{'=' * 70}\nTASK: {r['task_name']}\n{'=' * 70}")
    print(f"  session dirs: {r['num_session_dirs']}  episodes: {r['total_episodes']}  frames: {r['total_frames']}")
    print(f"  codebase_version(s): {sorted(r['codebase_versions'])}  robot_type(s): {sorted(r['robot_types'])}")
    print(f"  fps: {sorted(v for v in r['fps_values'] if v is not None)}")
    if r["task_descriptions"]:
        print(f"  task descriptions: {sorted(r['task_descriptions'])}")
    print(f"  Cameras ({len(r['cameras'])}):")
    for cam, spec in r["cameras"].items():
        print(f"    - {cam}: {spec['resolution']} @ {spec['fps']} fps, channels={spec['channels']}")
    print(f"  State features ({len(r['state_features'])}):")
    for key, spec in sorted(r["state_features"].items()):
        print(f"    - {key}: {spec['dtype']} shape={spec['shape']}")
    print(f"  Action features ({len(r['action_features'])}):")
    for key, spec in sorted(r["action_features"].items()):
        print(f"    - {key}: {spec['dtype']} shape={spec['shape']}")
    ak = set(r["action_features"])
    print("  action-space:", {"joint": "action.robot_q_desired" in ak, "ee": "action.ee_action" in ak,
                               "planner": "action.planner_cmd" in ak, "token": "action.token_state" in ak})
    for e in r["errors"]:
        print(f"    ! {e}")


def cmd_triage(argv):
    ap = argparse.ArgumentParser(prog="prep.py triage")
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args(argv)
    root = Path(args.data_root).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr); sys.exit(1)
    task_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    print(f"G1 Dataset Triage: {root}  tasks: {[t.name for t in task_dirs]}")
    reports = [triage_task(t) for t in task_dirs]
    for r in reports:
        _print_report(r)
    print(f"\n{'=' * 70}\nSUMMARY: {len(reports)} tasks, "
          f"{sum(r['total_episodes'] for r in reports)} episodes, {sum(r['total_frames'] for r in reports)} frames")
    for r in reports:
        print(f"    - {r['task_name']}: {r['total_episodes']} eps, {r['total_frames']} frames")


# ---------------- patch ----------------
def patch_one(ep_dir):
    cands = glob.glob(str(ep_dir / "meta" / "episodes" / "chunk-*" / "file-*.parquet"))
    if not cands:
        print(f"  SKIP (no episodes parquet): {ep_dir}"); return False
    video_keys = [p.name for p in (ep_dir / "videos").iterdir() if p.is_dir()] if (ep_dir / "videos").exists() else []
    fps = FPS_DEFAULT
    info_path = ep_dir / "meta" / "info.json"
    if info_path.exists():
        fps = load_json(info_path).get("fps", FPS_DEFAULT)
    changed = False
    for ep_parquet in cands:
        df = pd.read_parquet(ep_parquet)
        if "data/chunk_index" in df.columns:
            continue
        length = int(df["length"].iloc[0]) if "length" in df.columns else None
        df["data/chunk_index"] = 0; df["data/file_index"] = 0
        for key in video_keys:
            df[f"videos/{key}/chunk_index"] = 0; df[f"videos/{key}/file_index"] = 0
            df[f"videos/{key}/from_timestamp"] = 0.0
            df[f"videos/{key}/to_timestamp"] = (length / fps) if length else 0.0
        df["meta/episodes/chunk_index"] = 0; df["meta/episodes/file_index"] = 0
        if "episode_index" in df.columns and length is not None:
            df["dataset_from_index"] = 0; df["dataset_to_index"] = length
        df.to_parquet(ep_parquet); changed = True
    return changed


def cmd_patch(argv):
    data_root = Path(argv[0]).expanduser().resolve()
    ep_dirs = sorted(set(p.parent.parent for p in data_root.glob("*/*/meta/info.json")))
    print(f"Patching {len(ep_dirs)} episode-session directories...")
    n = 0
    for ep_dir in ep_dirs:
        print(f"- {ep_dir}")
        if patch_one(ep_dir):
            n += 1
    print(f"\nDone. Patched {n}/{len(ep_dirs)} (others already patched/skipped).")


# ---------------- validate ----------------
def cmd_validate(argv):
    root = argv[0] if argv else "/home/shaurya/g1_sonic_system1/data/g1_raw"
    parquet_files = sorted(glob.glob(f"{root}/*/*/data/chunk-*/file-*.parquet"))
    print(f"Found {len(parquet_files)} parquet files under {root}")
    cols = ["action.token_state", "action.ee_action", "action.robot_q_desired", "action.planner_cmd", "action.hand_cmd"]
    for pf in parquet_files:
        print(f"\n=== {pf} ===")
        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            print(f"  FAILED: {e}"); continue
        print(f"  rows={len(df)} cols={len(df.columns)}")
        for col in cols:
            if col not in df.columns:
                print(f"  {col}: NOT PRESENT"); continue
            arr = np.stack(df[col].to_numpy())
            nz = np.mean(np.abs(arr) > 1e-8)
            nonconst = int(np.sum(arr.std(axis=0) > 1e-6))
            print(f"  {col}: shape={arr.shape} nonzero_frac={nz:.4f} mean_abs={np.mean(np.abs(arr)):.6f} "
                  f"std={arr.std():.6f} nonconstant_dims={nonconst}/{arr.shape[1] if arr.ndim > 1 else 1} "
                  f"min={arr.min():.4f} max={arr.max():.4f}")


# ---------------- split ----------------
def _is_complete(ep_dir):
    videos_ok = (ep_dir / "videos").exists() and any((ep_dir / "videos").iterdir())
    return videos_ok and (ep_dir / "meta" / "stats.json").exists() and (ep_dir / "meta" / "tasks.parquet").exists()


def cmd_split(argv):
    import random
    data_root = Path(argv[0]).expanduser().resolve()
    out_path = Path(argv[1]).expanduser().resolve()
    rng = random.Random(SEED)
    result = {"seed": SEED,
              "policy": "hold out max(1, round(0.10*n_complete)) per task; incomplete dirs excluded",
              "generated_from_partial_dataset": True, "tasks": {}, "excluded_incomplete": []}
    for task_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        ep_dirs = sorted(p for p in task_dir.iterdir() if p.is_dir() and (p / "meta" / "info.json").exists())
        complete, incomplete = [], []
        for ep in ep_dirs:
            (complete if _is_complete(ep) else incomplete).append(ep)
        for ep in incomplete:
            result["excluded_incomplete"].append(f"{task_dir.name}/{ep.name}")
        names = sorted(p.name for p in complete)
        n = len(names)
        n_eval = max(1, round(0.10 * n)) if n > 0 else 0
        shuffled = names[:]; rng.shuffle(shuffled)
        result["tasks"][task_dir.name] = {"total_complete_episodes": n, "n_eval": n_eval,
                                          "eval_episode_ids": sorted(shuffled[:n_eval]),
                                          "train_episode_ids": sorted(shuffled[n_eval:])}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}\n" + json.dumps(result, indent=2))


# ---------------- encode ----------------
def cmd_encode(argv):
    SRC = "/lambdafs/shaurya/g1_sonic_system1/data/g1_raw_full"
    TASKS = ["bottle_cupnoodles_shelf", "cup_wipe_sponge_dryingrack", "floor_box_table"]
    VIDEO_KEYS = ["observation.images.head_cam", "observation.images.left_wrist_cam", "observation.images.right_wrist_cam"]
    FPS = 30
    ACTIVE_STD_THRESH = 1e-3
    hands_mode = "zero"
    OUT = Path("/lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic")
    exclude = set()
    for i, a in enumerate(argv):
        if a == "--hands-mode":
            hands_mode = argv[i + 1]
        if a == "--out":
            OUT = Path(argv[i + 1])
        if a == "--exclude":
            exclude = {int(x) for x in argv[i + 1].split(",") if x != ""}
    assert hands_mode in ("zero", "pico")
    print(f"hands_mode={hands_mode}  OUT={OUT}  exclude_original_idx={sorted(exclude)}")

    def projected_gravity(q):
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return np.stack([-(2 * (x * z - w * y)), -(2 * (y * z + w * x)), -(1 - 2 * (x * x + y * y))], axis=1).astype(np.float32)

    def build_state(q):
        j = q[:, 7:36]
        return np.concatenate([j[:, 0:6], j[:, 6:12], j[:, 12:15], j[:, 15:22], j[:, 22:29],
                               np.zeros((len(q), 7), np.float32), np.zeros((len(q), 7), np.float32),
                               projected_gravity(q[:, 3:7])], axis=1).astype(np.float32)

    ep_dirs = []
    for t in TASKS:
        for d in sorted(glob.glob(f"{SRC}/{t}/*/")):
            ep_dirs.append((t, Path(d)))

    hand_norm = {}
    if hands_mode == "pico":
        for task in TASKS:
            arrs = [np.stack(pd.read_parquet(p)["action.hand_cmd_pico"].to_numpy()).astype(np.float32)
                    for p in sorted(glob.glob(f"{SRC}/{task}/*/data/chunk-*/file-*.parquet"))]
            A = np.concatenate(arrs)
            active = (A.std(0) > ACTIVE_STD_THRESH)
            hand_norm[task] = {"active_dims": [int(i) for i in np.where(active)[0]],
                               "min": A.min(0).tolist(), "max": A.max(0).tolist(),
                               "normalization": "per-dim min-max to [0,1] on active dims; inactive->0"}
        print("hand active dims:", {t: hand_norm[t]["active_dims"] for t in TASKS})

    def norm_pico(task, pico):
        hn = hand_norm[task]
        mn = np.array(hn["min"], np.float32); mx = np.array(hn["max"], np.float32)
        out = np.zeros_like(pico)
        for d in hn["active_dims"]:
            rng = mx[d] - mn[d]
            out[:, d] = np.clip((pico[:, d] - mn[d]) / (rng if rng > 1e-9 else 1.0), 0.0, 1.0)
        return out.astype(np.float32)

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data" / "chunk-000").mkdir(parents=True)
    for vk in VIDEO_KEYS:
        (OUT / "videos" / "chunk-000" / vk).mkdir(parents=True)
    (OUT / "meta").mkdir(parents=True, exist_ok=True)

    task_to_idx = {t: i for i, t in enumerate(TASKS)}
    task_desc = {}
    episodes_jsonl, mapping, all_state, all_action = [], [], [], []
    gidx = 0
    e = -1
    for oi, (task, d) in enumerate(ep_dirs):
        if oi in exclude:
            print(f"  [skip original idx {oi}] {task}/{d.name} (held-out eval)"); continue
        e += 1
        df = pd.read_parquet(sorted(glob.glob(str(d / "data" / "chunk-*" / "file-*.parquet")))[0])
        T = len(df)
        q = np.stack(df["observation.state.robot_q_current"].to_numpy()).astype(np.float32)
        tok = np.stack(df["action.token_state"].to_numpy()).astype(np.float32)
        if hands_mode == "zero":
            hands = np.zeros((T, 14), np.float32)
        else:
            hands = norm_pico(task, np.stack(df["action.hand_cmd_pico"].to_numpy()).astype(np.float32))
        action = np.concatenate([tok, hands], axis=1).astype(np.float32)
        state = build_state(q)
        all_state.append(state); all_action.append(action)
        desc = str(pd.read_parquet(d / "meta" / "tasks.parquet")["task"].iloc[0])
        task_desc[task_to_idx[task]] = desc
        pd.DataFrame({"observation.state": list(state), "action": list(action),
                      "timestamp": (np.arange(T) / FPS).astype(np.float32),
                      "frame_index": np.arange(T, dtype=np.int64), "episode_index": np.full(T, e, np.int64),
                      "index": np.arange(gidx, gidx + T, dtype=np.int64),
                      "task_index": np.full(T, task_to_idx[task], np.int64)}
                     ).to_parquet(OUT / "data" / "chunk-000" / f"episode_{e:06d}.parquet")
        for vk in VIDEO_KEYS:
            shutil.copy(sorted(glob.glob(str(d / "videos" / vk / "chunk-*" / "file-*.mp4")))[0],
                        OUT / "videos" / "chunk-000" / vk / f"episode_{e:06d}.mp4")
        episodes_jsonl.append({"episode_index": e, "tasks": [desc], "length": T})
        mapping.append({"episode_index": e, "original_episode_index": oi, "task": task, "session": d.name, "length": T})
        gidx += T
        print(f"  [{e:02d}] {task}/{d.name} T={T}")

    with open(OUT / "meta" / "episodes.jsonl", "w") as f:
        for r in episodes_jsonl:
            f.write(json.dumps(r) + "\n")
    with open(OUT / "meta" / "tasks.jsonl", "w") as f:
        for i in sorted(task_desc):
            f.write(json.dumps({"task_index": i, "task": task_desc[i]}) + "\n")

    n_ep = len(mapping)
    per_task_counts = {t: sum(1 for m in mapping if m["task"] == t) for t in TASKS}
    print("per-task train-episode counts:", per_task_counts)
    src_info = json.load(open(sorted(glob.glob(f"{SRC}/{TASKS[0]}/*/meta/info.json"))[0]))
    features = {
        "observation.state": {"dtype": "float32", "shape": [46], "names": {"axes": [
            *[f"left_leg_{i}" for i in range(6)], *[f"right_leg_{i}" for i in range(6)], "waist_yaw", "waist_roll", "waist_pitch",
            *[f"left_arm_{i}" for i in range(7)], *[f"right_arm_{i}" for i in range(7)],
            *[f"left_hand_{i}" for i in range(7)], *[f"right_hand_{i}" for i in range(7)], "grav_x", "grav_y", "grav_z"]}},
        "action": {"dtype": "float32", "shape": [78], "names": {"axes": [
            *[f"motion_token_{i}" for i in range(64)], *[f"left_hand_joint_{i}" for i in range(7)], *[f"right_hand_joint_{i}" for i in range(7)]]}},
        "timestamp": {"dtype": "float32", "shape": [1]}, "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]}, "index": {"dtype": "int64", "shape": [1]}, "task_index": {"dtype": "int64", "shape": [1]}}
    for vk in VIDEO_KEYS:
        features[vk] = src_info["features"][vk]
    info = {"codebase_version": "v2.1", "robot_type": "unitree_g1", "total_episodes": n_ep,
            "total_frames": int(gidx), "total_tasks": len(TASKS), "total_videos": n_ep * len(VIDEO_KEYS),
            "total_chunks": 1, "chunks_size": 1000, "fps": FPS, "splits": {"train": f"0:{n_ep}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", "features": features}
    json.dump(info, open(OUT / "meta" / "info.json", "w"), indent=2)

    S = np.concatenate(all_state); A = np.concatenate(all_action)

    def st(a):
        return {"mean": a.mean(0).tolist(), "std": (a.std(0) + 1e-8).tolist(), "min": a.min(0).tolist(),
                "max": a.max(0).tolist(), "q01": np.quantile(a, 0.01, 0).tolist(), "q99": np.quantile(a, 0.99, 0).tolist()}
    json.dump({"observation.state": st(S), "action": st(A)}, open(OUT / "meta" / "stats.json", "w"), indent=2)

    modality = {
        "state": {"left_leg": {"start": 0, "end": 6}, "right_leg": {"start": 6, "end": 12}, "waist": {"start": 12, "end": 15},
                  "left_arm": {"start": 15, "end": 22}, "right_arm": {"start": 22, "end": 29}, "left_hand": {"start": 29, "end": 36},
                  "right_hand": {"start": 36, "end": 43}, "projected_gravity": {"start": 43, "end": 46}},
        "action": {"motion_token": {"start": 0, "end": 64}, "left_hand_joints": {"start": 64, "end": 71}, "right_hand_joints": {"start": 71, "end": 78}},
        "video": {"ego_view": {"original_key": "observation.images.head_cam"},
                  "left_wrist_view": {"original_key": "observation.images.left_wrist_cam"},
                  "right_wrist_view": {"original_key": "observation.images.right_wrist_cam"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}}}
    json.dump(modality, open(OUT / "meta" / "modality.json", "w"), indent=2)

    thin = {t: c for t, c in per_task_counts.items() if c <= 1}
    map_out = {"embodiment_tag": "unitree_g1_sonic", "format": "lerobot v2.1 (GR00T N1.7)", "hands_mode": hands_mode,
               "ordering": "task-then-session alphabetical", "n_episodes": n_ep,
               "excluded_original_indices": sorted(exclude), "per_task_episode_counts": per_task_counts,
               "note": ("episode_index = index within THIS dataset dir; original_episode_index = index in the full 22-ep "
                        "dataset. Excluded indices are held-out eval episodes; the eval harness reads those from the FULL datasets."),
               "episodes": mapping}
    if thin:
        map_out["DATA_THIN_WARNING"] = (f"Tasks with <=1 training episode: {thin}. bottle_cupnoodles_shelf had only 2 total; "
                                        "excluding held-out eval leaves 1 training episode -> expect weak generalization (documented limitation).")
    json.dump(map_out, open(OUT / "meta" / "episode_mapping.json", "w"), indent=2)

    if hands_mode == "pico":
        hm = {"note": ("action.left_hand_joints[64:71]<-pico[0:7], right_hand_joints[71:78]<-pico[7:14]. Per-task per-dim min-max "
                       "to [0,1] on active dims; inactive->0. Invert: x = norm*(max-min)+min. active_action_dims = dataset action "
                       "indices (64+pico_dim) carrying real signal (use for loss masking)."), "tasks": {}}
        for t in TASKS:
            ad = hand_norm[t]["active_dims"]
            hm["tasks"][t] = {"task_index": task_to_idx[t], "pico_active_dims": ad, "active_action_dims": [64 + d for d in ad],
                              "min": [hand_norm[t]["min"][d] for d in ad], "max": [hand_norm[t]["max"][d] for d in ad]}
        json.dump(hm, open(OUT / "meta" / "hand_normalization.json", "w"), indent=2)

    print(f"\nDONE ({hands_mode}). total_frames={int(gidx)}. Dataset at {OUT}")


CMDS = {"triage": cmd_triage, "patch": cmd_patch, "validate": cmd_validate, "split": cmd_split, "encode": cmd_encode}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print("usage: prep.py {triage,patch,validate,split,encode} [args...]"); sys.exit(2)
    CMDS[sys.argv[1]](sys.argv[2:])
