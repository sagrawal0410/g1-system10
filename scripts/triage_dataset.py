#!/usr/bin/env python3
"""
Phase 0 dataset triage for the G1 SONIC / VLA sprint.

Walks a directory of downloaded "G1-Data" episode folders (each folder is a
self-contained LeRobot-format dataset directory, as produced by the
GR00T-WholeBodyControl `run_data_exporter.py` / data-collection pipeline) and
reports, per *task* (inferred from the top-level subdirectory name):

  - episode count (sum of `total_episodes` across all session folders for that task)
  - observation keys (state + image + depth), with shapes/dtypes
  - camera names, resolution, and fps
  - state dims (broken out by semantic group, via meta/modality.json)
  - action dims + semantic meaning (ditto)
  - frequency (fps)
  - format (codebase_version, e.g. "v3.0")

Does NOT convert or merge anything -- read-only triage. Run from the `sonic`
or `lerobot` conda env (both have pandas/pyarrow available).

Usage:
    python triage_dataset.py --data-root ~/g1_sonic_system1/data/g1_raw
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)

def find_episode_dirs(task_dir: Path):
    """An 'episode dir' is any subdirectory containing meta/info.json."""
    out = []
    for p in sorted(task_dir.iterdir()):
        if p.is_dir() and (p / "meta" / "info.json").exists():
            out.append(p)
    return out

def summarize_features(features: dict):
    obs_state, obs_image, obs_depth, actions, other = {}, {}, {}, {}, {}
    for key, spec in features.items():
        dtype = spec.get("dtype")
        shape = spec.get("shape")
        names = spec.get("names")
        if isinstance(names, dict):
            names = names.get("axes")
        entry = {"dtype": dtype, "shape": shape, "names": names}
        if key.startswith("observation.images."):
            obs_image[key] = entry
        elif key.startswith("observation.depths."):
            obs_depth[key] = entry
        elif key.startswith("observation.state.") or key.startswith("observation."):
            obs_state[key] = entry
        elif key.startswith("action."):
            actions[key] = entry
        else:
            other[key] = entry
    return obs_state, obs_image, obs_depth, actions, other

def modality_semantic_groups(modality: dict, section: str):
    """Return {group_name: (start, end)} for 'state' or 'action' sections."""
    return {k: (v["start"], v["end"]) for k, v in modality.get(section, {}).items()}

def triage_task(task_dir: Path):
    episode_dirs = find_episode_dirs(task_dir)
    report = {
        "task_name": task_dir.name,
        "num_session_dirs": len(episode_dirs),
        "total_episodes": 0,
        "total_frames": 0,
        "codebase_versions": set(),
        "robot_types": set(),
        "fps_values": set(),
        "cameras": {},
        "depths": {},
        "state_features": {},
        "action_features": {},
        "state_semantic_groups": {},
        "action_semantic_groups": {},
        "task_descriptions": set(),
        "errors": [],
    }

    for ep_dir in episode_dirs:
        info_path = ep_dir / "meta" / "info.json"
        modality_path = ep_dir / "meta" / "modality.json"
        try:
            info = load_json(info_path)
        except Exception as e:
            report["errors"].append(f"{ep_dir.name}: failed to read info.json ({e})")
            continue

        report["codebase_versions"].add(info.get("codebase_version", "unknown"))
        report["robot_types"].add(info.get("robot_type", "unknown"))
        report["fps_values"].add(info.get("fps"))
        report["total_episodes"] += info.get("total_episodes", 0)
        report["total_frames"] += info.get("total_frames", 0)

        obs_state, obs_image, obs_depth, actions, _other = summarize_features(
            info.get("features", {})
        )
        report["state_features"].update(obs_state)
        report["action_features"].update(actions)

        for cam_key, spec in obs_image.items():
            shape = spec["shape"]
            report["cameras"][cam_key] = {
                "resolution": f"{shape[1]}x{shape[0]}" if shape else "unknown",
                "channels": shape[2] if shape and len(shape) > 2 else None,
                "fps": info.get("fps"),
            }
        for depth_key in obs_depth:
            report["depths"][depth_key] = {"fps": info.get("fps")}

        if modality_path.exists():
            try:
                modality = load_json(modality_path)
                report["state_semantic_groups"].update(
                    modality_semantic_groups(modality, "state")
                )
                report["action_semantic_groups"].update(
                    modality_semantic_groups(modality, "action")
                )
            except Exception as e:
                report["errors"].append(f"{ep_dir.name}: failed to read modality.json ({e})")

        tasks_parquet = ep_dir / "meta" / "tasks.parquet"
        if tasks_parquet.exists() and pd is not None:
            try:
                tdf = pd.read_parquet(tasks_parquet)
                for col in ("task", "task_description", "language_instruction"):
                    if col in tdf.columns:
                        report["task_descriptions"].update(tdf[col].astype(str).tolist())
                        break
            except Exception as e:
                report["errors"].append(f"{ep_dir.name}: failed to read tasks.parquet ({e})")

    return report

def print_report(report: dict):
    print(f"\n{'=' * 70}")
    print(f"TASK: {report['task_name']}")
    print(f"{'=' * 70}")
    print(f"  Session/episode dirs found : {report['num_session_dirs']}")
    print(f"  Total episodes             : {report['total_episodes']}")
    print(f"  Total frames               : {report['total_frames']}")
    print(f"  codebase_version(s)        : {sorted(report['codebase_versions'])}")
    print(f"  robot_type(s)              : {sorted(report['robot_types'])}")
    print(f"  fps                        : {sorted(v for v in report['fps_values'] if v is not None)}")
    if report["task_descriptions"]:
        print(f"  task description(s)        : {sorted(report['task_descriptions'])}")

    print(f"\n  Cameras ({len(report['cameras'])}):")
    for cam, spec in report["cameras"].items():
        print(f"    - {cam}: {spec['resolution']} @ {spec['fps']} fps, channels={spec['channels']}")
    if report["depths"]:
        print(f"  Depth streams ({len(report['depths'])}):")
        for d, spec in report["depths"].items():
            print(f"    - {d} @ {spec['fps']} fps")

    print(f"\n  Observation/state features ({len(report['state_features'])}):")
    for key, spec in sorted(report["state_features"].items()):
        names_str = f" names={spec['names']}" if spec["names"] else ""
        print(f"    - {key}: {spec['dtype']} shape={spec['shape']}{names_str}")

    print(f"\n  Action features ({len(report['action_features'])}):")
    for key, spec in sorted(report["action_features"].items()):
        names_str = f" names={spec['names']}" if spec["names"] else ""
        print(f"    - {key}: {spec['dtype']} shape={spec['shape']}{names_str}")

    if report["state_semantic_groups"]:
        print(f"\n  State semantic groups (modality.json slice ranges):")
        for grp, (s, e) in sorted(report["state_semantic_groups"].items(), key=lambda kv: kv[1]):
            print(f"    - {grp}: [{s}:{e}] (dim={e - s})")

    if report["action_semantic_groups"]:
        print(f"\n  Action semantic groups (modality.json slice ranges):")
        for grp, (s, e) in sorted(report["action_semantic_groups"].items(), key=lambda kv: kv[1]):
            print(f"    - {grp}: [{s}:{e}] (dim={e - s})")

    action_keys = set(report["action_features"].keys())
    has_joint = "action.robot_q_desired" in action_keys
    has_ee = "action.ee_action" in action_keys
    has_planner = "action.planner_cmd" in action_keys
    has_token = "action.token_state" in action_keys
    print(f"\n  --- Action-space determination ---")
    print(f"    joint-space present   (action.robot_q_desired) : {has_joint}")
    print(f"    teleop/EE-space present (action.ee_action)      : {has_ee}")
    print(f"    planner/nav command  (action.planner_cmd)       : {has_planner}")
    print(f"    SONIC latent tokens  (action.token_state)       : {has_token}")
    if has_joint and has_ee:
        print("    => BOTH representations already present natively.")
        print("       No FK computation needed; runbook's 'keep both action.teleop")
        print("       and action.joints' requirement is already satisfied by")
        print("       action.ee_action / action.planner_cmd (teleop) and")
        print("       action.robot_q_desired (joints).")
    elif has_joint and not has_ee:
        print("    => Joint-space only found -- FK to teleop poses would be required.")

    if report["errors"]:
        print(f"\n  Errors/warnings:")
        for e in report["errors"]:
            print(f"    ! {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path to data/g1_raw")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: data root {root} does not exist", file=sys.stderr)
        sys.exit(1)

    task_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not task_dirs:
        print(f"ERROR: no task subdirectories found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"G1 Dataset Triage Report")
    print(f"Data root: {root}")
    print(f"Task subdirectories found: {[t.name for t in task_dirs]}")

    all_reports = []
    for task_dir in task_dirs:
        report = triage_task(task_dir)
        all_reports.append(report)
        print_report(report)

    print(f"\n{'=' * 70}")
    print("SUMMARY ACROSS ALL TASKS")
    print(f"{'=' * 70}")
    total_eps = sum(r["total_episodes"] for r in all_reports)
    total_frames = sum(r["total_frames"] for r in all_reports)
    print(f"  Tasks               : {len(all_reports)}")
    print(f"  Total episodes      : {total_eps}")
    print(f"  Total frames        : {total_frames}")
    versions = set()
    for r in all_reports:
        versions |= r["codebase_versions"]
    print(f"  codebase_version(s) : {sorted(versions)}")
    for r in all_reports:
        print(f"    - {r['task_name']}: {r['total_episodes']} episodes, {r['total_frames']} frames")

if __name__ == "__main__":
    main()
