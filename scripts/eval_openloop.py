#!/usr/bin/env python3
"""
Phase A2 — Open-loop evaluation harness for GR00T N1.7 / UNITREE_G1_SONIC.

Compares checkpoints (base "zero-shot proxy", fine-tuned T1, optionally T2 /
Phase-D stack) on held-out episodes by running the *dataset* observations
(camera + language + proprio) through each policy and comparing the predicted
SONIC-latent-action chunks against the ground-truth targets recorded in the
dataset. This mirrors the pattern in Isaac-GR00T's own
`gr00t/eval/open_loop_eval.py` (single aggregate MSE/MAE per trajectory) but:

  1. Breaks every metric out **per action block** (latent / left_hand /
     right_hand) using the *real* block indices from `results/action_layout.json`
     (written by the Phase 0.5 agent) instead of guessing — never just total MSE.
  2. Adds: final-position error, command smoothness (mean-squared jerk),
     error-vs-horizon-index curves, and grasp-event F1 per hand.
  3. Emits one long-format `results/openloop_metrics.csv` plus GT-vs-pred
     overlay plots for a couple of episodes per task.

IMPORTANT SCOPE NOTE ON "DECODED WRIST XYZ" (read before extending this script)
--------------------------------------------------------------------------
The runbook asks for overlay plots of "decoded wrist xyz + hand open/close
traces". Hand open/close traces are directly available (the 7-dim
left/right hand *joint* blocks in the action are already actual joint
targets, not latents) and are plotted here with no decoding needed.

The 64-dim `motion_token` block is NOT a wrist pose — it is SONIC's latent
whole-body motion code. Turning it into an actual wrist xyz trajectory
requires running SONIC's *recurrent* decoder (`model_decoder.onnx`), which
was inspected here and needs a 436-dim observation: the 64-dim token PLUS
10-frame histories of base angular velocity, body joint positions, body
joint velocities, last actions, and gravity direction (372 dims total) — i.e.
a live 50 Hz proprioceptive history from a closed control loop, not a
single dataset frame. Reconstructing that (history buffers, encoder-mode
observation assembly, etc.) currently only exists in the C++ deploy stack
(`gear_sonic_deploy/src/...`); re-implementing it in Python is a MuJoCo/
closed-loop-sim-integration task — that's Phase C, explicitly out of scope
for Phase A.

So this harness plots the **latent block trajectory itself** (GT vs
predicted, summarized by per-step L2 norm + top principal-component traces)
as the "body/wrist-motion" overlay, and clearly labels it as latent-space,
not decoded xyz. If a later phase (C) wants true decoded xyz overlays, this
script's `--dump-raw-npz` output already has everything (raw pred/GT arrays)
needed to re-run through a proper closed-loop decode.

Usage
-----
    python eval_openloop.py \\
        --action-layout results/action_layout.json \\
        --eval-split results/eval_split.json \\
        --checkpoint base_zeroshot_proxy=/path/to/checkpoint-fewsteps \\
        --checkpoint T1=/path/to/output/checkpoint-20000 \\
        --embodiment-tag UNITREE_G1_SONIC \\
        --execution-horizon 40 \\
        --output-csv results/openloop_metrics.csv \\
        --plot-dir results/openloop_plots

Each `--checkpoint LABEL=PATH` may be repeated. Requires the `groot` conda
env (Isaac-GR00T installed) to actually run policies; the metric/plotting
code has no other Isaac-GR00T-specific runtime dependency.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_openloop")


# --------------------------------------------------------------------------
# Action layout (blocks) loading — reads the REAL layout from action_layout.json
# rather than hard-coding indices. Falls back to the values we independently
# verified from gr00t/configs/data/embodiment_configs.py's `unitree_g1_sonic`
# entry (motion_token=64, left_hand_joints=7, right_hand_joints=7, chunk=40)
# ONLY if the file is missing/unreadable, and loudly warns when it does.
# --------------------------------------------------------------------------

_FALLBACK_BLOCKS = {
    "latent": (0, 64),
    "left_hand": (64, 71),
    "right_hand": (71, 78),
}
_FALLBACK_TOTAL_DIM = 78
_FALLBACK_CHUNK_LENGTH = 40


@dataclass
class ActionLayout:
    total_dim: int
    chunk_length: int
    blocks: dict[str, tuple[int, int]]  # name -> (start, end) exclusive
    state_keys: Optional[list[str]] = None
    latent_continuous: Optional[bool] = None
    source: str = "action_layout.json"

    def validate(self) -> None:
        covered = sum(e - s for s, e in self.blocks.values())
        if covered != self.total_dim:
            raise ValueError(
                f"[action_layout MISMATCH] blocks {self.blocks} cover {covered} dims "
                f"but total_dim={self.total_dim}. Refusing to guess — fix action_layout.json "
                f"or pass --force-fallback-layout if this is expected."
            )
        # Cross-check against the embodiment config we verified by hand
        # (gr00t/configs/data/embodiment_configs.py: unitree_g1_sonic).
        if self.total_dim != _FALLBACK_TOTAL_DIM:
            log.warning(
                "action_layout.json total_dim=%d differs from the unitree_g1_sonic "
                "modality config's expected 78 (64 latent + 7 + 7 hand). "
                "This MUST be reconciled before launching training — flagging, not blocking eval.",
                self.total_dim,
            )


def _first_present(d: dict, keys: list[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _parse_block_list(items: list) -> Optional[dict[str, tuple[int, int]]]:
    """Parse a LIST of block dicts, each like
    {"name": "motion_token", "indices": [0, 64], "dim": 64, ...}.
    This is the schema the Phase 0.5 agent actually emits
    (action.block_layout). Also accepts "start"/"end" or "slice" keys."""
    out = {}
    for item in items:
        if not isinstance(item, dict):
            return None
        name = item.get("name") or item.get("modality_key")
        if name is None:
            return None
        idx = item.get("indices") or item.get("slice")
        if isinstance(idx, (list, tuple)) and len(idx) == 2:
            s, e = idx
        elif "start" in item and "end" in item:
            s, e = item["start"], item["end"]
        else:
            return None
        out[name] = (int(s), int(e))
    return out or None


def _parse_blocks(d: dict) -> Optional[dict[str, tuple[int, int]]]:
    """Try several plausible schemas for the block-index section of
    action_layout.json. Returns None if nothing recognizable is found."""
    # REAL schema (Phase 0.5 agent): action.block_layout is a list of dicts.
    action_sec = d.get("action")
    if isinstance(action_sec, dict):
        block_list = action_sec.get("block_layout") or action_sec.get("blocks")
        if isinstance(block_list, list) and block_list:
            parsed = _parse_block_list(block_list)
            if parsed is not None:
                return parsed
        # Or a dict-of-dicts nested under action.
        if isinstance(block_list, dict) and block_list:
            parsed = _parse_blocks_dict(block_list)
            if parsed is not None:
                return parsed

    # Top-level list schema.
    for key in ("block_layout", "blocks", "block_indices", "action_blocks"):
        v = d.get(key)
        if isinstance(v, list) and v:
            parsed = _parse_block_list(v)
            if parsed is not None:
                return parsed
        if isinstance(v, dict) and v:
            parsed = _parse_blocks_dict(v)
            if parsed is not None:
                return parsed

    # Alternative flat schema: separate top-level keys per block.
    alt_names = {
        "latent": ["latent_indices", "latent_block", "motion_token_indices"],
        "left_hand": ["left_hand_indices", "left_hand_block"],
        "right_hand": ["right_hand_indices", "right_hand_block"],
    }
    out = {}
    for canonical, aliases in alt_names.items():
        spec = _first_present(d, aliases)
        if spec is None:
            return None
        if isinstance(spec, dict):
            out[canonical] = (int(spec["start"]), int(spec["end"]))
        elif isinstance(spec, (list, tuple)) and len(spec) == 2:
            out[canonical] = (int(spec[0]), int(spec[1]))
        else:
            return None
    return out or None


def _parse_blocks_dict(raw: dict) -> Optional[dict[str, tuple[int, int]]]:
    out = {}
    for name, spec in raw.items():
        if isinstance(spec, dict):
            s, e = spec.get("start"), spec.get("end")
            if (s is None or e is None) and isinstance(spec.get("indices"), (list, tuple)):
                s, e = spec["indices"]
        elif isinstance(spec, (list, tuple)) and len(spec) == 2:
            s, e = spec
        else:
            return None
        if s is None or e is None:
            return None
        out[name] = (int(s), int(e))
    return out or None


def load_action_layout(path: str | Path) -> ActionLayout:
    path = Path(path)
    if not path.exists():
        log.warning(
            "action_layout.json not found at %s — using the hand-verified fallback "
            "layout derived from gr00t/configs/data/embodiment_configs.py "
            "(latent=0:64, left_hand=64:71, right_hand=71:78, chunk=40). "
            "DO NOT launch training off this fallback; re-run once the real file exists.",
            path,
        )
        layout = ActionLayout(
            total_dim=_FALLBACK_TOTAL_DIM,
            chunk_length=_FALLBACK_CHUNK_LENGTH,
            blocks=dict(_FALLBACK_BLOCKS),
            source="fallback (embodiment_configs.py-derived)",
        )
        layout.validate()
        return layout

    with open(path) as f:
        d = json.load(f)

    blocks = _parse_blocks(d)
    if blocks is None:
        raise ValueError(
            f"Could not find a recognizable block-index schema in {path}. "
            f"Top-level keys present: {sorted(d.keys())}. "
            f"Extend `_parse_blocks()` to handle this schema rather than guessing silently."
        )

    # total_dim / chunk_length may live at top level OR under the "action" sub-dict
    # (the real Phase-0.5 schema puts them under action.{total_dim,chunk_length}).
    action_sec = d.get("action") if isinstance(d.get("action"), dict) else {}
    total_dim = _first_present(d, ["total_action_dim", "total_dim", "action_dim"])
    if total_dim is None:
        total_dim = _first_present(action_sec, ["total_action_dim", "total_dim", "action_dim"])
    if total_dim is None:
        total_dim = sum(e - s for s, e in blocks.values())
        log.warning("No total_dim key in %s; inferring %d from blocks.", path, total_dim)

    chunk_length = _first_present(d, ["chunk_length", "action_horizon", "chunk_len", "horizon"])
    if chunk_length is None:
        chunk_length = _first_present(action_sec, ["chunk_length", "action_horizon", "chunk_len", "horizon"])
    if chunk_length is None:
        log.warning("No chunk_length key in %s; falling back to %d.", path, _FALLBACK_CHUNK_LENGTH)
        chunk_length = _FALLBACK_CHUNK_LENGTH

    # expected obs / state keys: top level or under observation.*
    obs_sec = d.get("observation") if isinstance(d.get("observation"), dict) else {}
    state_keys = _first_present(d, ["expected_obs_keys", "state_keys", "obs_keys"])
    if state_keys is None:
        state_keys = _first_present(obs_sec, ["state_modality_keys", "state_keys", "expected_obs_keys"])

    layout = ActionLayout(
        total_dim=int(total_dim),
        chunk_length=int(chunk_length),
        blocks=blocks,
        state_keys=state_keys,
        latent_continuous=_first_present(d, ["latent_continuous"]),
        source=str(path),
    )
    layout.validate()
    return layout


# --------------------------------------------------------------------------
# eval_split.json loading — flexible over a few plausible shapes.
# Normalizes to: dict[task_name] -> {"dataset_path": str|None, "episode_ids": [int,...]}
# --------------------------------------------------------------------------


def load_eval_split(
    path: str | Path, default_dataset_path: Optional[str] = None
) -> dict[str, dict[str, Any]]:
    path = Path(path)
    with open(path) as f:
        d = json.load(f)

    out: dict[str, dict[str, Any]] = {}

    # REAL schema (Phase-0.5 build_eval_split for the encoded merged dataset):
    #   {"eval": [{"episode_index": 0, "task": "bottle_...", "session": "..."}, ...],
    #    "train_episode_indices": [...], "n_total": 22, ...}
    # episode_index is an INTEGER row index into the FULL encoded dataset.
    if isinstance(d, dict) and isinstance(d.get("eval"), list) and d["eval"]:
        for rec in d["eval"]:
            if not isinstance(rec, dict):
                raise ValueError(f"Unexpected eval record (not a dict): {rec}")
            task_name = rec.get("task") or rec.get("task_name") or "all"
            ep_id = rec.get("episode_index")
            if ep_id is None:
                ep_id = rec.get("episode_id")
            if ep_id is None:
                raise ValueError(f"eval record missing episode_index/episode_id: {rec}")
            ds_path = rec.get("dataset_path", default_dataset_path)
            entry = out.setdefault(task_name, {"dataset_path": ds_path, "episode_ids": []})
            entry["episode_ids"].append(ep_id)
        return out

    if isinstance(d, dict) and "tasks" in d and isinstance(d["tasks"], dict):
        for task_name, spec in d["tasks"].items():
            ds_path = spec.get("dataset_path", default_dataset_path)
            # NOTE: as of the Phase-0 `build_eval_split.py` script, episode
            # identifiers are the raw *episode-session directory names*
            # (e.g. "2026-07-06-09-24-03_bottle_cupnoodles_shelf") from the
            # WBC data exporter's per-session LeRobotDataset layout, NOT
            # plain integer row indices into one merged dataset. Keep
            # whatever type/value is given here — resolution to an actual
            # loader index happens in `resolve_episode_index()` at rollout
            # time, once we know the shape of the final encoded/merged
            # dataset the Phase 0.5 agent produces.
            ep_ids = (
                spec.get("eval_episode_ids")
                or spec.get("episode_ids")
                or spec.get("episodes")
                or spec.get("held_out")
            )
            out[task_name] = {"dataset_path": ds_path, "episode_ids": list(ep_ids)}
        return out

    if isinstance(d, dict) and all(
        isinstance(v, (list, tuple)) for v in d.values()
    ):
        # {"task_name": [episode_ids, ...], ...}
        for task_name, ep_ids in d.items():
            out[task_name] = {"dataset_path": default_dataset_path, "episode_ids": list(ep_ids)}
        return out

    if isinstance(d, list):
        # [{"task": ..., "episode_id": ..., "dataset_path": ...}, ...]
        for rec in d:
            task_name = rec.get("task") or rec.get("task_name")
            ds_path = rec.get("dataset_path", default_dataset_path)
            ep_id = rec.get("episode_id")
            if task_name is None or ep_id is None:
                raise ValueError(f"Unrecognized eval_split.json record: {rec}")
            entry = out.setdefault(task_name, {"dataset_path": ds_path, "episode_ids": []})
            entry["episode_ids"].append(ep_id)
        return out

    raise ValueError(
        f"Unrecognized eval_split.json schema at {path} (top-level type "
        f"{type(d)}). Extend `load_eval_split()` rather than guessing."
    )


# --------------------------------------------------------------------------
# Metric primitives
# --------------------------------------------------------------------------


def block_slice(traj: np.ndarray, block: tuple[int, int]) -> np.ndarray:
    s, e = block
    return traj[..., s:e]


# The Phase 0.5 agent writes action_layout.json's block names; they are NOT
# guaranteed to be exactly "latent"/"left_hand"/"right_hand". The embodiment
# config's action modality_keys are actually "motion_token",
# "left_hand_joints", "right_hand_joints", and — per the coordinator's
# heads-up — the real hand blocks may be 12-dim (dex hand) or 2-dim (gripper)
# rather than 7, possibly with different names. So classify blocks by
# name-substring instead of assuming exact names, so grasp detection and
# plotting attach to the right blocks regardless of the layout's naming.


def is_latent_block(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("latent", "token", "motion")) and "hand" not in n


def is_hand_block(name: str) -> bool:
    n = name.lower()
    return "hand" in n or "gripper" in n or "finger" in n


def hand_side(name: str) -> str:
    n = name.lower()
    if "left" in n or n.startswith("l_") or "_l_" in n:
        return "left"
    if "right" in n or n.startswith("r_") or "_r_" in n:
        return "right"
    return "unknown"


def mean_squared_jerk(traj: np.ndarray) -> float:
    """Discrete 3rd-order difference (proportional to jerk for uniform dt),
    squared and averaged over time and dims. traj: (T, D)."""
    if traj.shape[0] < 4:
        return float("nan")
    jerk = np.diff(traj, n=3, axis=0)
    return float(np.mean(jerk**2))


def final_position_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """L2 norm of the (pred - gt) vector at the last executed timestep."""
    return float(np.linalg.norm(pred[-1] - gt[-1]))


def error_vs_horizon(
    pred: np.ndarray, gt: np.ndarray, execution_horizon: int
) -> np.ndarray:
    """Mean squared error bucketed by position-within-chunk (0..execution_horizon-1),
    averaged over all chunks in the episode. Returns array of length execution_horizon
    (NaN where no data landed in that bucket, e.g. a short final chunk)."""
    T = pred.shape[0]
    buckets = [[] for _ in range(execution_horizon)]
    for t in range(T):
        pos = t % execution_horizon
        buckets[pos].append(np.mean((pred[t] - gt[t]) ** 2))
    return np.array([np.mean(b) if b else np.nan for b in buckets])


def hand_closure_signal(hand_traj: np.ndarray) -> np.ndarray:
    """Scalar closure proxy per timestep: mean across the 7 hand-joint dims.
    Sign/scale convention is whatever the dataset uses; detect_close_events
    below is threshold-relative (min/max of the *episode's own* GT signal),
    so it's robust to the exact convention."""
    return hand_traj.mean(axis=-1)


@dataclass
class GraspEvent:
    step: int
    kind: str  # "close" or "open"


def detect_close_events(
    signal: np.ndarray, low_q: float = 0.25, high_q: float = 0.75, min_gap: int = 3
) -> list[GraspEvent]:
    """Hysteresis threshold-crossing event detector on a scalar closure signal.

    Re-implementation of the runbook's referenced `detect_close_events` (the
    original draft file wasn't available to this agent) — a simple two-
    threshold (Schmitt-trigger) state machine: crossing above `high_q` after
    having been below `low_q` fires a "close" event; crossing below `low_q`
    after having been above `high_q` fires an "open" event. Thresholds are
    the given percentiles of the signal's own range (per-episode adaptive,
    since the dataset's hand-joint sign/scale convention isn't assumed).
    """
    if len(signal) == 0:
        return []
    lo = np.quantile(signal, low_q)
    hi = np.quantile(signal, high_q)
    if hi <= lo:
        return []
    events: list[GraspEvent] = []
    state = "unknown"
    last_event_step = -min_gap - 1
    for t, v in enumerate(signal):
        if state != "closed" and v >= hi:
            if t - last_event_step >= min_gap:
                events.append(GraspEvent(step=t, kind="close"))
                last_event_step = t
            state = "closed"
        elif state != "open" and v <= lo:
            if t - last_event_step >= min_gap:
                events.append(GraspEvent(step=t, kind="open"))
                last_event_step = t
            state = "open"
    return events


def match_events(
    gt_events: list[GraspEvent], pred_events: list[GraspEvent], tol_steps: int
) -> dict[str, float]:
    """Greedy nearest-neighbor matching (same `kind`, within `tol_steps`)."""
    unmatched_pred = list(pred_events)
    tp = 0
    timing_errors = []
    for g in gt_events:
        candidates = [
            p for p in unmatched_pred if p.kind == g.kind and abs(p.step - g.step) <= tol_steps
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda p: abs(p.step - g.step))
        unmatched_pred.remove(best)
        tp += 1
        timing_errors.append(best.step - g.step)

    fn = len(gt_events) - tp
    fp = len(unmatched_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall)
        else float("nan")
    )
    return {
        "grasp_precision": precision,
        "grasp_recall": recall,
        "grasp_f1": f1,
        "grasp_n_gt_events": len(gt_events),
        "grasp_n_pred_events": len(pred_events),
        "grasp_onset_timing_error_mean_steps": (
            float(np.mean(timing_errors)) if timing_errors else float("nan")
        ),
    }


def resolve_episode_index(loader, episode_id: Any, dataset_path: str) -> int:
    """Map an `eval_split.json` episode identifier to an integer row index
    into `loader` (a `LeRobotEpisodeLoader`).

    As of the Phase-0 `build_eval_split.py`, identifiers are the raw
    episode-session *directory names* from the WBC data exporter (one
    directory == one mostly-single-episode LeRobotDataset), e.g.
    "2026-07-06-09-24-03_bottle_cupnoodles_shelf" — not plain integer
    indices into one merged dataset. The Phase 0.5 agent's encoded/merged
    training dataset may or may not preserve that name; this function tries,
    in order:
      1. episode_id is already an int, or a string that parses cleanly as one
         -> use directly.
      2. `<dataset_path>/meta/episodes.jsonl` has a record whose fields
         contain episode_id as a substring/exact match anywhere in its
         string-valued fields -> use that record's `episode_index`.
    Raises a loud, actionable error (listing sample records) rather than
    silently falling back to episode 0 if nothing matches — per the "flag
    mismatches, don't force a launch" guidance for this task.
    """
    if isinstance(episode_id, int):
        return episode_id
    if isinstance(episode_id, str) and episode_id.strip().lstrip("-").isdigit():
        return int(episode_id)

    episodes_jsonl = Path(dataset_path) / "meta" / "episodes.jsonl"
    if not episodes_jsonl.exists():
        raise ValueError(
            f"episode_id {episode_id!r} is not an integer and {episodes_jsonl} "
            f"doesn't exist to resolve it by name. Verify how the encoded dataset "
            f"preserves the original session identity before running eval."
        )
    records = []
    with open(episodes_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    for rec in records:
        for v in rec.values():
            if isinstance(v, str) and episode_id in v:
                if "episode_index" in rec:
                    return int(rec["episode_index"])
    sample = records[:3]
    raise ValueError(
        f"Could not resolve episode_id {episode_id!r} to an episode_index in "
        f"{episodes_jsonl}. Sample records for reference: {sample}. "
        f"Fix this mapping (or extend resolve_episode_index) before trusting "
        f"any metrics from this dataset — do not guess silently."
    )


# --------------------------------------------------------------------------
# Rollout (requires the `groot` env / Isaac-GR00T importable)
# --------------------------------------------------------------------------


@dataclass
class RolloutResult:
    gt_action: np.ndarray  # (T, total_dim)
    pred_action: np.ndarray  # (T, total_dim)
    state_joints: np.ndarray  # (T, state_dim)


def run_rollout(
    policy,
    loader,
    episode_idx: int,
    embodiment_tag,
    execution_horizon: int,
    steps: int,
) -> RolloutResult:
    """Adapted from Isaac-GR00T's `gr00t/eval/open_loop_eval.py:evaluate_single_trajectory`,
    generalized to keep the full per-step arrays (not just aggregate MSE/MAE)."""
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from copy import deepcopy

    traj = loader[episode_idx]
    traj_length = len(traj)
    actual_steps = min(steps, traj_length)

    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = loader.modality_configs["action"].modality_keys

    # Fail fast if execution_horizon doesn't fit the model's predicted chunk
    # (confirmed API: gr00t/eval/open_loop_eval.py uses from_modality_config).
    PolicyHorizonSpec.from_modality_config(loader.modality_configs, n_action_steps=execution_horizon)

    modality_configs_no_action = deepcopy(loader.modality_configs)
    modality_configs_no_action.pop("action")

    pred_action_across_time = []
    policy.reset()
    for step_count in range(0, actual_steps, execution_horizon):
        data_point = extract_step_data(
            traj, step_count, modality_configs_no_action, embodiment_tag
        )
        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        action_chunk_raw, _ = policy.get_action(parsed_obs)
        action_chunk = {f"action.{k}": action_chunk_raw[k][0] for k in action_chunk_raw}
        for j in range(execution_horizon):
            if step_count + j >= actual_steps:
                break
            concat_pred = np.concatenate(
                [
                    np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                    for key in action_keys
                ],
                axis=0,
            )
            pred_action_across_time.append(concat_pred)

    def extract_cols(df, columns):
        np_dict = {c: np.vstack([np.asarray(a) for a in df[c]]) for c in columns}
        return np.concatenate([np_dict[c] for c in columns], axis=-1)

    state_joints = extract_cols(traj, [f"state.{k}" for k in state_keys])[:actual_steps]
    gt_action = extract_cols(traj, [f"action.{k}" for k in action_keys])[:actual_steps]
    pred_action = np.array(pred_action_across_time)[:actual_steps]

    assert gt_action.shape == pred_action.shape, (
        f"shape mismatch gt={gt_action.shape} pred={pred_action.shape} "
        f"(episode {episode_idx})"
    )
    return RolloutResult(gt_action=gt_action, pred_action=pred_action, state_joints=state_joints)


# --------------------------------------------------------------------------
# Per-episode metrics -> long-format rows
# --------------------------------------------------------------------------


def compute_episode_metrics(
    result: RolloutResult,
    layout: ActionLayout,
    checkpoint_label: str,
    task: str,
    episode_id: int,
    execution_horizon: int,
    grasp_tol_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = dict(layout.blocks)
    blocks["all"] = (0, layout.total_dim)

    for block_name, block in blocks.items():
        pred_b = block_slice(result.pred_action, block)
        gt_b = block_slice(result.gt_action, block)

        mse = float(np.mean((pred_b - gt_b) ** 2))
        mae = float(np.mean(np.abs(pred_b - gt_b)))
        fpe = final_position_error(pred_b, gt_b)
        jerk_pred = mean_squared_jerk(pred_b)
        jerk_gt = mean_squared_jerk(gt_b)

        base = dict(
            checkpoint=checkpoint_label,
            task=task,
            episode_id=episode_id,
            block=block_name,
        )
        rows.append({**base, "metric": "mse", "horizon_idx": -1, "value": mse})
        rows.append({**base, "metric": "mae", "horizon_idx": -1, "value": mae})
        rows.append({**base, "metric": "final_position_error", "horizon_idx": -1, "value": fpe})
        rows.append(
            {**base, "metric": "mean_squared_jerk_pred", "horizon_idx": -1, "value": jerk_pred}
        )
        rows.append({**base, "metric": "mean_squared_jerk_gt", "horizon_idx": -1, "value": jerk_gt})
        rows.append(
            {
                **base,
                "metric": "jerk_ratio_pred_over_gt",
                "horizon_idx": -1,
                "value": (jerk_pred / jerk_gt) if jerk_gt not in (0, float("nan")) else float("nan"),
            }
        )

        horizon_curve = error_vs_horizon(pred_b, gt_b, execution_horizon)
        for h_idx, v in enumerate(horizon_curve):
            rows.append({**base, "metric": "mse_at_horizon_idx", "horizon_idx": h_idx, "value": float(v)})

        if block_name != "all" and is_hand_block(block_name):
            gt_signal = hand_closure_signal(gt_b)
            pred_signal = hand_closure_signal(pred_b)
            gt_events = detect_close_events(gt_signal)
            pred_events = detect_close_events(pred_signal)
            grasp_metrics = match_events(gt_events, pred_events, tol_steps=grasp_tol_steps)
            for k, v in grasp_metrics.items():
                rows.append({**base, "metric": k, "horizon_idx": -1, "value": v})

    return rows


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def make_overlay_plot(
    result: RolloutResult,
    layout: ActionLayout,
    checkpoint_label: str,
    task: str,
    episode_id: int,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Classify blocks by name (see is_latent_block / is_hand_block) rather than
    # assuming exact names, so plots attach correctly whatever the layout calls
    # its blocks and however many hand dims it has.
    latent_items = [(n, b) for n, b in layout.blocks.items() if is_latent_block(n)]
    hand_items = [(n, b) for n, b in layout.blocks.items() if is_hand_block(n)]

    n_rows = max(1, len(latent_items) + len(hand_items))
    fig, axes = plt.subplots(nrows=n_rows, ncols=1, figsize=(9, 3.2 * n_rows))
    if n_rows == 1:
        axes = [axes]
    ax_i = 0

    for name, block in latent_items:
        pred_l = block_slice(result.pred_action, block)
        gt_l = block_slice(result.gt_action, block)
        ax = axes[ax_i]
        ax.plot(np.linalg.norm(gt_l, axis=-1), label="GT |block|_2")
        ax.plot(np.linalg.norm(pred_l, axis=-1), label="pred |block|_2", linestyle="--")
        ax.set_title(
            f"'{name}' latent block (SONIC latent-space L2 norm, NOT decoded wrist xyz "
            f"— see script docstring)"
        )
        ax.set_ylabel("L2 norm")
        ax.legend()
        ax_i += 1

    for name, block in hand_items:
        pred_h = hand_closure_signal(block_slice(result.pred_action, block))
        gt_h = hand_closure_signal(block_slice(result.gt_action, block))
        ax = axes[ax_i]
        ax.plot(gt_h, label="GT closure signal")
        ax.plot(pred_h, label="pred closure signal", linestyle="--")
        for ev in detect_close_events(gt_h):
            ax.axvline(ev.step, color="green" if ev.kind == "close" else "gray", alpha=0.4)
        for ev in detect_close_events(pred_h):
            ax.axvline(ev.step, color="red" if ev.kind == "close" else "orange", alpha=0.4, linestyle=":")
        ax.set_title(f"'{name}' open/close trace (green/gray = GT close/open, red/orange = pred)")
        ax.legend()
        ax_i += 1

    fig.suptitle(f"{checkpoint_label} | task={task} | episode={episode_id}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_checkpoint_arg(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--checkpoint expects LABEL=PATH, got: {spec}")
    label, path = spec.split("=", 1)
    return label, path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--action-layout", default="results/action_layout.json")
    p.add_argument("--eval-split", default="results/eval_split.json")
    p.add_argument("--dataset-path", default=None, help="Fallback dataset root if eval_split.json doesn't carry per-task paths")
    p.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint_arg,
        default=[],
        help="LABEL=PATH, repeatable, e.g. --checkpoint base_zeroshot_proxy=/path --checkpoint T1=/path",
    )
    p.add_argument("--embodiment-tag", default="UNITREE_G1_SONIC")
    p.add_argument("--execution-horizon", type=int, default=None, help="Defaults to action_layout.json chunk_length")
    p.add_argument("--steps", type=int, default=400, help="Max steps per episode (capped by episode length)")
    p.add_argument("--denoising-steps", type=int, default=4)
    p.add_argument("--grasp-tol-steps", type=int, default=8)
    p.add_argument("--output-csv", default="results/openloop_metrics.csv")
    p.add_argument("--plot-dir", default="results/openloop_plots")
    p.add_argument("--episodes-per-task-for-plots", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--isaac-gr00t-repo",
        default=os.environ.get(
            "ISAAC_GR00T_REPO",
            str(Path.home() / "g1_sonic_system1" / "repos" / "Isaac-GR00T"),
        ),
    )
    p.add_argument("--dump-raw-npz", default=None, help="Optional dir to dump raw pred/GT arrays per episode for later re-analysis (e.g. Phase C decode)")
    args = p.parse_args()

    sys.path.insert(0, args.isaac_gr00t_repo)

    layout = load_action_layout(args.action_layout)
    log.info("Loaded action layout from %s: %s (total_dim=%d, chunk_length=%d)",
              layout.source, layout.blocks, layout.total_dim, layout.chunk_length)

    execution_horizon = args.execution_horizon or layout.chunk_length

    eval_split = load_eval_split(args.eval_split, default_dataset_path=args.dataset_path)
    log.info("Eval split: %d tasks -> %s", len(eval_split), {k: len(v["episode_ids"]) for k, v in eval_split.items()})

    if not args.checkpoint:
        raise SystemExit("Pass at least one --checkpoint LABEL=PATH")

    # Imports that require the `groot` env.
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)

    all_rows: list[dict[str, Any]] = []
    plot_dir = Path(args.plot_dir)

    for label, ckpt_path in args.checkpoint:
        log.info("=== Loading checkpoint %s from %s ===", label, ckpt_path)
        policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=ckpt_path, device=args.device)
        policy.model.action_head.num_inference_timesteps = args.denoising_steps

        for task, spec in eval_split.items():
            ds_path = spec["dataset_path"]
            if ds_path is None:
                raise ValueError(
                    f"No dataset_path for task {task!r} in eval_split.json and no --dataset-path given."
                )
            modality = policy.get_modality_config()
            loader = LeRobotEpisodeLoader(dataset_path=ds_path, modality_configs=modality)

            for i, raw_episode_id in enumerate(spec["episode_ids"]):
                try:
                    episode_idx = resolve_episode_index(loader, raw_episode_id, ds_path)
                except ValueError as e:
                    log.error(
                        "Skipping episode %r for task %s — could not resolve to a dataset "
                        "index: %s", raw_episode_id, task, e,
                    )
                    continue
                if episode_idx >= len(loader):
                    log.warning("Episode %r (resolved idx %d) out of range for task %s (dataset has %d) — skipping",
                                raw_episode_id, episode_idx, task, len(loader))
                    continue
                log.info("[%s] task=%s episode=%r (resolved idx=%d) (%d/%d)",
                          label, task, raw_episode_id, episode_idx, i + 1, len(spec["episode_ids"]))
                result = run_rollout(
                    policy, loader, episode_idx, embodiment_tag, execution_horizon, args.steps
                )
                # Use the original (human-readable) episode identifier in outputs.
                episode_label = raw_episode_id
                rows = compute_episode_metrics(
                    result, layout, label, task, episode_label, execution_horizon, args.grasp_tol_steps
                )
                all_rows.extend(rows)

                if i < args.episodes_per_task_for_plots:
                    plot_path = plot_dir / task / f"{label}_ep{episode_label}.png"
                    make_overlay_plot(result, layout, label, task, episode_label, plot_path)

                if args.dump_raw_npz:
                    dump_dir = Path(args.dump_raw_npz)
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        dump_dir / f"{label}__{task}__ep{episode_label}.npz",
                        pred_action=result.pred_action,
                        gt_action=result.gt_action,
                        state_joints=result.state_joints,
                    )

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    except ImportError:
        import csv

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else [])
            writer.writeheader()
            writer.writerows(all_rows)
    log.info("Wrote %d metric rows to %s", len(all_rows), out_csv)


if __name__ == "__main__":
    main()
