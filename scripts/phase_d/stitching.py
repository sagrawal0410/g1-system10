#!/usr/bin/env python3
"""
Phase D1 — receding-horizon replanning + overlap stitching (RTC-style).

Predict a length-``chunk_length`` chunk, execute ``execute`` steps, replan from
the next observation, and stitch the new chunk onto the still-in-flight tail of
the previous plan so the executed command stream is continuous across replans:
  * freeze the first ``freeze`` (2-4) steps to the previous plan (they are
    already committed / in-flight),
  * linear-blend over a ``blend_len`` (16-24) overlap from previous -> new,
  * then run the fresh chunk.

CRITICAL DISCRETE GATE (latent_continuous == False):
The 64-dim motion_token block is FSQ-discrete. Linearly interpolating raw FSQ
latents produces off-grid values the SONIC decoder never sees -> meaningless
motion. So the latent block is NOT linearly blended. Its ``latent_strategy``:
  * "newest_only" (DEFAULT): freeze to the previous plan for ``freeze`` steps,
    then hard-switch to the newest chunk. No interpolation; every executed
    latent stays exactly on the FSQ grid. This is the safe, decoder-free
    default the runbook asks for.
  * "decoded_pose": decode the previous-tail and new latent chunks through the
    SONIC decoder and linearly blend the resulting *body-joint pose*
    trajectories (continuous, safe to blend). The token stream still switches
    newest-only (tokens can't represent a blended pose); the blended poses are
    returned as ``body_pose_override`` for a closed-loop executor (Phase C) to
    drive the body directly across the overlap. Higher fidelity, needs a
    decoder + pose-level executor.
  * "fsq_snap_blend": linear-blend then re-snap to the FSQ grid. On-grid output
    but the interpolation itself is in latent space -- lower fidelity; provided
    for ablation, not recommended.
The executed latent is additionally snapped to the FSQ grid by default
(``snap_latent_to_grid``) so raw off-grid predictions never reach the decoder.

Hands (continuous) ARE blended, but WEAKLY: ``hand_alpha = alpha ** hand_alpha_power``
(default 0.5) so a grasp closure isn't smoothed into a failed half-grasp. Hands
may be zeroed (body-only run) or per-task-normalized (hands-in run); block
identity/width is read from the layout, never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .layout import ActionLayout
from .fsq import fsq_project_values

@dataclass
class StitchConfig:
    freeze: int = 3
    blend_len: int = 20
    latent_strategy: str = "newest_only"
    hand_alpha_power: float = 0.5
    snap_latent_to_grid: bool = True
    fsq_step: float = 0.0625
    fsq_clamp: tuple = (-0.625, 0.625)

@dataclass
class StitchResult:
    actions: np.ndarray
    body_pose_override: Optional[np.ndarray] = None
    latent_strategy: str = "newest_only"

def _alpha_ramp(overlap: int, freeze: int, blend_len: int) -> np.ndarray:
    """alpha[i] in [0,1] over the overlap region: 0 while frozen, linear ramp
    across blend_len, then 1."""
    a = np.zeros(overlap, dtype=np.float64)
    for i in range(overlap):
        if i < freeze:
            a[i] = 0.0
        elif i < freeze + blend_len:
            a[i] = (i - freeze + 1) / (blend_len + 1)
        else:
            a[i] = 1.0
    return a

def rtc_style_stitch(
    new_chunk: np.ndarray,
    prev_overlap: Optional[np.ndarray],
    layout: ActionLayout,
    cfg: Optional[StitchConfig] = None,
    decoder=None,
) -> StitchResult:
    """Stitch ``new_chunk`` [chunk_length, D] onto ``prev_overlap`` [overlap, D]
    (the previous plan's predictions for the timesteps the new chunk also
    covers, aligned so prev_overlap[i] and new_chunk[i] address the same
    absolute time). ``prev_overlap=None`` (first plan) returns the new chunk
    (latent snapped to grid if enabled)."""
    cfg = cfg or StitchConfig()
    new_chunk = np.asarray(new_chunk, dtype=np.float64)
    T, D = new_chunk.shape
    out = new_chunk.copy()
    body_override = None

    def _snap(latent_vals):
        if not cfg.snap_latent_to_grid:
            return latent_vals
        return fsq_project_values(latent_vals, step=cfg.fsq_step, clamp=cfg.fsq_clamp).astype(np.float64)

    latent_idx = np.asarray(layout.latent_indices(), dtype=int)

    if prev_overlap is None:
        if latent_idx.size:
            out[:, latent_idx] = _snap(out[:, latent_idx])
        return StitchResult(actions=out, latent_strategy=cfg.latent_strategy)

    prev_overlap = np.asarray(prev_overlap, dtype=np.float64)
    overlap = min(prev_overlap.shape[0], T)
    if overlap == 0:
        if latent_idx.size:
            out[:, latent_idx] = _snap(out[:, latent_idx])
        return StitchResult(actions=out, latent_strategy=cfg.latent_strategy)

    freeze = min(cfg.freeze, overlap)
    alpha = _alpha_ramp(overlap, freeze, cfg.blend_len)

    for b in layout.blocks:
        sl = b.slice
        prev_b = prev_overlap[:overlap, sl]
        new_b = new_chunk[:overlap, sl]

        if b.is_latent:
            if cfg.latent_strategy in ("newest_only", "decoded_pose"):

                merged = new_b.copy()
                merged[:freeze] = prev_b[:freeze]
            elif cfg.latent_strategy == "fsq_snap_blend":
                a = alpha[:, None]
                merged = (1.0 - a) * prev_b + a * new_b
                merged = fsq_project_values(merged, step=cfg.fsq_step, clamp=cfg.fsq_clamp).astype(np.float64)
            else:
                raise ValueError(f"unknown latent_strategy {cfg.latent_strategy!r}")
            out[:overlap, sl] = merged
        elif b.is_hand:
            a = (alpha ** cfg.hand_alpha_power)[:, None]
            out[:overlap, sl] = (1.0 - a) * prev_b + a * new_b
        else:

            a = alpha[:, None]
            out[:overlap, sl] = (1.0 - a) * prev_b + a * new_b

    if latent_idx.size:
        out[:, latent_idx] = _snap(out[:, latent_idx])

    if cfg.latent_strategy == "decoded_pose":
        if decoder is None:
            raise ValueError("latent_strategy='decoded_pose' requires a PoseDecoder")
        if len(layout.latent_blocks) != 1:
            raise ValueError("decoded_pose expects exactly one latent block")
        lb = layout.latent_blocks[0].slice
        decoder.reset()
        pose_prev = decoder.decode_chunk(prev_overlap[:overlap, lb])
        decoder.reset()
        pose_new = decoder.decode_chunk(new_chunk[:overlap, lb])
        body_dim = pose_new.shape[1]
        body_override = np.full((T, body_dim), np.nan, dtype=np.float64)
        a = alpha[:, None]
        body_override[:overlap] = (1.0 - a) * pose_prev + a * pose_new

    return StitchResult(actions=out, body_pose_override=body_override, latent_strategy=cfg.latent_strategy)

class RecedingHorizonStitcher:
    """Stateful driver: feed it a freshly predicted chunk each replan; it stitches
    against the in-flight tail of the previous plan and returns the next
    ``execute`` actions to run. overlap == chunk_length - execute is implied by
    the replanning cadence."""

    def __init__(
        self,
        layout: ActionLayout,
        cfg: Optional[StitchConfig] = None,
        execute: int = 8,
        decoder=None,
    ):
        self.layout = layout
        self.cfg = cfg or StitchConfig()
        self.execute = int(execute)
        self.decoder = decoder
        self.reset()

    def reset(self) -> None:
        self._prev_plan: Optional[np.ndarray] = None

    def step(self, new_chunk: np.ndarray) -> StitchResult:
        """Stitch ``new_chunk`` and return the executed slice (first ``execute``
        steps) as a StitchResult; retains the stitched plan internally."""
        new_chunk = np.asarray(new_chunk, dtype=np.float64)
        if self._prev_plan is None:
            res = rtc_style_stitch(new_chunk, None, self.layout, self.cfg, self.decoder)
        else:
            prev_overlap = self._prev_plan[self.execute :]
            res = rtc_style_stitch(new_chunk, prev_overlap, self.layout, self.cfg, self.decoder)
        self._prev_plan = res.actions

        ex = min(self.execute, res.actions.shape[0])
        exec_pose = None
        if res.body_pose_override is not None:
            exec_pose = res.body_pose_override[:ex]
        return StitchResult(
            actions=res.actions[:ex],
            body_pose_override=exec_pose,
            latent_strategy=res.latent_strategy,
        )
