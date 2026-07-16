"""Shared test helpers for Phase D unit tests (no pytest required)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from phase_d.layout import ActionLayout, Block, load_layout

FSQ_STEP = 0.0625
FSQ_CLAMP = (-0.625, 0.625)
FSQ_GRID = np.round(np.arange(-0.625, 0.625 + 1e-9, FSQ_STEP) / FSQ_STEP) * FSQ_STEP

def synthetic_layout(hand_dim: int = 7) -> ActionLayout:
    """78-dim (or variable-hand) synthetic layout matching the real schema:
    motion_token(64, discrete) + left/right hand (continuous)."""
    lat = Block("motion_token", 0, 64, continuous=False)
    lh = Block("left_hand_joints", 64, 64 + hand_dim, continuous=True)
    rh = Block("right_hand_joints", 64 + hand_dim, 64 + 2 * hand_dim, continuous=True)
    lay = ActionLayout(
        total_dim=64 + 2 * hand_dim,
        chunk_length=40,
        blocks=[lat, lh, rh],
        latent_continuous=False,
        fsq_step=FSQ_STEP,
        fsq_range=FSQ_CLAMP,
        fsq_num_levels=32,
        source="synthetic",
    )
    lay.validate()
    return lay

def real_layout_path() -> Path | None:
    """Locate results/action_layout.json relative to the repo, if present."""
    here = Path(__file__).resolve()
    for up in here.parents:
        cand = up / "results" / "action_layout.json"
        if cand.exists():
            return cand
    env = os.environ.get("ACTION_LAYOUT")
    if env and Path(env).exists():
        return Path(env)
    return None

def on_grid(x: np.ndarray, step: float = FSQ_STEP, atol: float = 1e-6) -> bool:
    x = np.asarray(x, dtype=np.float64)
    return bool(np.all(np.abs(x / step - np.round(x / step)) < atol))

def make_token_chunk(T: int, D_latent: int, rng, on_grid_vals: bool = True) -> np.ndarray:
    """A smooth-ish token chunk on the FSQ grid."""
    base = rng.integers(-6, 7, size=D_latent) * FSQ_STEP
    drift = np.cumsum(rng.integers(-1, 2, size=(T, D_latent)) * FSQ_STEP, axis=0)
    chunk = np.clip(base[None, :] + drift, FSQ_CLAMP[0], FSQ_CLAMP[1])
    if on_grid_vals:
        chunk = np.round(chunk / FSQ_STEP) * FSQ_STEP
    return chunk
