#!/usr/bin/env python3
"""
Phase D2 — FSQ manifold projection.

The 64-dim motion_token block is FSQ-discrete (``latent_continuous == False``):
32-level FSQ, values on a 1/16 = 0.0625 grid, observed range +/-0.625. A flow /
regression head predicts *continuous* floats that only *approximately* land on
that grid. Feeding an off-grid value to the SONIC decoder is feeding it a token
the codebook never emits -> meaningless / jittery motion.

FSQ manifold projection is the discrete analog of "manifold projection" and is
near-zero risk: snap each predicted latent value to its nearest FSQ grid point
(``round(v / step) * step``, i.e. ``round(v*16)/16`` for step=1/16) and clamp to
the observed value range. This is a *deterministic, idempotent* transform.

It is applied ONLY to the latent (discrete/FSQ) dims — hand-joint dims are
continuous and are left untouched.

CLAMP NOTE: default clamp is the *observed* alphabet range (+/-0.625, 21 levels).
FSQ theory allows 32 levels (~[-1, 1)). If a fine-tuned checkpoint legitimately
emits |v| > 0.625, clamping here would clip it; expose ``clamp`` so a caller can
widen it. Default follows the runbook (+/-0.625) and the empirically observed
alphabet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def fsq_project_values(
    x: np.ndarray,
    step: float = 0.0625,
    clamp: Optional[tuple[float, float]] = (-0.625, 0.625),
) -> np.ndarray:
    """Snap ``x`` to the FSQ grid: round to nearest multiple of ``step`` then
    (optionally) clamp to ``clamp``. Pure/deterministic; shape-preserving."""
    x = np.asarray(x, dtype=np.float64)
    q = np.round(x / step) * step
    if clamp is not None:
        q = np.clip(q, clamp[0], clamp[1])
    # kill -0.0 and tiny fp dust so grid membership tests are exact
    q = q + 0.0
    return q.astype(np.float32)


def fsq_project_action(
    action: np.ndarray,
    latent_indices: Sequence[int],
    step: float = 0.0625,
    clamp: Optional[tuple[float, float]] = (-0.625, 0.625),
) -> np.ndarray:
    """Project only the latent dims of an action array (``[..., total_dim]``);
    leave every other dim (hands) untouched. Returns a copy."""
    out = np.array(action, dtype=np.float32, copy=True)
    if len(latent_indices) == 0:
        return out
    idx = np.asarray(list(latent_indices), dtype=int)
    out[..., idx] = fsq_project_values(out[..., idx], step=step, clamp=clamp)
    return out


@dataclass
class FsqProjectionReport:
    n_values: int
    frac_offgrid_before: float          # fraction of latent values not already on-grid
    mean_abs_shift: float               # mean |v - project(v)| over latent values
    max_abs_shift: float
    frac_clamped: float                 # fraction pushed by the clamp
    frac_offgrid_after: float           # should be ~0.0 (sanity)
    rms_shift: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def fsq_projection_report(
    latent_values: np.ndarray,
    step: float = 0.0625,
    clamp: Optional[tuple[float, float]] = (-0.625, 0.625),
    on_grid_atol: float = 1e-6,
) -> FsqProjectionReport:
    """Quantify the effect of projecting ``latent_values`` (any shape) onto the
    FSQ grid, so D2 can *report* how much off-grid jitter it removed."""
    x = np.asarray(latent_values, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return FsqProjectionReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    q = fsq_project_values(x, step=step, clamp=clamp).astype(np.float64)
    shift = np.abs(x - q)

    # off-grid (before) = distance to nearest grid point is > atol (ignoring clamp)
    round_only = np.round(x / step) * step
    offgrid_before = np.abs(x - round_only) > on_grid_atol

    if clamp is not None:
        clamped = (round_only < clamp[0] - on_grid_atol) | (round_only > clamp[1] + on_grid_atol)
    else:
        clamped = np.zeros_like(x, dtype=bool)

    # after projection: verify on-grid
    q_round = np.round(q / step) * step
    offgrid_after = np.abs(q - q_round) > on_grid_atol

    return FsqProjectionReport(
        n_values=int(n),
        frac_offgrid_before=float(offgrid_before.mean()),
        mean_abs_shift=float(shift.mean()),
        max_abs_shift=float(shift.max()),
        frac_clamped=float(clamped.mean()),
        frac_offgrid_after=float(offgrid_after.mean()),
        rms_shift=float(np.sqrt(np.mean(shift ** 2))),
    )
