#!/usr/bin/env python3
"""
Phase D2 — best-of-N SONIC-aware reranking.

GR00T's flow-matching action head is stochastic: sampling K chunks per replan
(seeded off ``(step, k)`` -- NOT a single reused seed) gives K candidates. We
score each with a cheap, self-supervised ``candidate_cost`` and execute the
argmin. No ground truth is used (see the oracle note at the bottom).

``candidate_cost`` = weighted sum of:
  * boundary discontinuity : jump from the last executed action into the
    candidate's first step (continuity across the replan),
  * velocity               : mean-squared 1st difference (smoothness),
  * acceleration           : mean-squared 2nd difference (smoothness),
  * range                  : out-of-range penalty (latent beyond the FSQ
    alphabet; other blocks beyond running min/max if provided),
  * z-score                : per-dim outlier penalty vs running executed-action
    statistics (feasibility / distribution match),
  * SONIC roundtrip        : decode the candidate token chunk through the SONIC
    decoder and penalize *pose-space* discontinuity + joint-limit infeasibility
    -- the System-0-native score (only when a PoseDecoder is supplied).

Smoothness (velocity/accel/roundtrip) uses SEPARATE body-vs-hand weights: the
FSQ/latent (body) block should be smooth, while hand blocks legitimately snap
(grasp open/close), so hand smoothness is penalized far less.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .layout import ActionLayout

@dataclass
class RerankConfig:
    K: int = 4
    w_boundary: float = 1.0
    w_velocity: float = 1.0
    w_accel: float = 1.0
    w_range: float = 1.0
    w_zscore: float = 0.5
    w_roundtrip: float = 1.0
    body_smooth_scale: float = 1.0
    hand_smooth_scale: float = 0.2
    fsq_clamp: tuple = (-0.625, 0.625)
    base_seed: int = 0

@dataclass
class RunningStats:
    """Optional per-dim mean/std/min/max of executed actions, for z-score/range."""
    mean: np.ndarray
    std: np.ndarray
    lo: Optional[np.ndarray] = None
    hi: Optional[np.ndarray] = None

def candidate_seed(step: int, k: int, base: int = 0, K: int = 4) -> int:
    """Deterministic per-(step,k) seed. NOT a fixed reused seed -- each candidate
    of each replan gets its own sampling seed so best-of-N sees real diversity."""
    return int((base * 1_000_003 + step * (K + 1) + k) & 0x7FFFFFFF)

def _msq_diff(x: np.ndarray, n: int) -> float:
    if x.shape[0] <= n:
        return 0.0
    d = np.diff(x, n=n, axis=0)
    return float(np.mean(d ** 2))

def candidate_cost(
    candidate: np.ndarray,
    layout: ActionLayout,
    cfg: Optional[RerankConfig] = None,
    prev_last: Optional[np.ndarray] = None,
    decoder=None,
    stats: Optional[RunningStats] = None,
    joint_limits: Optional[tuple] = None,
) -> tuple[float, dict]:
    """Return (total_cost, breakdown). ``candidate`` is [chunk_length, total_dim].
    ``prev_last`` is the last executed action [total_dim] (for boundary)."""
    cfg = cfg or RerankConfig()
    x = np.asarray(candidate, dtype=np.float64)
    bd = {"boundary": 0.0, "velocity": 0.0, "accel": 0.0, "range": 0.0, "zscore": 0.0, "roundtrip": 0.0}

    for b in layout.blocks:
        sl = b.slice
        xb = x[:, sl]
        smooth = cfg.hand_smooth_scale if b.is_hand else cfg.body_smooth_scale

        if prev_last is not None:
            pl = np.asarray(prev_last, dtype=np.float64)[sl]
            bd["boundary"] += smooth * float(np.mean((xb[0] - pl) ** 2))

        bd["velocity"] += smooth * _msq_diff(xb, 1)
        bd["accel"] += smooth * _msq_diff(xb, 2)

        if b.is_latent:
            lo, hi = cfg.fsq_clamp
            over = np.maximum(0.0, np.abs(xb) - max(abs(lo), abs(hi)))
            bd["range"] += float(np.mean(over ** 2))
        elif stats is not None and stats.lo is not None and stats.hi is not None:
            lo_b, hi_b = stats.lo[sl], stats.hi[sl]
            over = np.maximum(0.0, xb - hi_b) + np.maximum(0.0, lo_b - xb)
            bd["range"] += float(np.mean(over ** 2))

    if stats is not None:
        std = np.where(stats.std < 1e-6, 1.0, stats.std)
        z = (x - stats.mean[None, :]) / std[None, :]
        bd["zscore"] = float(np.mean(z ** 2))

    if decoder is not None and len(layout.latent_blocks) == 1:
        lb = layout.latent_blocks[0].slice
        decoder.reset()
        pose = np.asarray(decoder.decode_chunk(x[:, lb]), dtype=np.float64)
        rt = _msq_diff(pose, 1) + _msq_diff(pose, 2)
        if joint_limits is not None:
            lo_j, hi_j = joint_limits
            infeas = np.maximum(0.0, pose - hi_j[None, :]) + np.maximum(0.0, lo_j[None, :] - pose)
            rt += float(np.mean(infeas ** 2))
        bd["roundtrip"] = float(rt)

    total = (
        cfg.w_boundary * bd["boundary"]
        + cfg.w_velocity * bd["velocity"]
        + cfg.w_accel * bd["accel"]
        + cfg.w_range * bd["range"]
        + cfg.w_zscore * bd["zscore"]
        + cfg.w_roundtrip * bd["roundtrip"]
    )
    return float(total), bd

def rerank(
    candidates: list[np.ndarray],
    layout: ActionLayout,
    cfg: Optional[RerankConfig] = None,
    prev_last: Optional[np.ndarray] = None,
    decoder=None,
    stats: Optional[RunningStats] = None,
    joint_limits: Optional[tuple] = None,
) -> dict:
    """Score every candidate; return {best_index, costs, breakdowns}."""
    cfg = cfg or RerankConfig()
    costs, breaks = [], []
    for cand in candidates:
        c, bd = candidate_cost(cand, layout, cfg, prev_last, decoder, stats, joint_limits)
        costs.append(c)
        breaks.append(bd)
    best = int(np.argmin(costs))
    return {"best_index": best, "costs": costs, "breakdowns": breaks, "best_cost": costs[best]}

def best_of_n(
    predict_fn: Callable[[int], np.ndarray],
    step: int,
    layout: ActionLayout,
    cfg: Optional[RerankConfig] = None,
    prev_last: Optional[np.ndarray] = None,
    decoder=None,
    stats: Optional[RunningStats] = None,
    joint_limits: Optional[tuple] = None,
) -> dict:
    """Sample K candidates via ``predict_fn(seed)`` with seeds derived from
    (step, k), rerank, and return the winner. Result adds 'candidate' and
    'seeds'."""
    cfg = cfg or RerankConfig()
    seeds = [candidate_seed(step, k, cfg.base_seed, cfg.K) for k in range(cfg.K)]
    candidates = [np.asarray(predict_fn(s), dtype=np.float64) for s in seeds]
    res = rerank(candidates, layout, cfg, prev_last, decoder, stats, joint_limits)
    res["candidate"] = candidates[res["best_index"]]
    res["candidates"] = candidates
    res["seeds"] = seeds
    return res

def oracle_best_of_k(candidates: list[np.ndarray], gt_chunk: np.ndarray) -> dict:
    """UPPER-BOUND ONLY. Picks the candidate closest (MSE) to the held-out
    ground-truth chunk. This is NOT a deployable policy -- it peeks at GT. Report
    it strictly as an 'oracle best-of-K' ceiling row alongside the real
    (GT-free) best_of_n result, never as the selection method."""
    gt = np.asarray(gt_chunk, dtype=np.float64)
    errs = [float(np.mean((np.asarray(c, dtype=np.float64) - gt) ** 2)) for c in candidates]
    best = int(np.argmin(errs))
    return {"best_index": best, "errors": errs, "candidate": candidates[best], "is_oracle": True}
