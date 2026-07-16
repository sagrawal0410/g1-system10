#!/usr/bin/env python3
"""
Phase D1 — ACT-style temporal ensembler.

Overlapping action chunks predicted at successive timesteps are averaged with
exponential weights, which smooths the executed command stream. This reproduces
the reference (ACT / "temporal aggregation") behavior.

The ACT reference keeps a dense ``[T, T + chunk_length, action_dim]`` buffer
indexed by (query time, absolute target time). At our scale that is an
O(T^2 * D) memory bomb (T up to ~3000 frames, D=78 -> ~2.5 GB) and only a
width-``chunk_length`` band is ever populated (query ``t`` writes columns
``t .. t+chunk_length-1``). So this stores the SAME information in a banded /
sparse form (one chunk per query time) that is mathematically IDENTICAL to the
dense buffer -- same coverage, same weights, same output (the unit tests pin
this) -- but O(window * D) memory. Semantics:

  * ``add_chunk(t, chunk)`` records ``chunk`` as the prediction made at query
    time ``t`` (covering absolute times ``t .. t+len-1``). A per-chunk length is
    tracked so an all-zero prediction is never mistaken for "no prediction".
  * ``get_action(t)`` gathers every *live* prediction whose window covers step
    ``t`` (query times ``t - chunk_length + 1 .. t``), ordered oldest-first, and
    returns the exponentially-weighted mean
        w_i = exp(-m * i),  i = 0 is the OLDEST prediction,  m ~ 0.01
    (normalized to sum 1). With m ~ 0.01 the weights are near-uniform, slightly
    favoring older predictions -- matching the ACT reference.

Two extensions the runbook calls out:
  * ``newest_only_dims``: dims that must NOT be averaged and instead take the
    value from the *newest* covering prediction. This is the discrete gate for
    the FSQ latent block -- averaging discrete FSQ tokens produces off-grid
    garbage -- and is also right for grasp/close bits.
  * ``quat_groups``: index tuples (length-4) that are unit quaternions and must
    be averaged on the sphere (sign-align + normalized weighted mean +
    renormalize), NOT componentwise. Per the runbook this is ONLY meaningful in
    pose space (e.g. run_hierarchy operating on decoded root/EEF poses); it is
    unused for the raw 78-dim token+hand action (no quaternions there).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class TemporalEnsembler:
    def __init__(
        self,
        action_dim: int,
        chunk_length: int,
        max_timesteps: int = 100000,
        m: float = 0.01,
        newest_only_dims: Optional[Sequence[int]] = None,
        quat_groups: Optional[Sequence[Sequence[int]]] = None,
        prune: bool = True,
    ):
        self.action_dim = int(action_dim)
        self.chunk_length = int(chunk_length)
        self.max_timesteps = int(max_timesteps)  # sanity cap only (banded store, cheap)
        self.m = float(m)
        self.prune = prune
        self.newest_only_dims = np.asarray(sorted(set(newest_only_dims or [])), dtype=int)
        self.quat_groups = [tuple(int(i) for i in g) for g in (quat_groups or [])]
        for g in self.quat_groups:
            if len(g) != 4:
                raise ValueError(f"quat group {g} must have exactly 4 dims")
        excluded = set(self.newest_only_dims.tolist())
        for g in self.quat_groups:
            excluded.update(g)
        self._avg_dims = np.asarray([d for d in range(self.action_dim) if d not in excluded], dtype=int)
        self.reset()

    def reset(self) -> None:
        # query_time -> chunk [L, D]. Banded/sparse equivalent of the dense
        # [T, T+C, D] ACT buffer.
        self._chunks: dict[int, np.ndarray] = {}

    def add_chunk(self, t: int, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != self.action_dim:
            raise ValueError(f"chunk shape {chunk.shape} != (<=chunk_length, {self.action_dim})")
        if t >= self.max_timesteps:
            raise IndexError(f"query time t={t} >= max_timesteps={self.max_timesteps}")
        L = min(chunk.shape[0], self.chunk_length)
        self._chunks[int(t)] = chunk[:L].copy()
        if self.prune:
            # drop chunks that can no longer cover any t' >= (t - C + 1)
            cutoff = t - self.chunk_length
            for q in [q for q in self._chunks if q < cutoff]:
                del self._chunks[q]

    def _covering_rows(self, t: int) -> list[int]:
        """Query times (ascending == oldest-first) with a live prediction for t."""
        rows = []
        for q in range(max(0, t - self.chunk_length + 1), t + 1):
            ch = self._chunks.get(q)
            if ch is not None and (t - q) < ch.shape[0]:
                rows.append(q)
        return rows  # already ascending

    def coverage(self, t: int) -> int:
        return len(self._covering_rows(t))

    def get_action(self, t: int, require_coverage: bool = True) -> np.ndarray:
        rows = self._covering_rows(t)  # oldest-first
        n = len(rows)
        if n == 0:
            if require_coverage:
                raise AssertionError(
                    f"get_action(t={t}) has NO covering prediction. add_chunk at a query "
                    f"time in [{max(0, t - self.chunk_length + 1)}, {t}] first."
                )
            return np.full(self.action_dim, np.nan)

        stacked = np.stack([self._chunks[q][t - q] for q in rows], axis=0)  # [n, D], row 0 oldest
        w = np.exp(-self.m * np.arange(n, dtype=np.float64))
        w /= w.sum()

        out = np.zeros(self.action_dim, dtype=np.float64)
        if self._avg_dims.size:
            out[self._avg_dims] = (w[:, None] * stacked[:, self._avg_dims]).sum(axis=0)
        if self.newest_only_dims.size:
            out[self.newest_only_dims] = stacked[-1, self.newest_only_dims]  # newest covering
        for g in self.quat_groups:
            out[list(g)] = _weighted_quat_mean(stacked[:, list(g)], w)
        return out


def _weighted_quat_mean(quats: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted average of unit quaternions. Sign-align to the newest (last)
    quaternion to resolve the double-cover, normalize each, then take the
    weighted mean and renormalize. (A robust, cheap approximation to the
    Markley eigenvector solution; exact when the quats are close, which is the
    ensembling regime.)"""
    q = np.asarray(quats, dtype=np.float64).copy()
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    q /= norms
    ref = q[-1]  # newest
    signs = np.sign((q * ref).sum(axis=1))
    signs[signs == 0] = 1.0
    q *= signs[:, None]
    mean = (w[:, None] * q).sum(axis=0)
    nm = np.linalg.norm(mean)
    if nm < 1e-12:
        return ref
    return mean / nm
