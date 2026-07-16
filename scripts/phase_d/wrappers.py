#!/usr/bin/env python3
"""
Phase D — inference-time wrappers.

Turn a policy's *chunk-prediction callable* into a stitched / reranked action
STREAM, pluggable into scripts/eval_openloop.py (Phase A) and
scripts/run_hierarchy.py (Phase C) without either script knowing about the
internals.

Contract expected of the caller's policy:
    predict_chunk_fn(obs, seed=None) -> np.ndarray of shape [chunk_length, total_dim]
The array is the full concatenated action (all blocks, in layout order) -- the
SAME concatenation scripts/eval_openloop.py builds via `action_keys`. Adapters
`chunk_dict_to_array` / `chunk_array_to_dict` convert to/from GR00T's per-
modality dict form ({modality_key: [chunk_length, dim]}). ``seed`` is honored
only by stochastic heads; best-of-N passes per-(step,k) seeds.

Two controllers (choose per run):
  * RecedingHorizonController: predict chunk -> [optional best-of-N rerank] ->
    RTC-style stitch -> execute ``execute`` steps -> replan. Discrete-safe latent
    handling + FSQ snap come from stitching.py / fsq.py.
  * TemporalEnsembleController: predict a chunk EVERY step, temporally ensemble
    overlapping chunks. Latent (FSQ) dims are ``newest_only`` by default (the
    discrete gate) so tokens are never averaged off-grid.

Both are stateful; call ``reset()`` per episode and feed observations per step.
The controllers never call the policy directly except through predict_chunk_fn,
so they are agnostic to GR00T / ACT / any chunked policy.

INTEGRATION NOTES
  * eval_openloop.py (Phase A): its rollout already does
        action_chunk_raw, _ = policy.get_action(parsed_obs)
        chunk = concat over action_keys of action_chunk_raw[k][0]
    Wrap that as
        predict_chunk_fn(obs, seed=None):
            <optionally torch.manual_seed(seed)>       # see seeding note below
            raw, _ = policy.get_action(obs)
            return chunk_dict_to_array({k: raw[k] for k in raw}, layout)
    then drive RecedingHorizonController.step(obs, i) (execute-step cadence) or
    TemporalEnsembleController.step(obs, t) (per-step). The concat order here
    matches eval_openloop's action_keys concat as long as the layout block order
    matches (it does -- both come from action_layout.json).
  * run_hierarchy.py (Phase C): same predict_chunk_fn contract; pass a real
    SonicOnnxDecoder for decoded-pose blending / roundtrip reranking, and (if
    operating on decoded root/EEF poses) set ensemble quat_groups.
  * SEEDING for best-of-N: GR00T's flow head is stochastic via the torch RNG.
    predict_chunk_fn must make ``seed`` actually change the sample, e.g.
    ``torch.manual_seed(seed)`` (and cuda manual_seed) right before
    ``policy.get_action``. best_of_n derives seeds from (step, k) via
    reranker.candidate_seed -- never a single reused seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .layout import ActionLayout
from .stitching import RecedingHorizonStitcher, StitchConfig, StitchResult
from .ensembler import TemporalEnsembler
from .reranker import RerankConfig, RunningStats, best_of_n

def chunk_dict_to_array(chunk: dict, layout: ActionLayout) -> np.ndarray:
    """{block_name: [chunk_length, dim]} -> [chunk_length, total_dim] in layout
    order. Accepts GR00T's [1, chunk_length, dim] by squeezing a leading 1."""
    parts = []
    for b in layout.blocks:
        v = np.asarray(chunk[b.name])
        if v.ndim == 3 and v.shape[0] == 1:
            v = v[0]
        if v.ndim == 1:
            v = v[:, None]
        parts.append(v)
    return np.concatenate(parts, axis=-1)

def chunk_array_to_dict(arr: np.ndarray, layout: ActionLayout) -> dict:
    """[chunk_length, total_dim] -> {block_name: [chunk_length, dim]}."""
    return {b.name: np.asarray(arr)[:, b.slice] for b in layout.blocks}

@dataclass
class WrapperConfig:
    mode: str = "receding_horizon"
    execute: int = 8
    stitch: StitchConfig = field(default_factory=StitchConfig)
    use_best_of_n: bool = False
    rerank: RerankConfig = field(default_factory=RerankConfig)
    ensemble_m: float = 0.01
    ensemble_max_timesteps: int = 100000

    ensemble_newest_only_dims: Optional[Sequence[int]] = None
    ensemble_quat_groups: Optional[Sequence[Sequence[int]]] = None

class RecedingHorizonController:
    def __init__(
        self,
        predict_chunk_fn: Callable[..., np.ndarray],
        layout: ActionLayout,
        cfg: Optional[WrapperConfig] = None,
        decoder=None,
        stats: Optional[RunningStats] = None,
        joint_limits: Optional[tuple] = None,
    ):
        self.predict = predict_chunk_fn
        self.layout = layout
        self.cfg = cfg or WrapperConfig()
        self.decoder = decoder
        self.stats = stats
        self.joint_limits = joint_limits
        self.stitcher = RecedingHorizonStitcher(
            layout, self.cfg.stitch, execute=self.cfg.execute, decoder=decoder
        )
        self.reset()

    def reset(self) -> None:
        self.stitcher.reset()
        self._prev_last: Optional[np.ndarray] = None
        self._replan_count = 0

    def step(self, obs, step_index: Optional[int] = None) -> StitchResult:
        """One replan: predict (best-of-N optionally), stitch, return the executed
        slice [execute, total_dim] as a StitchResult. ``step_index`` seeds
        best-of-N; defaults to an internal replan counter."""
        step_index = self._replan_count if step_index is None else step_index

        if self.cfg.use_best_of_n:
            predict_fn = lambda seed: self.predict(obs, seed)
            res = best_of_n(
                predict_fn, step_index, self.layout, self.cfg.rerank,
                prev_last=self._prev_last, decoder=self.decoder,
                stats=self.stats, joint_limits=self.joint_limits,
            )
            chunk = res["candidate"]
        else:
            chunk = np.asarray(self.predict(obs), dtype=np.float64)

        stitched = self.stitcher.step(chunk)
        self._prev_last = stitched.actions[-1].copy()
        self._replan_count += 1
        return stitched

class TemporalEnsembleController:
    def __init__(
        self,
        predict_chunk_fn: Callable[..., np.ndarray],
        layout: ActionLayout,
        cfg: Optional[WrapperConfig] = None,
    ):
        self.predict = predict_chunk_fn
        self.layout = layout
        self.cfg = cfg or WrapperConfig()
        newest = self.cfg.ensemble_newest_only_dims
        if newest is None:
            newest = layout.latent_indices()
        self.ens = TemporalEnsembler(
            action_dim=layout.total_dim,
            chunk_length=layout.chunk_length,
            max_timesteps=self.cfg.ensemble_max_timesteps,
            m=self.cfg.ensemble_m,
            newest_only_dims=newest,
            quat_groups=self.cfg.ensemble_quat_groups,
        )
        self.reset()

    def reset(self) -> None:
        self.ens.reset()

    def step(self, obs, t: int) -> np.ndarray:
        """Predict a chunk at time ``t``, add it, and return the ensembled action
        [total_dim] for step ``t``."""
        chunk = np.asarray(self.predict(obs), dtype=np.float64)
        self.ens.add_chunk(t, chunk)
        return self.ens.get_action(t)

def make_controller(predict_chunk_fn, layout, cfg=None, **kw):
    cfg = cfg or WrapperConfig()
    if cfg.mode == "receding_horizon":
        return RecedingHorizonController(predict_chunk_fn, layout, cfg, **kw)
    if cfg.mode == "temporal_ensemble":
        return TemporalEnsembleController(predict_chunk_fn, layout, cfg)
    raise ValueError(f"unknown wrapper mode {cfg.mode!r}")
