"""Unit tests: inference-time controllers + dict adapters (integration surface)."""
from __future__ import annotations

import numpy as np

from phase_d.wrappers import (
    RecedingHorizonController, TemporalEnsembleController, WrapperConfig, make_controller,
    chunk_dict_to_array, chunk_array_to_dict,
)
from phase_d.stitching import StitchConfig
from phase_d.reranker import RerankConfig
from ._common import synthetic_layout, make_token_chunk, on_grid

def test_dict_array_adapters_roundtrip():
    lay = synthetic_layout()
    rng = np.random.default_rng(0)
    arr = np.concatenate([make_token_chunk(40, 64, rng), rng.uniform(0, 1, (40, 14))], axis=1)
    d = chunk_array_to_dict(arr, lay)
    assert set(d.keys()) == {"motion_token", "left_hand_joints", "right_hand_joints"}
    assert d["motion_token"].shape == (40, 64)
    back = chunk_dict_to_array(d, lay)
    assert np.allclose(back, arr)

def test_dict_adapter_squeezes_groot_leading_batch():
    lay = synthetic_layout()
    d = {
        "motion_token": np.zeros((1, 40, 64)),
        "left_hand_joints": np.zeros((1, 40, 7)),
        "right_hand_joints": np.zeros((1, 40, 7)),
    }
    arr = chunk_dict_to_array(d, lay)
    assert arr.shape == (40, 78)

def _make_predict(rng):
    def predict(obs, seed=None):
        r = np.random.default_rng(seed if seed is not None else rng.integers(1 << 30))
        return np.concatenate([make_token_chunk(40, 64, r), r.uniform(0, 1, (40, 14))], axis=1)
    return predict

def test_receding_horizon_controller_streams():
    lay = synthetic_layout()
    rng = np.random.default_rng(1)
    cfg = WrapperConfig(mode="receding_horizon", execute=8, stitch=StitchConfig(freeze=3, blend_len=20))
    ctrl = RecedingHorizonController(_make_predict(rng), lay, cfg)
    ctrl.reset()
    stream = []
    for i in range(5):
        res = ctrl.step(obs={"dummy": i}, step_index=i)
        assert res.actions.shape == (8, 78)
        assert not np.isnan(res.actions).any()
        assert on_grid(res.actions[:, :64])
        stream.append(res.actions)
    full = np.concatenate(stream, axis=0)
    assert full.shape == (40, 78)

def test_receding_horizon_controller_best_of_n():
    lay = synthetic_layout()
    rng = np.random.default_rng(2)
    seen_seeds = []

    def predict(obs, seed=None):
        seen_seeds.append(seed)
        r = np.random.default_rng(seed if seed is not None else 0)
        return np.concatenate([make_token_chunk(40, 64, r), r.uniform(0, 1, (40, 14))], axis=1)

    cfg = WrapperConfig(mode="receding_horizon", execute=8, use_best_of_n=True,
                        rerank=RerankConfig(K=4))
    ctrl = RecedingHorizonController(predict, lay, cfg)
    ctrl.reset()
    res = ctrl.step(obs=None, step_index=0)
    assert res.actions.shape == (8, 78)

    assert len([s for s in seen_seeds if s is not None]) == 4
    assert len(set(seen_seeds)) == 4

def test_temporal_ensemble_controller_latent_stays_on_grid():
    lay = synthetic_layout()
    rng = np.random.default_rng(3)
    cfg = WrapperConfig(mode="temporal_ensemble", ensemble_m=0.01)
    ctrl = TemporalEnsembleController(_make_predict(rng), lay, cfg)
    ctrl.reset()
    outs = []
    for t in range(12):
        a = ctrl.step(obs=None, t=t)
        assert a.shape == (78,)
        assert not np.isnan(a).any()
        outs.append(a)
    outs = np.stack(outs)

    assert on_grid(outs[:, :64]), "ensembler must not average FSQ latent off-grid"

    assert np.isfinite(outs[:, 64:]).all()

def test_make_controller_factory():
    lay = synthetic_layout()
    rng = np.random.default_rng(4)
    rc = make_controller(_make_predict(rng), lay, WrapperConfig(mode="receding_horizon"))
    ec = make_controller(_make_predict(rng), lay, WrapperConfig(mode="temporal_ensemble"))
    assert isinstance(rc, RecedingHorizonController)
    assert isinstance(ec, TemporalEnsembleController)
