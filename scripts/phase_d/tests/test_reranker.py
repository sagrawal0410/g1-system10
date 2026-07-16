"""Unit tests: best-of-N reranking, roundtrip term, FSQ range, oracle (D2)."""
from __future__ import annotations

import numpy as np

from phase_d.reranker import (
    candidate_cost, rerank, best_of_n, oracle_best_of_k, candidate_seed,
    RerankConfig, RunningStats,
)
from phase_d.sonic_decoder import LinearMockDecoder
from ._common import synthetic_layout, FSQ_STEP

def _chunk(latent, hands=None, T=40):
    lat = np.asarray(latent, dtype=float)
    if lat.ndim == 1:
        lat = np.tile(lat, (T, 1))
    if hands is None:
        hands = np.zeros((T, 14))
    return np.concatenate([lat, hands], axis=1)

def test_rerank_picks_smoothest_candidate():
    lay = synthetic_layout()
    rng = np.random.default_rng(0)
    smooth = _chunk(np.cumsum(np.full((40, 64), 0.001), axis=0))
    jumpy1 = _chunk(rng.normal(0, 0.3, size=(40, 64)))
    jumpy2 = _chunk(rng.normal(0, 0.6, size=(40, 64)))
    res = rerank([jumpy1, smooth, jumpy2], lay)
    assert res["best_index"] == 1, (res["costs"])

def test_boundary_term_prefers_continuous_start():
    lay = synthetic_layout()
    prev_last = np.zeros(78)
    cont = _chunk(np.zeros((40, 64)))
    jump = _chunk(np.full((40, 64), 0.5))
    cfg = RerankConfig(w_velocity=0.0, w_accel=0.0, w_range=0.0, w_zscore=0.0, w_roundtrip=0.0)
    res = rerank([jump, cont], lay, cfg, prev_last=prev_last)
    assert res["best_index"] == 1

def test_candidate_seed_varies_and_is_deterministic():
    seeds = {candidate_seed(step, k, base=0, K=4) for step in range(5) for k in range(4)}
    assert len(seeds) == 20, "each (step,k) must get a distinct seed"
    assert candidate_seed(3, 2, 0, 4) == candidate_seed(3, 2, 0, 4)

def test_roundtrip_term_zero_without_decoder_positive_with():
    lay = synthetic_layout()
    rng = np.random.default_rng(1)
    cand = _chunk(rng.normal(0, 0.3, size=(40, 64)))
    _, bd_no = candidate_cost(cand, lay)
    assert bd_no["roundtrip"] == 0.0
    _, bd_yes = candidate_cost(cand, lay, decoder=LinearMockDecoder(seed=3))
    assert bd_yes["roundtrip"] > 0.0

def test_roundtrip_penalizes_jerky_pose():
    """Two candidates with matched latent-space stats but the decoder makes one
    decode to a jerkier pose -> roundtrip term separates them."""
    lay = synthetic_layout()
    dec = LinearMockDecoder(seed=5)
    smooth = _chunk(np.cumsum(np.full((40, 64), 0.0005), axis=0))
    rng = np.random.default_rng(2)
    jumpy = _chunk(rng.normal(0, 0.4, size=(40, 64)))
    cfg = RerankConfig(w_boundary=0, w_velocity=0, w_accel=0, w_range=0, w_zscore=0, w_roundtrip=1.0)
    res = rerank([jumpy, smooth], lay, cfg, decoder=dec)
    assert res["best_index"] == 1

def test_fsq_range_penalty():
    lay = synthetic_layout()
    in_range = _chunk(np.full((40, 64), 0.5))
    out_range = _chunk(np.full((40, 64), 1.2))
    _, bd_in = candidate_cost(in_range, lay)
    _, bd_out = candidate_cost(out_range, lay)
    assert bd_in["range"] == 0.0
    assert bd_out["range"] > 0.0

def test_body_vs_hand_smoothness_weighting():
    lay = synthetic_layout()
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.3, size=(40, 64))
    hand_noise = rng.normal(0, 0.3, size=(40, 14))
    jumpy_body = _chunk(noise, hands=np.zeros((40, 14)))
    jumpy_hands = _chunk(np.zeros((40, 64)), hands=hand_noise)
    cfg = RerankConfig(w_boundary=0, w_range=0, w_zscore=0, w_roundtrip=0,
                       body_smooth_scale=1.0, hand_smooth_scale=0.2)
    c_body, _ = candidate_cost(jumpy_body, lay, cfg)
    c_hands, _ = candidate_cost(jumpy_hands, lay, cfg)

    assert c_hands < c_body

def test_best_of_n_uses_seeded_predict_fn():
    lay = synthetic_layout()
    calls = {}

    def predict_fn(seed):
        calls[seed] = calls.get(seed, 0) + 1
        rng = np.random.default_rng(seed)
        return _chunk(rng.normal(0, 0.1, size=(40, 64)))

    cfg = RerankConfig(K=4)
    res = best_of_n(predict_fn, step=7, layout=lay, cfg=cfg)
    assert len(res["seeds"]) == 4 and len(set(res["seeds"])) == 4
    assert 0 <= res["best_index"] < 4
    assert res["candidate"].shape == (40, 78)

def test_oracle_picks_closest_to_gt_and_is_labeled():
    lay = synthetic_layout()
    gt = _chunk(np.full((40, 64), 0.25))
    near = _chunk(np.full((40, 64), 0.26))
    far = _chunk(np.full((40, 64), -0.5))
    res = oracle_best_of_k([far, near], gt)
    assert res["best_index"] == 1
    assert res["is_oracle"] is True
