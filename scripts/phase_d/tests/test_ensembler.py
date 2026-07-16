"""Unit tests: ACT-style TemporalEnsembler (D1)."""
from __future__ import annotations

import numpy as np

from phase_d.ensembler import TemporalEnsembler, _weighted_quat_mean

def test_single_chunk_returns_itself():
    ens = TemporalEnsembler(action_dim=3, chunk_length=5, max_timesteps=50, m=0.01)
    chunk = np.arange(5 * 3, dtype=float).reshape(5, 3)
    ens.add_chunk(0, chunk)
    for j in range(5):
        assert np.allclose(ens.get_action(j), chunk[j])

def test_weighting_exp_oldest_first_and_normalized():

    D = 1
    ens = TemporalEnsembler(action_dim=D, chunk_length=4, max_timesteps=50, m=0.5)

    ens.add_chunk(0, np.full((4, D), 10.0))
    ens.add_chunk(1, np.full((4, D), 20.0))
    ens.add_chunk(2, np.full((4, D), 30.0))

    assert ens.coverage(3) == 3
    m = 0.5
    w = np.exp(-m * np.arange(3)); w /= w.sum()
    expected = w @ np.array([10.0, 20.0, 30.0])
    assert np.allclose(ens.get_action(3), expected)

    assert w[0] > w[1] > w[2]

def test_coverage_assertion_when_no_prediction():
    ens = TemporalEnsembler(action_dim=2, chunk_length=3, max_timesteps=50)
    ens.add_chunk(0, np.ones((3, 2)))

    try:
        ens.get_action(5)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "get_action must assert when no covering prediction exists"
    assert np.all(np.isnan(ens.get_action(5, require_coverage=False)))

def test_newest_only_dims_bypass_averaging():
    D = 2
    ens = TemporalEnsembler(action_dim=D, chunk_length=3, max_timesteps=50, m=0.01,
                            newest_only_dims=[1])
    ens.add_chunk(0, np.tile([1.0, 100.0], (3, 1)))
    ens.add_chunk(1, np.tile([2.0, 200.0], (3, 1)))
    out = ens.get_action(2)

    assert out[1] == 200.0
    assert 1.0 < out[0] < 2.0

def test_all_zero_prediction_still_counts_as_covered():

    ens = TemporalEnsembler(action_dim=2, chunk_length=3, max_timesteps=50)
    ens.add_chunk(0, np.zeros((3, 2)))
    assert ens.coverage(1) == 1
    assert np.allclose(ens.get_action(1), 0.0)

def test_quat_mean_unit_and_sign_invariant():
    q = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.02, 0.9998]])
    w = np.array([0.5, 0.5])
    out = _weighted_quat_mean(q, w)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-6

    out2 = _weighted_quat_mean(np.array([q[0], -q[1]]), w)
    assert np.allclose(np.abs(out), np.abs(out2), atol=1e-6)
