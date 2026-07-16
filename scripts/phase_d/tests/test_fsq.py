"""Unit tests: FSQ manifold projection (D2)."""
from __future__ import annotations

import numpy as np

from phase_d.fsq import fsq_project_values, fsq_project_action, fsq_projection_report
from ._common import synthetic_layout, on_grid, FSQ_STEP, FSQ_CLAMP


def test_projection_lands_exactly_on_grid():
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.7, 0.7, size=(40, 64))
    q = fsq_project_values(x)
    assert on_grid(q), "projected values must be exact multiples of 1/16"
    # every value within [-0.625, 0.625]
    assert q.min() >= FSQ_CLAMP[0] - 1e-9 and q.max() <= FSQ_CLAMP[1] + 1e-9


def test_projection_is_idempotent():
    rng = np.random.default_rng(1)
    x = rng.uniform(-0.7, 0.7, size=(10, 64))
    q1 = fsq_project_values(x)
    q2 = fsq_project_values(q1)
    assert np.array_equal(q1, q2)


def test_on_grid_input_unchanged():
    grid_vals = np.array([-0.625, -0.0625, 0.0, 0.0625, 0.625] * 12, dtype=np.float64)[:60]
    q = fsq_project_values(grid_vals)
    assert np.allclose(q, grid_vals, atol=1e-7)


def test_clamp_pushes_out_of_range():
    x = np.array([1.0, -1.0, 0.9, -0.9])
    q = fsq_project_values(x)
    assert np.all(q <= 0.625 + 1e-9) and np.all(q >= -0.625 - 1e-9)
    assert q[0] == 0.625 and q[1] == -0.625


def test_only_latent_dims_projected():
    lay = synthetic_layout()
    rng = np.random.default_rng(2)
    act = rng.uniform(-0.7, 0.7, size=(40, 78))
    out = fsq_project_action(act, lay.latent_indices())
    # latent dims on grid, hand dims untouched
    assert on_grid(out[:, :64])
    assert np.array_equal(out[:, 64:], act[:, 64:].astype(np.float32))


def test_report_removes_offgrid_jitter():
    rng = np.random.default_rng(3)
    onnx_like = np.round(rng.uniform(-0.5, 0.5, size=(200,)) / FSQ_STEP) * FSQ_STEP
    jitter = onnx_like + rng.normal(0, 0.01, size=onnx_like.shape)  # push off grid
    rep = fsq_projection_report(jitter)
    assert rep.frac_offgrid_before > 0.9   # nearly all off-grid before
    assert rep.frac_offgrid_after < 1e-6   # all on-grid after
    assert rep.mean_abs_shift > 0.0
    print(f"  fsq report: offgrid {rep.frac_offgrid_before:.2f}->{rep.frac_offgrid_after:.2f} "
          f"mean_shift={rep.mean_abs_shift:.4f}")
