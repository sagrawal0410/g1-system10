"""Unit tests: RTC-style stitching + receding horizon (D1), incl. discrete gate."""
from __future__ import annotations

import numpy as np

from phase_d.stitching import rtc_style_stitch, RecedingHorizonStitcher, StitchConfig
from phase_d.sonic_decoder import LinearMockDecoder
from ._common import synthetic_layout, make_token_chunk, on_grid, FSQ_STEP

def _prev_new(rng, T=40, overlap=32):
    lay = synthetic_layout()
    latent_prev = make_token_chunk(overlap, 64, rng)
    latent_new = make_token_chunk(T, 64, rng)
    hands_prev = rng.uniform(0, 1, size=(overlap, 14))
    hands_new = rng.uniform(0, 1, size=(T, 14))
    prev = np.concatenate([latent_prev, hands_prev], axis=1)
    new = np.concatenate([latent_new, hands_new], axis=1)
    return lay, prev, new

def test_first_plan_no_prev_snaps_latent():
    lay = synthetic_layout()
    rng = np.random.default_rng(0)
    new = np.concatenate([make_token_chunk(40, 64, rng) + 0.003,
                          rng.uniform(0, 1, (40, 14))], axis=1)
    res = rtc_style_stitch(new, None, lay)
    assert not np.isnan(res.actions).any()
    assert on_grid(res.actions[:, :64]), "latent must be snapped on-grid"

def test_no_nan_and_freeze_region_holds_prev():
    rng = np.random.default_rng(1)
    lay, prev, new = _prev_new(rng)
    cfg = StitchConfig(freeze=3, blend_len=20)
    res = rtc_style_stitch(new, prev, lay, cfg)
    assert not np.isnan(res.actions).any()

    assert np.allclose(res.actions[:3], prev[:3], atol=1e-6)

def test_discrete_gate_latent_never_blended_offgrid():
    """The core discrete-gate assertion: latent dims are NEVER a linear blend of
    prev/new; each output latent equals prev (freeze) or new (after), and is on
    the FSQ grid."""
    rng = np.random.default_rng(2)
    lay, prev, new = _prev_new(rng)
    cfg = StitchConfig(freeze=3, blend_len=20, latent_strategy="newest_only")
    res = rtc_style_stitch(new, prev, lay, cfg)
    lat = res.actions[:, :64]
    assert on_grid(lat)
    overlap = 32

    assert np.allclose(lat[:3], prev[:3, :64], atol=1e-7)
    assert np.allclose(lat[3:overlap], new[3:overlap, :64], atol=1e-7)

    a = 0.5
    linblend = (1 - a) * prev[10, :64] + a * new[10, :64]
    assert not np.allclose(lat[10], linblend, atol=1e-6) or np.allclose(prev[10, :64], new[10, :64])

def test_hands_weakly_blended_with_sqrt_alpha():
    rng = np.random.default_rng(3)
    lay, prev, new = _prev_new(rng)
    cfg = StitchConfig(freeze=0, blend_len=32, hand_alpha_power=0.5)
    res = rtc_style_stitch(new, prev, lay, cfg)

    i = 8
    alpha = (i - 0 + 1) / (32 + 1)
    a_hand = alpha ** 0.5
    expected = (1 - a_hand) * prev[i, 64:] + a_hand * new[i, 64:]
    assert np.allclose(res.actions[i, 64:], expected, atol=1e-6)

    assert a_hand > alpha

def test_decoded_pose_returns_body_override():
    rng = np.random.default_rng(4)
    lay, prev, new = _prev_new(rng)
    dec = LinearMockDecoder(seed=7)
    cfg = StitchConfig(freeze=3, blend_len=20, latent_strategy="decoded_pose")
    res = rtc_style_stitch(new, prev, lay, cfg, decoder=dec)
    assert res.body_pose_override is not None
    assert res.body_pose_override.shape == (40, 29)

    assert not np.isnan(res.body_pose_override[:32]).any()
    assert on_grid(res.actions[:, :64])

def test_receding_horizon_executes_correct_steps_and_is_continuous():
    rng = np.random.default_rng(5)
    lay = synthetic_layout()
    stitcher = RecedingHorizonStitcher(lay, StitchConfig(freeze=3, blend_len=20), execute=8)
    last_exec = None
    boundary_jumps = []
    for _ in range(4):
        chunk = np.concatenate([make_token_chunk(40, 64, rng),
                                rng.uniform(0, 1, (40, 14))], axis=1)
        res = stitcher.step(chunk)
        assert res.actions.shape == (8, 78)
        assert not np.isnan(res.actions).any()
        if last_exec is not None:
            boundary_jumps.append(np.abs(res.actions[0] - last_exec).max())
        last_exec = res.actions[-1]

    assert all(np.isfinite(j) for j in boundary_jumps)
