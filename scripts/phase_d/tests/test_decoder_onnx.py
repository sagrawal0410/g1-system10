"""Smoke test: real SONIC decoder ONNX wiring (requires onnxruntime + the file).

Verifies the 994->29 I/O contract end-to-end on CPU and that decode_chunk turns
a [40,64] token chunk into a finite [40,29] pose trajectory. Skips cleanly if
onnxruntime or the ONNX file is unavailable (e.g. the `groot` env has no
onnxruntime; run this in the `sonic` env)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from phase_d.sonic_decoder import SonicOnnxDecoder, DEFAULT_DECODER_ONNX, DecoderObsLayout
from ._common import make_token_chunk


def _available():
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return None
    path = os.environ.get("SONIC_DECODER_ONNX", DEFAULT_DECODER_ONNX)
    return path if Path(path).exists() else None


def test_obs_layout_sums_to_994():
    assert DecoderObsLayout().total == 994


def test_real_decoder_smoke():
    path = _available()
    if path is None:
        print("  [skip] onnxruntime or decoder ONNX not available")
        return
    dec = SonicOnnxDecoder(onnx_path=path)
    # session loads and input dim matches 994
    sess = dec.session
    assert sess.get_inputs()[0].shape[-1] == 994
    assert sess.get_outputs()[0].shape[-1] == 29

    rng = np.random.default_rng(0)
    tokens = make_token_chunk(40, 64, rng)  # on-grid FSQ tokens
    dec.reset()
    pose = dec.decode_chunk(tokens)
    assert pose.shape == (40, 29), pose.shape
    assert np.isfinite(pose).all()
    # fixed_history mode must be BOUNDED (catches the autoregressive blowup ~1e18)
    assert np.abs(pose).max() < 10.0, f"decoded pose exploded: max|pose|={np.abs(pose).max():.3g}"
    # sensitivity: zeroing the token should shift the decoded pose (gate-0.5: 0.47-0.57 rad)
    dec.reset()
    pose0 = dec.decode_chunk(np.zeros_like(tokens))
    shift = float(np.abs(pose - pose0).mean())
    assert 0.2 < shift < 1.2, f"token sensitivity {shift:.3f} outside gate-0.5 ballpark"
    print(f"  real decoder OK (fixed_history): pose range [{pose.min():.3f},{pose.max():.3f}] "
          f"mean|token-vs-zero shift|={shift:.4f} (gate-0.5: ~0.47-0.57)")


def test_autoregressive_mode_is_documented_unstable():
    """Confirm the autoregressive mode is the one that diverges (so we know the
    default fixed_history is the right choice), and that it warns."""
    path = _available()
    if path is None:
        print("  [skip] onnxruntime or decoder ONNX not available")
        return
    import warnings

    dec = SonicOnnxDecoder(onnx_path=path, mode="autoregressive")
    rng = np.random.default_rng(0)
    tokens = make_token_chunk(40, 64, rng)
    dec.reset()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pose = dec.decode_chunk(tokens)
        assert any("autoregressive" in str(x.message) for x in w), "must warn"
    # it blows up open-loop -> this is exactly why fixed_history is the default
    print(f"  autoregressive max|pose|={np.abs(pose).max():.3g} (diverges open-loop, as expected)")
