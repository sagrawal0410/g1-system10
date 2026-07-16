"""Unit tests: layout loading + block classification."""
from __future__ import annotations

import numpy as np

from phase_d.layout import load_layout
from ._common import synthetic_layout, real_layout_path

def test_synthetic_layout_tiles_and_classifies():
    lay = synthetic_layout()
    assert lay.total_dim == 78
    assert lay.chunk_length == 40

    assert [b.name for b in lay.latent_blocks] == ["motion_token"]
    assert {b.name for b in lay.hand_blocks} == {"left_hand_joints", "right_hand_joints"}
    assert lay.latent_indices() == list(range(0, 64))
    assert lay.hand_indices() == list(range(64, 78))

    lay.validate()

def test_latent_is_not_misclassified_as_hand():
    lay = synthetic_layout()
    for b in lay.blocks:
        assert not (b.is_latent and b.is_hand)

def test_real_action_layout_loads_if_present():
    p = real_layout_path()
    if p is None:
        print("  [skip] real action_layout.json not found")
        return
    lay = load_layout(p)
    assert lay.total_dim == 78, lay.total_dim
    assert lay.chunk_length == 40
    assert lay.latent_continuous is False
    assert lay.latent_indices() == list(range(0, 64))
    assert abs(lay.fsq_step - 0.0625) < 1e-9
    assert lay.fsq_range == (-0.625, 0.625)
    print(f"  real layout OK: {[ (b.name,b.start,b.end) for b in lay.blocks ]}")
