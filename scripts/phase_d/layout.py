#!/usr/bin/env python3
"""
Phase D — action-layout helper.

Single source of truth for *which action dimensions are what*. Everything in
Phase D (stitching / ensembling / reranking / FSQ projection) reads block
indices from ``results/action_layout.json`` through this module and NEVER
hard-codes 0:64 / 64:78. If the layout file changes (e.g. hands become 12-dim
dexterous or 2-dim gripper, or hands are zeroed for a body-only run), Phase D
follows automatically.

Schema handled (the real Phase-0.5 schema):
    {
      "action": {
        "total_dim": 78,
        "chunk_length": 40,
        "block_layout": [
          {"name": "motion_token", "indices": [0, 64], "continuous": false, ...},
          {"name": "left_hand_joints", "indices": [64, 71], "continuous": true, ...},
          {"name": "right_hand_joints", "indices": [71, 78], "continuous": true, ...}
        ]
      },
      "latent_continuous": false,
      "quantization": {"empirical_grid_step": 0.0625, "observed_value_range": [-0.625, 0.625], ...}
    }

Block classification is by name-substring + the per-block ``continuous`` flag
(mirrors scripts/eval_openloop.py's is_latent_block / is_hand_block) so we
attach to the right blocks regardless of exact naming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional


# Fallback ONLY if the file is unreadable. Loudly flagged by the loader.
_FALLBACK_BLOCKS = [
    ("motion_token", (0, 64), False),
    ("left_hand_joints", (64, 71), True),
    ("right_hand_joints", (71, 78), True),
]
_FALLBACK_TOTAL = 78
_FALLBACK_CHUNK = 40

# FSQ grid (from action_layout.json quantization block). Defaults used only if
# the file omits them.
_DEFAULT_FSQ_STEP = 0.0625  # 1/16
_DEFAULT_FSQ_RANGE = (-0.625, 0.625)  # observed alphabet (21 levels); FSQ theory is 32 levels ~[-1,1)


def is_latent_block(name: str, continuous: Optional[bool] = None) -> bool:
    n = name.lower()
    if any(k in n for k in ("latent", "token", "motion")) and "hand" not in n:
        return True
    # discrete non-hand block -> treat as latent even if oddly named
    if continuous is False and not is_hand_block(name):
        return True
    return False


def is_hand_block(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("hand", "gripper", "finger"))


def hand_side(name: str) -> str:
    n = name.lower()
    if "left" in n or n.startswith("l_") or "_l_" in n:
        return "left"
    if "right" in n or n.startswith("r_") or "_r_" in n:
        return "right"
    return "unknown"


@dataclass
class Block:
    name: str
    start: int
    end: int  # exclusive
    continuous: Optional[bool] = None

    @property
    def slice(self) -> slice:
        return slice(self.start, self.end)

    @property
    def dim(self) -> int:
        return self.end - self.start

    @property
    def is_latent(self) -> bool:
        return is_latent_block(self.name, self.continuous)

    @property
    def is_hand(self) -> bool:
        return is_hand_block(self.name)


@dataclass
class ActionLayout:
    total_dim: int
    chunk_length: int
    blocks: list[Block]
    latent_continuous: Optional[bool] = None
    fsq_step: float = _DEFAULT_FSQ_STEP
    fsq_range: tuple[float, float] = _DEFAULT_FSQ_RANGE
    fsq_num_levels: Optional[int] = None
    source: str = "action_layout.json"

    # ---- convenience selectors --------------------------------------------
    @property
    def latent_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.is_latent]

    @property
    def hand_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.is_hand]

    def latent_indices(self) -> list[int]:
        idx: list[int] = []
        for b in self.latent_blocks:
            idx.extend(range(b.start, b.end))
        return idx

    def hand_indices(self) -> list[int]:
        idx: list[int] = []
        for b in self.hand_blocks:
            idx.extend(range(b.start, b.end))
        return idx

    def block_by_name(self, name: str) -> Block:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(name)

    def validate(self) -> None:
        covered = sum(b.dim for b in self.blocks)
        if covered != self.total_dim:
            raise ValueError(
                f"[action_layout MISMATCH] blocks cover {covered} dims but "
                f"total_dim={self.total_dim}. Fix action_layout.json; refusing to guess."
            )
        # blocks must tile [0,total_dim) without gaps/overlaps
        ordered = sorted(self.blocks, key=lambda b: b.start)
        cursor = 0
        for b in ordered:
            if b.start != cursor:
                raise ValueError(
                    f"[action_layout] block '{b.name}' starts at {b.start}, expected {cursor} "
                    f"(gap/overlap in block tiling)."
                )
            cursor = b.end
        if cursor != self.total_dim:
            raise ValueError(
                f"[action_layout] blocks end at {cursor}, expected total_dim={self.total_dim}."
            )


def _parse_block_list(items: list) -> list[Block]:
    out: list[Block] = []
    for item in items:
        name = item.get("name") or item.get("modality_key")
        idx = item.get("indices") or item.get("slice")
        if isinstance(idx, (list, tuple)) and len(idx) == 2:
            s, e = idx
        elif "start" in item and "end" in item:
            s, e = item["start"], item["end"]
        else:
            raise ValueError(f"Unrecognized block record (no indices/start-end): {item}")
        out.append(Block(name=name, start=int(s), end=int(e), continuous=item.get("continuous")))
    return out


def load_layout(path: str | Path) -> ActionLayout:
    """Load and validate the action layout. Raises loudly on an unrecognized
    schema rather than silently guessing."""
    path = Path(path)
    if not path.exists():
        import warnings

        warnings.warn(
            f"action_layout.json not found at {path}; using hand-verified fallback "
            f"(motion_token=0:64, hands=64:78, chunk=40). DO NOT trust results off this "
            f"fallback — regenerate the real file.",
            stacklevel=2,
        )
        layout = ActionLayout(
            total_dim=_FALLBACK_TOTAL,
            chunk_length=_FALLBACK_CHUNK,
            blocks=[Block(n, s, e, c) for n, (s, e), c in _FALLBACK_BLOCKS],
            latent_continuous=False,
            source="fallback",
        )
        layout.validate()
        return layout

    with open(path) as f:
        d = json.load(f)

    action = d.get("action", {})
    block_list = action.get("block_layout") or action.get("blocks") or d.get("block_layout")
    if not isinstance(block_list, list) or not block_list:
        raise ValueError(
            f"Could not find action.block_layout (list) in {path}. Top-level keys: {sorted(d.keys())}."
        )
    blocks = _parse_block_list(block_list)

    total_dim = action.get("total_dim") or d.get("total_dim") or sum(b.dim for b in blocks)
    chunk_length = action.get("chunk_length") or d.get("chunk_length") or _FALLBACK_CHUNK

    q = d.get("quantization", {})
    fsq_step = float(q.get("empirical_grid_step", _DEFAULT_FSQ_STEP))
    rng = q.get("observed_value_range", list(_DEFAULT_FSQ_RANGE))
    fsq_range = (float(rng[0]), float(rng[1]))
    fsq_num_levels = q.get("num_fsq_levels")

    layout = ActionLayout(
        total_dim=int(total_dim),
        chunk_length=int(chunk_length),
        blocks=blocks,
        latent_continuous=d.get("latent_continuous"),
        fsq_step=fsq_step,
        fsq_range=fsq_range,
        fsq_num_levels=int(fsq_num_levels) if fsq_num_levels is not None else None,
        source=str(path),
    )
    layout.validate()
    return layout
