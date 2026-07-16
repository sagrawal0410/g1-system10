#!/usr/bin/env python3
"""
Phase D — SONIC decoder wrapper (System-0-native pose decode).

Used by two Phase-D features:
  * D1 decoded-pose blending: blend overlapping token chunks in *decoded pose
    space* instead of raw FSQ-latent space (FSQ dims must not be linearly
    interpolated -> off-grid garbage).
  * D2 roundtrip reranking term: decode each candidate token chunk and score
    pose-space discontinuity / infeasibility.

VERIFIED ONNX I/O (this agent, 2026-07-16, against the released
model_decoder.onnx on lambdafs, matches the Phase-0.5 gate finding):
    input  "obs_dict"  : [1, 994]
    output "action"    : [1, 29]      (29 body-joint targets: legs12 + waist3 + arms14)

994-dim obs layout (Phase-0.5 gate 0.5 `decoder_onnx.layout`, sums to 994):
    token(64) | gravity_10f(30) | base_ang_vel_10f(30)
              | joint_pos_rel_10f(290) | joint_vel_10f(290) | last_actions_10f(290)
    (10 history frames; 29 body joints -> 29*10 = 290; 3-vecs -> 3*10 = 30)

NOTE the release `observation_config.yaml` advertises a *different* 436-dim obs
(64+12+116+116+116+12); that config does NOT match the released decoder ONNX
(994). We follow the 994 layout, which is what the ONNX actually accepts and
what the gate-0.5 roundtrip validated.

IMPORTANT / FLAGGED FOR PHASE C (deferred verification):
The SONIC decoder is a *closed-loop 50 Hz tracker*: token + live proprio history
-> next body action, iterated inside a MuJoCo/WBC control loop. There is no
faithful one-shot open-loop token->pose map.

Two decode modes:
  * mode="fixed_history" (DEFAULT): hold the proprio history at a fixed reference
    (default pose / zero velocity / zero last-action / provided gravity) and
    sweep only the token across the chunk. This is STABLE and bounded, and its
    per-step output reproduces the gate-0.5 token sensitivity (~0.5 rad;
    measured here: meanabs 0.61 rad, token-vs-zero shift 0.55 rad -- matches
    gate05_summary token_sensitivity_rad_mean 0.47-0.57). A candidate token
    chunk therefore decodes to a pose trajectory whose *discontinuity* tracks
    the token chunk's discontinuity through the REAL decoder -- exactly what the
    D2 roundtrip score and D1 decoded-pose blend need for RELATIVE comparison.
  * mode="autoregressive": feed the decoder's own output back as the next proprio
    frame. VERIFIED to DIVERGE open-loop (no physics): outputs blow up to ~1e18
    within a chunk, because the closed loop is only stable inside the MuJoCo/WBC
    control loop. Kept for the day the real closed-loop stack is wired (Phase C);
    NOT usable for scoring standalone. A loud warning fires if selected.

In BOTH modes the ABSOLUTE pose values and the exact obs conventions (frame
order, IsaacLab joint permutation, default-pose offset, gravity sign) still need
validation against the C++ deploy stack before absolute decoded poses are
trusted; the fixed-history score is used only for RELATIVE candidate ranking and
pose-continuity, which is robust to those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_DECODER_ONNX = (
    "/lambdafs/shaurya/g1_sonic_system1/repos/GR00T-WholeBodyControl/"
    "gear_sonic_deploy/policy/release/model_decoder.onnx"
)

TOKEN_DIM = 64
BODY_DIM = 29
HISTORY = 10

@dataclass
class DecoderObsLayout:
    """Segment sizes of the 994-dim decoder obs, in order."""
    token: int = TOKEN_DIM
    gravity_10f: int = 3 * HISTORY
    base_ang_vel_10f: int = 3 * HISTORY
    joint_pos_rel_10f: int = BODY_DIM * HISTORY
    joint_vel_10f: int = BODY_DIM * HISTORY
    last_actions_10f: int = BODY_DIM * HISTORY

    @property
    def total(self) -> int:
        return (
            self.token
            + self.gravity_10f
            + self.base_ang_vel_10f
            + self.joint_pos_rel_10f
            + self.joint_vel_10f
            + self.last_actions_10f
        )

class PoseDecoder:
    """Interface: turn a token chunk ``[T, token_dim]`` into a body-joint pose
    trajectory ``[T, body_dim]``. Implementations may be stateful (closed-loop);
    call ``reset()`` before decoding an independent chunk."""

    token_dim: int = TOKEN_DIM
    body_dim: int = BODY_DIM

    def reset(self) -> None:
        pass

    def decode_chunk(self, token_chunk: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class LinearMockDecoder(PoseDecoder):
    """Deterministic, dependency-free stand-in for unit tests / no-onnxruntime
    environments. pose[t] = tanh(W @ token[t] + b) * scale. Because it is a
    (smooth) fixed map, a smooth token trajectory decodes to a smooth pose
    trajectory and a jumpy token trajectory to a jumpy pose trajectory -- which
    is exactly the property D1/D2 rely on -- while needing no ONNX/physics."""

    def __init__(self, token_dim: int = TOKEN_DIM, body_dim: int = BODY_DIM, seed: int = 0,
                 scale: float = 1.0):
        self.token_dim = token_dim
        self.body_dim = body_dim
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 1.0 / np.sqrt(token_dim), size=(body_dim, token_dim)).astype(np.float64)
        self.b = rng.normal(0.0, 0.05, size=(body_dim,)).astype(np.float64)
        self.scale = scale

    def decode_chunk(self, token_chunk: np.ndarray) -> np.ndarray:
        x = np.asarray(token_chunk, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        pose = np.tanh(x @ self.W.T + self.b) * self.scale
        return pose.astype(np.float32)

class SonicOnnxDecoder(PoseDecoder):
    """Real released SONIC decoder ONNX, driven with an autoregressive proprio
    proxy (see module docstring for the closed-loop caveat). onnxruntime is
    imported lazily so importing Phase D never hard-requires it (the `groot`
    env, used by eval_openloop, has onnx but not onnxruntime; the `sonic` env
    has both)."""

    def __init__(
        self,
        onnx_path: str | Path = DEFAULT_DECODER_ONNX,
        providers: Optional[list[str]] = None,
        history: int = HISTORY,
        default_pose: Optional[np.ndarray] = None,
        gravity_dir: tuple[float, float, float] = (0.0, 0.0, -1.0),
        base_ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
        obs_layout: Optional[DecoderObsLayout] = None,
        joint_permutation: Optional[np.ndarray] = None,
        dt: float = 1.0 / 50.0,
        mode: str = "fixed_history",
    ):
        if mode not in ("fixed_history", "autoregressive"):
            raise ValueError(f"mode must be 'fixed_history' or 'autoregressive', got {mode!r}")
        self.mode = mode
        self.onnx_path = str(onnx_path)
        self.providers = providers or ["CPUExecutionProvider"]
        self.history = history
        self.layout = obs_layout or DecoderObsLayout()
        self.body_dim = self.layout.joint_pos_rel_10f // history
        self.token_dim = self.layout.token
        self.default_pose = (
            np.zeros(self.body_dim) if default_pose is None else np.asarray(default_pose, dtype=np.float64)
        )
        self.gravity_dir = np.asarray(gravity_dir, dtype=np.float64)
        self.base_ang_vel = np.asarray(base_ang_vel, dtype=np.float64)
        self.joint_permutation = joint_permutation
        self.dt = dt
        self._sess = None
        self.reset()

    @property
    def session(self):
        if self._sess is None:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            self._sess = ort.InferenceSession(self.onnx_path, sess_options=so, providers=self.providers)
            self._in_name = self._sess.get_inputs()[0].name
            self._out_name = self._sess.get_outputs()[0].name
            in_shape = self._sess.get_inputs()[0].shape
            expected = self.layout.total
            if in_shape[-1] not in (expected, "obs_dim", None):
                raise ValueError(
                    f"Decoder ONNX input dim {in_shape[-1]} != expected {expected} "
                    f"(obs layout mismatch). Re-verify decoder_onnx layout."
                )
        return self._sess

    def reset(self) -> None:

        h, b = self.history, self.body_dim
        self._pos_rel_hist = np.zeros((h, b))
        self._vel_hist = np.zeros((h, b))
        self._act_hist = np.zeros((h, b))
        self._last_pos = self.default_pose.copy()

    def _assemble_obs(self, token: np.ndarray) -> np.ndarray:
        parts = [
            np.asarray(token, dtype=np.float64).ravel(),
            np.tile(self.gravity_dir, self.history),
            np.tile(self.base_ang_vel, self.history),
            self._pos_rel_hist.reshape(-1),
            self._vel_hist.reshape(-1),
            self._act_hist.reshape(-1),
        ]
        obs = np.concatenate(parts).astype(np.float32)
        assert obs.shape[0] == self.layout.total, (obs.shape[0], self.layout.total)
        return obs[None, :]

    def _push(self, buf: np.ndarray, frame: np.ndarray) -> np.ndarray:
        buf = np.roll(buf, -1, axis=0)
        buf[-1] = frame
        return buf

    def decode_chunk(self, token_chunk: np.ndarray) -> np.ndarray:
        """Decode a token chunk -> [T, body_dim] pose.

        fixed_history (default): proprio held at the reference for every step;
        only the token varies -> stable, bounded, reproduces gate-0.5 token
        sensitivity. autoregressive: feed output back (diverges open-loop; warned)."""
        tokens = np.asarray(token_chunk, dtype=np.float64)
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        sess = self.session
        out = np.empty((tokens.shape[0], self.body_dim), dtype=np.float32)

        if self.mode == "autoregressive":
            import warnings

            warnings.warn(
                "SonicOnnxDecoder mode='autoregressive' diverges open-loop (no physics); "
                "it is only stable inside the real closed-loop WBC stack. Use 'fixed_history' "
                "for standalone scoring/blending.",
                stacklevel=2,
            )

        for t in range(tokens.shape[0]):
            obs = self._assemble_obs(tokens[t])
            action = sess.run([self._out_name], {self._in_name: obs})[0].reshape(-1)
            if self.joint_permutation is not None:
                action = action[self.joint_permutation]
            out[t] = action
            if self.mode == "autoregressive":

                pos_rel = action - self.default_pose
                vel = (action - self._last_pos) / self.dt
                self._pos_rel_hist = self._push(self._pos_rel_hist, pos_rel)
                self._vel_hist = self._push(self._vel_hist, vel)
                self._act_hist = self._push(self._act_hist, action)
                self._last_pos = action

        return out

    def set_reference(
        self,
        joint_pos: Optional[np.ndarray] = None,
        joint_vel: Optional[np.ndarray] = None,
        last_action: Optional[np.ndarray] = None,
        gravity_dir: Optional[np.ndarray] = None,
        base_ang_vel: Optional[np.ndarray] = None,
    ) -> None:
        """Contextualize the fixed-history reference to the robot's current
        measured proprio (e.g. from the live obs). All args optional; each fills
        its 10-frame history with the given single frame. Defaults (reset) are
        default-pose / zero-velocity / zero-action."""
        if joint_pos is not None:
            self._pos_rel_hist[:] = (np.asarray(joint_pos, float) - self.default_pose)[None, :]
        if joint_vel is not None:
            self._vel_hist[:] = np.asarray(joint_vel, float)[None, :]
        if last_action is not None:
            self._act_hist[:] = np.asarray(last_action, float)[None, :]
        if gravity_dir is not None:
            self.gravity_dir = np.asarray(gravity_dir, float)
        if base_ang_vel is not None:
            self.base_ang_vel = np.asarray(base_ang_vel, float)

def make_decoder(kind: str = "onnx", **kwargs) -> PoseDecoder:
    """Factory. kind='onnx' -> SonicOnnxDecoder; kind='mock' -> LinearMockDecoder."""
    if kind == "onnx":
        return SonicOnnxDecoder(**kwargs)
    if kind == "mock":
        return LinearMockDecoder(**kwargs)
    raise ValueError(f"unknown decoder kind {kind!r}")
