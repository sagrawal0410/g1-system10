#!/usr/bin/env python3
"""
Phase C2 -- System-0 / System-1 hierarchy runner.

    held-out episode camera obs (DATASET frames, open-loop, PRINTED not
    sim-rendered)
        -> System 1 = GR00T N1.7 served via Isaac-GR00T PolicyServer (ZMQ)
        -> motion_token(40x64) + left/right hand(40x7) action chunks
        -> System 0 = SONIC universal-token decoder (closed-loop, 994->29 body)
        -> sim stepping at the repo's loop rates

This reproduces the *exact* serving + scheduling of the repo's own VLA inference
path (`gear_sonic/scripts/run_vla_inference.py` + `utils/inference/vla_utils.py`):
  - PolicyServer / PolicyClient (ZMQ REQ/REP), `gr00t/eval/run_gr00t_server.py`
  - inference rate 2.5 Hz (1/0.4 s), action_publish_rate 50 Hz, horizon 40
  - latency-compensated chunk indexing (calculate_latency_compensated_index)
  - should_trigger_new_inference gating with an async single-slot chunk cache
The three scheduling helpers are re-implemented here IDENTICALLY to vla_utils
(that module lives in the `sonic`/deploy env, not this client env) and unit-checked
against the docstring semantics -- they are not re-invented behaviour.

------------------------------------------------------------------------------
SIM BACKENDS (`--sim-backend`)
------------------------------------------------------------------------------
  mujoco   [RUNS ON THIS CLUSTER -- wiring / smoke]
      Closed-loop rollout of the SONIC decoder ONNX under a PERFECT-TRACKING
      assumption (achieved body pose == commanded target each 50 Hz step). This
      makes the decoder's proprio feedback self-consistent WITHOUT needing the
      IsaacLab joint permutation or default-offset table:
          joint_pos_rel[t] == last_actions[t] == decoder_output[t]
          joint_vel[t]     == (a[t]-a[t-1]) * control_freq
      It exercises the whole hierarchy (obs -> server -> chunks -> decoder ->
      50 Hz control loop) and yields a decoded body(29)+hand(14) trajectory, but
      it has NO contact physics: no falls, no ground reaction, no drift. Do not
      read stability/roll-pitch from this backend.

  isaaclab [RUNS ON THE RTX 5090 -- faithful contact physics]
      The only sanctioned token->physics closed loop (SONIC's own IsaacLab env
      assembles the decoder obs in the trained order and applies real PD +
      contacts). Delegates to `eval_agent_trl.py` / token-injection per
      `results/isaacsim_roundtrip_runbook.md`. This mode intentionally REFUSES
      to run on the cluster (no Isaac Sim) and points you at that runbook.

NOTE on the WBC MuJoCo caveat: the repo's stock `run_sim_loop.py` runs the
*decoupled* WBC (arm interpolation + RL legs) over unitree_sdk2py DDS -- it does
NOT consume the SONIC universal token. This script does not use it; the `mujoco`
backend here is a direct decoder-in-the-loop rollout, which is the token path.
"""
import argparse
import collections
import functools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HOME = os.path.expanduser("~")
PROJ = f"{HOME}/g1_sonic_system1"
DEF_DATASET = "/lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic"
DEF_DECODER = ("/lambdafs/shaurya/g1_sonic_system1/repos/GR00T-WholeBodyControl/"
               "gear_sonic_deploy/policy/release/model_decoder.onnx")
DEF_ISAAC_REPO = f"{PROJ}/repos/Isaac-GR00T"
DEF_SERVER_PY = f"{HOME}/miniconda3/envs/groot/bin/python"

TOKEN_DIM = 64
N_BODY = 29
N_HIST = 10
STATE_GROUPS = [
    ("left_leg", 0, 6), ("right_leg", 6, 12), ("waist", 12, 15),
    ("left_arm", 15, 22), ("right_arm", 22, 29), ("left_hand", 29, 36),
    ("right_hand", 36, 43), ("projected_gravity", 43, 46),
]

def green(x):
    print(f"\033[92m{x}\033[0m", flush=True)

def calculate_latency_compensated_index(inference_delay, control_freq, action_horizon):
    raw_index = np.round(inference_delay * control_freq)
    return int(np.clip(raw_index, 0, action_horizon - 1))

def should_trigger_new_inference(cached_chunk_exists, inference_thread_running,
                                 time_since_last_inference, inference_interval):
    if not cached_chunk_exists:
        return True
    if inference_thread_running:
        return False
    return time_since_last_inference >= inference_interval

class MiniPolicyClient:
    def __init__(self, host="localhost", port=5555, timeout_ms=60000):
        import msgpack_numpy as mnp
        import zmq
        self._mnp = mnp
        self._zmq = zmq
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.context = zmq.Context()
        self._init_socket()

    def _init_socket(self):
        self.socket = self.context.socket(self._zmq.REQ)
        self.socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def _to_bytes(self, data):
        import msgpack
        return msgpack.packb(data, default=functools.partial(self._mnp.encode, chain=None))

    def _from_bytes(self, data):
        import msgpack
        return msgpack.unpackb(data, object_hook=functools.partial(self._mnp.decode, chain=None),
                               raw=False)

    def _call(self, endpoint, data=None, requires_input=True):
        req = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        try:
            self.socket.send(self._to_bytes(req))
            msg = self.socket.recv()
        except self._zmq.error.Again:
            self._init_socket()
            raise
        resp = self._from_bytes(msg)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp

    def ping(self):
        try:
            self._call("ping", requires_input=False)
            return True
        except Exception:
            self._init_socket()
            return False

    def get_action(self, observation, options=None):
        resp = self._call("get_action", {"observation": observation, "options": options})
        return tuple(resp)

    def reset(self, options=None):
        return self._call("reset", {"options": options})

    def kill_server(self):
        try:
            self._call("kill", requires_input=False)
        except Exception:
            pass

    def close(self):
        try:
            self.socket.close(linger=0)
            self.context.term()
        except Exception:
            pass

def load_episode(dataset_path, episode_index):
    import pyarrow.parquet as pq
    import glob
    cand = glob.glob(f"{dataset_path}/data/**/episode_{episode_index:06d}.parquet", recursive=True)
    if not cand:
        raise FileNotFoundError(f"episode {episode_index} parquet not found under {dataset_path}")
    tbl = pq.read_table(cand[0])
    state = np.asarray(tbl.column("observation.state").to_pylist(), dtype=np.float32)
    action = np.asarray(tbl.column("action").to_pylist(), dtype=np.float32)

    prompt = "demo"
    ep_meta = Path(dataset_path) / "meta" / "episodes.jsonl"
    if ep_meta.exists():
        for line in ep_meta.read_text().splitlines():
            d = json.loads(line)
            if d.get("episode_index") == episode_index and d.get("tasks"):
                prompt = d["tasks"][0]
                break
    return state, action, prompt, cand[0]

def load_head_cam_frame(dataset_path, episode_index, frame_index, cache={}):
    """Decode one head_cam frame (open-loop dataset camera). Cached per episode."""
    import glob
    key = (dataset_path, episode_index)
    if key not in cache:
        vids = glob.glob(f"{dataset_path}/videos/**/observation.images.head_cam/"
                         f"**/episode_{episode_index:06d}.mp4", recursive=True)
        cache[key] = vids[0] if vids else None
    path = cache[key]
    if path is None:
        return np.zeros((480, 640, 3), np.uint8)
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return np.zeros((480, 640, 3), np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

def build_observation(state_row, head_img, prompt):
    """(B=1,T=1) observation dict matching the unitree_g1_sonic modality config."""
    obs = {"video": {}, "state": {}, "language": {}}
    obs["video"]["ego_view"] = head_img[None, None].astype(np.uint8)
    for name, s, e in STATE_GROUPS:
        obs["state"][name] = state_row[s:e][None, None].astype(np.float32)
    obs["language"]["annotation.human.task_description"] = [[prompt]]
    return obs

class DecoderClosedLoop:
    """Closed-loop rollout of model_decoder.onnx (994->29) with a simple
    first-order actuator-lag tracking model (NO contact physics).

    The decoder was trained expecting MEASURED proprio (which lags the commanded
    target through the PD loop) plus the previous COMMANDED action. We emulate
    that with one bounded, in-distribution actuator model:
        measured_rel[t] = measured_rel[t-1] + gain * (action[t-1] - measured_rel[t-1])
        joint_pos_rel   = measured_rel      (lagged, != command)
        last_actions    = action            (commanded)
        joint_vel       = d(measured_rel) * control_freq
    gain=1.0 recovers the perfect-tracking identity. This keeps the loop bounded
    and closer to the training distribution than perfect-track; it is still a
    wiring smoke, not physics. The faithful PD+contact loop is the 5090/IsaacLab
    backend.
    """

    def __init__(self, onnx_path, control_freq=50.0, track_gain=0.3, clip=6.2832):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        indim = self.sess.get_inputs()[0].shape[-1]
        assert indim == 994, f"decoder input dim {indim} != 994"
        self.control_freq = control_freq
        self.gain = float(track_gain)
        self.clip = float(clip)
        self.reset()

    def reset(self):
        self.hist_pos = collections.deque([np.zeros(N_BODY, np.float32)] * N_HIST, maxlen=N_HIST)
        self.hist_vel = collections.deque([np.zeros(N_BODY, np.float32)] * N_HIST, maxlen=N_HIST)
        self.hist_act = collections.deque([np.zeros(N_BODY, np.float32)] * N_HIST, maxlen=N_HIST)
        self.measured = np.zeros(N_BODY, np.float32)
        self.grav = np.array([0, 0, -1], np.float32)
        self.max_abs = 0.0
        self.clipped_steps = 0

    def step(self, token):

        ba = np.zeros(3 * N_HIST, np.float32)
        bp = np.concatenate(list(self.hist_pos))
        bv = np.concatenate(list(self.hist_vel))
        la = np.concatenate(list(self.hist_act))
        gd = np.tile(self.grav, N_HIST)
        obs = np.concatenate([token.astype(np.float32), ba, bp, bv, la, gd])[None]
        raw = self.sess.run(None, {self.iname: obs.astype(np.float32)})[0].ravel().astype(np.float32)
        self.max_abs = max(self.max_abs, float(np.abs(raw).max()))
        act = np.clip(raw, -self.clip, self.clip).astype(np.float32)
        if np.any(np.abs(raw) > self.clip):
            self.clipped_steps += 1

        prev_meas = self.measured.copy()
        cmd_prev = self.hist_act[-1]
        self.measured = np.clip(prev_meas + self.gain * (cmd_prev - prev_meas),
                                -self.clip, self.clip).astype(np.float32)
        vel = (self.measured - prev_meas) * self.control_freq
        self.hist_pos.append(self.measured.copy())
        self.hist_vel.append(vel.astype(np.float32))
        self.hist_act.append(act.copy())
        return act

def start_server(args, port):
    cmd = [args.server_python, "gr00t/eval/run_gr00t_server.py",
           "--embodiment-tag", args.embodiment_tag,
           "--host", "127.0.0.1", "--port", str(port)]
    if args.model_path:
        cmd += ["--model-path", args.model_path, "--device", args.device]
        kind = f"Gr00tPolicy({args.model_path})"
    else:
        cmd += ["--dataset-path", args.dataset_path,
                "--execution-horizon", str(args.action_horizon)]
        kind = f"ReplayPolicy({args.dataset_path})  [GT-token replay -- wiring smoke]"
    green(f"[server] launching {kind}")
    print("[server] cmd:", " ".join(cmd), flush=True)
    logf = open(args.out_dir + "/server.log", "w")
    proc = subprocess.Popen(cmd, cwd=args.isaac_gr00t_repo, stdout=logf, stderr=subprocess.STDOUT)
    return proc, logf

def run_mujoco(args):
    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 78)
    green("PHASE C2 hierarchy runner -- backend=mujoco (CLUSTER SMOKE, kinematic)")
    print("Camera observations are DATASET frames (open-loop). They are NOT "
          "rendered from a running sim.")
    print("This backend has NO contact physics (first-order actuator-lag decoder")
    print("loop). It proves the hierarchy WIRING + timing; it is expected to drift/")
    print("diverge over a long horizon -- that is exactly why the faithful token->")
    print("physics rollout belongs on the 5090 (--sim-backend isaaclab).")
    print("=" * 78)

    state, gt_action, prompt, pq_path = load_episode(args.dataset_path, args.episode_index)
    T = len(state)
    print(f"[data] episode {args.episode_index}: {T} frames, parquet={pq_path}")
    print(f"[data] prompt: {prompt[:90]}{'...' if len(prompt) > 90 else ''}")

    proc = logf = None
    port = args.policy_port
    if args.autostart_server:
        proc, logf = start_server(args, port)

    client = MiniPolicyClient(host=args.policy_host, port=port, timeout_ms=args.timeout_ms)
    green(f"[client] connecting to PolicyServer {args.policy_host}:{port} ...")
    t0 = time.time()
    while not client.ping():
        if time.time() - t0 > args.server_wait_s:
            if proc:
                proc.terminate()
            raise RuntimeError(f"PolicyServer not reachable after {args.server_wait_s}s "
                               f"(see {args.out_dir}/server.log)")
        time.sleep(1.0)
    green("[client] PolicyServer reachable.")

    decoder = DecoderClosedLoop(args.decoder_onnx, control_freq=args.action_publish_rate,
                                track_gain=args.track_gain)

    inference_interval = 1.0 / args.rate
    loop_period = 1.0 / args.action_publish_rate
    max_steps = args.max_steps if args.max_steps > 0 else min(T, 600)

    ex_body, ex_hands, used_tokens, gt_tokens_at_step = [], [], [], []
    step_time, chunk_marks, latencies = [], [], []

    cached_chunk = None
    chunk_index = 0
    last_inf_time = 0.0
    n_inferences = 0
    sim_frame = 0

    green(f"[loop] control @ {args.action_publish_rate} Hz, inference @ {args.rate} Hz, "
          f"horizon {args.action_horizon}, steps {max_steps} "
          f"({'realtime' if args.realtime else 'fast virtual-time'})")

    for step in range(max_steps):
        t_start = time.monotonic()

        now = time.monotonic() if args.realtime else step * loop_period

        need_inf = should_trigger_new_inference(
            cached_chunk_exists=(cached_chunk is not None),
            inference_thread_running=False,
            time_since_last_inference=(now - last_inf_time),
            inference_interval=inference_interval,
        )
        if need_inf:
            frame = min(sim_frame, T - 1)
            head = load_head_cam_frame(args.dataset_path, args.episode_index, frame)
            obs = build_observation(state[frame], head, prompt)
            t_inf = time.monotonic()
            action, _info = client.get_action(obs)
            delay = time.monotonic() - t_inf
            mt = np.asarray(action.get("motion_token", action.get("action.motion_token")),
                            dtype=np.float32)
            lh = np.asarray(action.get("left_hand_joints", action.get("action.left_hand_joints")),
                            dtype=np.float32)
            rh = np.asarray(action.get("right_hand_joints", action.get("action.right_hand_joints")),
                            dtype=np.float32)
            if mt.ndim == 3:
                mt, lh, rh = mt[0], lh[0], rh[0]
            cached_chunk = {"motion_token": mt, "left_hand_joints": lh, "right_hand_joints": rh}
            chunk_index = calculate_latency_compensated_index(
                delay, args.action_publish_rate, args.action_horizon)
            last_inf_time = now
            n_inferences += 1
            latencies.append(delay)
            chunk_marks.append(step)
            green(f"[loop] step {step}: new chunk (inf #{n_inferences}, "
                  f"latency {delay*1000:.1f} ms, start idx {chunk_index})")

        if cached_chunk is None:
            continue
        h = cached_chunk["motion_token"].shape[0]
        idx = min(chunk_index, h - 1)
        token = cached_chunk["motion_token"][idx]
        left = cached_chunk["left_hand_joints"][idx]
        right = cached_chunk["right_hand_joints"][idx]

        body = decoder.step(token)
        ex_body.append(body)
        ex_hands.append(np.concatenate([left, right]).astype(np.float32))
        used_tokens.append(token.copy())
        gt_tokens_at_step.append(gt_action[min(sim_frame, T - 1), :TOKEN_DIM].copy())
        step_time.append(step * loop_period)

        chunk_index = min(chunk_index + 1, args.action_horizon - 1)
        sim_frame += 1

        if args.realtime:
            rem = loop_period - (time.monotonic() - t_start)
            if rem > 0:
                time.sleep(rem)

    client.close()
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    if logf:
        logf.close()

    ex_body = np.asarray(ex_body, np.float32)
    ex_hands = np.asarray(ex_hands, np.float32)
    used_tokens = np.asarray(used_tokens, np.float32)
    gt_tokens_at_step = np.asarray(gt_tokens_at_step, np.float32)

    out_npz = os.path.join(args.out_dir, f"rollout_ep{args.episode_index}.npz")
    np.savez(
        out_npz,
        backend="mujoco_kinematic",
        embodiment_tag=args.embodiment_tag,
        episode_index=args.episode_index,
        prompt=prompt,
        control_freq=args.action_publish_rate,
        inference_rate=args.rate,
        action_horizon=args.action_horizon,
        executed_body=ex_body,
        executed_hands=ex_hands,
        used_tokens=used_tokens,
        gt_tokens=gt_tokens_at_step,
        step_time_s=np.asarray(step_time, np.float32),
        chunk_boundaries=np.asarray(chunk_marks, np.int64),
        inference_latencies_s=np.asarray(latencies, np.float32),
        n_inferences=n_inferences,
        track_gain=args.track_gain,
        physics=False,
    )
    summary = {
        "backend": "mujoco_kinematic",
        "episode_index": args.episode_index,
        "steps_executed": int(ex_body.shape[0]),
        "n_inferences": n_inferences,
        "mean_inference_latency_ms": float(np.mean(latencies) * 1000) if latencies else None,
        "control_freq_hz": args.action_publish_rate,
        "inference_rate_hz": args.rate,
        "action_horizon": args.action_horizon,
        "token_match_pred_vs_gt_mae": float(np.abs(used_tokens - gt_tokens_at_step).mean())
        if used_tokens.size else None,
        "executed_body_shape": list(ex_body.shape),
        "executed_hands_shape": list(ex_hands.shape),
        "decoder_peak_abs_target_rad": round(decoder.max_abs, 3),
        "decoder_clipped_steps": decoder.clipped_steps,
        "track_gain": args.track_gain,
        "npz": out_npz,
        "physics": False,
        "note": ("ReplayPolicy replays the server's loaded episode's GT tokens; "
                 "with a real Gr00tPolicy checkpoint the same client streams "
                 "predicted tokens. Stability/roll-pitch require the isaaclab "
                 "backend on the 5090."),
    }
    with open(os.path.join(args.out_dir, f"summary_ep{args.episode_index}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    green(f"[done] wrote {out_npz}")
    print(json.dumps(summary, indent=2))

    if args.plot:
        _plot_rollout(args, ex_body, ex_hands, used_tokens, gt_tokens_at_step)
    return summary

def _plot_rollout(args, ex_body, ex_hands, tokens, gt_tokens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 9))
    ax[0].plot(np.linalg.norm(ex_body, axis=1), label="|decoded body target|_2")
    ax[0].set_title(f"ep{args.episode_index} decoded body (29) -- kinematic closed loop")
    ax[0].set_ylabel("rad"); ax[0].legend()
    for j in range(min(ex_hands.shape[1], 14)):
        ax[1].plot(ex_hands[:, j], lw=0.7)
    ax[1].set_title("hand joints (14) passthrough"); ax[1].set_ylabel("cmd")
    ax[2].plot(np.linalg.norm(tokens, axis=1), label="|policy token|_2")
    ax[2].plot(np.linalg.norm(gt_tokens, axis=1), "--", label="|GT token|_2")
    ax[2].set_title("motion token norm (pred vs GT)"); ax[2].set_xlabel("50 Hz control step")
    ax[2].legend()
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"rollout_ep{args.episode_index}.png")
    fig.savefig(out, dpi=90)
    green(f"[plot] wrote {out}")

def run_dataset_gt(args):
    """No-server GT-targets rollout: decode the held-out episode's OWN recorded
    SONIC tokens closed-loop. This is SONIC's own tracking floor (the 'GT-targets'
    row of executed_metrics) and needs no PolicyServer -- runs fully on cluster."""
    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 78)
    green("PHASE C2 hierarchy runner -- token-source=dataset_gt (GT-targets floor)")
    print("Decoding the episode's RECORDED SONIC tokens closed-loop (no server).")
    print("Kinematic actuator-lag loop, NO contact physics (5090 for physics).")
    print("=" * 78)
    state, gt_action, prompt, pq_path = load_episode(args.dataset_path, args.episode_index)
    T = len(state)
    fps = 30.0
    print(f"[data] episode {args.episode_index}: {T} frames @ {fps} Hz; prompt: {prompt[:70]}...")
    decoder = DecoderClosedLoop(args.decoder_onnx, control_freq=fps, track_gain=args.track_gain)
    n = min(args.max_steps if args.max_steps > 0 else T, T)
    ex_body, ex_hands, used_tokens = [], [], []
    for f in range(n):
        tok = gt_action[f, :TOKEN_DIM]
        ex_body.append(decoder.step(tok))
        ex_hands.append(gt_action[f, TOKEN_DIM:TOKEN_DIM + 14].astype(np.float32))
        used_tokens.append(tok.copy())
    ex_body = np.asarray(ex_body, np.float32)
    ex_hands = np.asarray(ex_hands, np.float32)
    used_tokens = np.asarray(used_tokens, np.float32)
    out_npz = os.path.join(args.out_dir, f"rollout_ep{args.episode_index}.npz")
    np.savez(out_npz, backend="dataset_gt", episode_index=args.episode_index, prompt=prompt,
             control_freq=fps, executed_body=ex_body, executed_hands=ex_hands,
             used_tokens=used_tokens, gt_tokens=used_tokens,
             step_time_s=np.arange(n, dtype=np.float32) / fps,
             track_gain=args.track_gain, physics=False)
    summary = {"backend": "dataset_gt", "episode_index": args.episode_index,
               "steps_executed": int(n), "control_freq_hz": fps,
               "decoder_peak_abs_target_rad": round(decoder.max_abs, 3),
               "decoder_clipped_steps": decoder.clipped_steps, "npz": out_npz,
               "physics": False, "row": "GT-targets (SONIC tracking floor, kinematic)"}
    with open(os.path.join(args.out_dir, f"summary_ep{args.episode_index}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    green(f"[done] wrote {out_npz}")
    print(json.dumps(summary, indent=2))
    if args.plot:
        _plot_rollout(args, ex_body, ex_hands, used_tokens, used_tokens)
    return summary

def run_isaaclab(args):
    msg = (
        "\n" + "=" * 78 + "\n"
        "sim-backend=isaaclab is the FAITHFUL contact-physics token->physics loop\n"
        "and requires Isaac Sim + Isaac Lab (SONIC's eval_agent_trl / token\n"
        "injection). It is NOT installable on this cluster.\n\n"
        "Run it on the RTX 5090 following:\n"
        "  results/isaacsim_roundtrip_runbook.md   (SONIC checkpoint in physics)\n"
        "  results/eval_on_5090_runbook.md          (full base-vs-FT executed eval)\n\n"
        "The hierarchy wiring (obs -> PolicyServer -> chunks -> SONIC decoder ->\n"
        "50 Hz control) is identical; on the 5090 the decoder is embedded in the\n"
        "IsaacLab env instead of the kinematic loop used by --sim-backend mujoco.\n"
        + "=" * 78 + "\n"
    )
    print(msg)
    raise SystemExit(3)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-backend", choices=["mujoco", "isaaclab"], default="mujoco")
    ap.add_argument("--token-source", choices=["server", "dataset_gt"], default="server",
                    help="server: obs->PolicyServer->chunks (full hierarchy). "
                         "dataset_gt: decode the episode's recorded tokens (GT-targets "
                         "floor, no server). mujoco backend only.")
    ap.add_argument("--episode-index", type=int, default=0, help="held-out episode (0/2/16)")
    ap.add_argument("--dataset-path", default=DEF_DATASET)
    ap.add_argument("--decoder-onnx", default=DEF_DECODER)
    ap.add_argument("--embodiment-tag", default="UNITREE_G1_SONIC")

    ap.add_argument("--policy-host", default="127.0.0.1")
    ap.add_argument("--policy-port", type=int, default=5551)
    ap.add_argument("--autostart-server", action="store_true",
                    help="spawn run_gr00t_server.py (ReplayPolicy unless --model-path)")
    ap.add_argument("--server-python", default=DEF_SERVER_PY)
    ap.add_argument("--isaac-gr00t-repo", default=DEF_ISAAC_REPO)
    ap.add_argument("--model-path", default=None,
                    help="GR00T checkpoint dir -> Gr00tPolicy (needs GPU); "
                         "omit for ReplayPolicy GT-token smoke")
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--server-wait-s", type=float, default=180.0)
    ap.add_argument("--timeout-ms", type=int, default=120000)

    ap.add_argument("--action-publish-rate", type=float, default=50.0)
    ap.add_argument("--rate", type=float, default=2.5, help="inference forward-pass rate (Hz)")
    ap.add_argument("--action-horizon", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=240, help="50Hz control steps (0=auto)")
    ap.add_argument("--track-gain", type=float, default=0.3,
                    help="first-order actuator-lag gain for the kinematic loop "
                         "(1.0=perfect track); mujoco backend only")
    ap.add_argument("--realtime", action="store_true", help="sleep to real 50Hz (default: fast)")

    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.out_dir is None:
        tag = "gt_targets" if (args.sim_backend == "mujoco" and args.token_source == "dataset_gt") \
            else args.sim_backend
        args.out_dir = f"{PROJ}/results/hierarchy/{tag}_ep{args.episode_index}"
    os.makedirs(args.out_dir, exist_ok=True)

    if args.sim_backend == "isaaclab":
        run_isaaclab(args)
    elif args.token_source == "dataset_gt":
        run_dataset_gt(args)
    else:
        run_mujoco(args)

if __name__ == "__main__":
    main()
