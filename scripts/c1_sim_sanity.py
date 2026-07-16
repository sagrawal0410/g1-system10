#!/usr/bin/env python3
"""
Phase C1 -- stock-sim + SONIC-decoder sanity (cluster, `sonic` conda env).

Two independent checks, both runnable on the login node (CPU + headless EGL):

  (1) MuJoCo sim substrate: load the WBC repo's own G1 scene and step physics
      headless for N steps; confirm the model loads, steps without NaN, and can
      render offscreen (EGL) -> the sim env is healthy on this box.

  (2) SONIC universal-token decoder ONNX, OPEN-LOOP: pull a REAL 64-dim motion
      token from a held-out episode of g1_encoded_sonic and push it through the
      released decoder (`model_decoder.onnx`, 994->29). Confirm the decoder both
      runs and *consumes* the token (zeroing the token perturbs the 29-dim body
      action). Reproduces the Phase-0.5 token-sensitivity evidence.

IMPORTANT ARCHITECTURE NOTE (read the log this script prints):
  The WBC repo's stock MuJoCo entry point `gear_sonic/scripts/run_sim_loop.py`
  runs the *decoupled* WBC (arm interpolation + RL legs) over unitree_sdk2py DDS
  channels -- it does NOT consume the SONIC universal token, and it needs
  `unitree_sdk2py` (absent on this cluster). So "stock MuJoCo demo" here means:
  confirm the MuJoCo substrate + the released token decoder in isolation. The
  faithful token->physics closed loop is Isaac-Lab / C++-TensorRT only (5090).

Decoder obs layout follows the deploy config that actually feeds THIS onnx
(`gear_sonic_deploy/policy/release/observation_config.yaml`), term order:
  token(64) | base_ang_vel_10f(30) | body_joint_pos_10f(290) |
  body_joint_vel_10f(290) | last_actions_10f(290) | gravity_dir_10f(30) = 994
(The Phase-0.5 verdict lists gravity second; that was its offline-probe order.
 Either way this is an open-loop sanity, not the faithful physics decode.)
"""
import argparse
import glob
import os
import sys
import traceback

import numpy as np

REPO = "/lambdafs/shaurya/g1_sonic_system1/repos/GR00T-WholeBodyControl"
DECODER = f"{REPO}/gear_sonic_deploy/policy/release/model_decoder.onnx"
SCENES = [
    f"{REPO}/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml",
    f"{REPO}/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml",
]
DATASET = "/lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic"
N_BODY = 29
N_HIST = 10
TOKEN_DIM = 64

def log(msg):
    print(msg, flush=True)

def check_mujoco(steps: int, render: bool):
    log("\n=== (1) MuJoCo sim substrate ===")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    xml = next((s for s in SCENES if os.path.exists(s)), None)
    if xml is None:
        log(f"  FAIL: no G1 scene found in {SCENES}")
        return False
    log(f"  scene: {xml}")
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    log(f"  model loaded: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody}")
    mujoco.mj_forward(m, d)
    for _ in range(steps):
        mujoco.mj_step(m, d)
    ok = np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
    log(f"  stepped {steps} steps; qpos finite={np.all(np.isfinite(d.qpos))} "
        f"qvel finite={np.all(np.isfinite(d.qvel))}")
    if render:
        try:
            r = mujoco.Renderer(m, 240, 320)
            r.update_scene(d)
            img = r.render()
            log(f"  offscreen EGL render OK: {img.shape}")
            del r
        except Exception as e:
            log(f"  offscreen render unavailable ({type(e).__name__}: {str(e)[:80]}) "
                "-- videos would need xvfb/osmesa")
    log(f"  MuJoCo substrate: {'PASS' if ok else 'FAIL'}")
    return ok

def _real_token(episode_index: int):
    import pyarrow.parquet as pq

    cand = glob.glob(f"{DATASET}/data/**/episode_{episode_index:06d}.parquet", recursive=True)
    if not cand:
        return None
    tbl = pq.read_table(cand[0], columns=["action"])
    a = np.asarray(tbl.column("action").to_pylist(), dtype=np.float32)

    row = a[min(len(a) // 2, len(a) - 1)]
    return row[:TOKEN_DIM].astype(np.float32)

def _assemble_obs(token: np.ndarray) -> np.ndarray:
    """Deploy-order 994-dim decoder obs, upright/at-default proprio (open-loop)."""
    base_ang_vel = np.zeros(3 * N_HIST, np.float32)
    body_pos = np.zeros(N_BODY * N_HIST, np.float32)
    body_vel = np.zeros(N_BODY * N_HIST, np.float32)
    last_act = np.zeros(N_BODY * N_HIST, np.float32)
    grav = np.tile(np.array([0, 0, -1], np.float32), N_HIST)
    obs = np.concatenate([token.astype(np.float32), base_ang_vel, body_pos,
                          body_vel, last_act, grav]).astype(np.float32)
    assert obs.shape[0] == 994, obs.shape
    return obs[None, :]

def check_decoder():
    log("\n=== (2) SONIC token decoder ONNX, open-loop ===")
    if not os.path.exists(DECODER):
        log(f"  FAIL: decoder not found at {DECODER}")
        return False
    import onnxruntime as ort

    sess = ort.InferenceSession(DECODER, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    log(f"  decoder input {iname} {sess.get_inputs()[0].shape} -> "
        f"{sess.get_outputs()[0].name} {sess.get_outputs()[0].shape}")

    tok = None
    used = None
    for ep in (0, 2, 16):
        tok = _real_token(ep)
        if tok is not None:
            used = ep
            break
    if tok is None:
        log("  WARN: no dataset token found; using a synthetic FSQ-grid token")
        tok = (np.round(np.random.uniform(-0.6, 0.6, TOKEN_DIM) * 16) / 16).astype(np.float32)
    else:
        log(f"  real token from held-out episode {used}: "
            f"range [{tok.min():.3f}, {tok.max():.3f}], "
            f"on-1/16-grid={np.allclose(tok*16, np.round(tok*16), atol=1e-3)}")

    out = sess.run(None, {iname: _assemble_obs(tok)})[0].ravel()
    out0 = sess.run(None, {iname: _assemble_obs(np.zeros(TOKEN_DIM, np.float32))})[0].ravel()
    sens = float(np.abs(out - out0).mean())
    arms = slice(15, 29)
    sens_arms = float(np.abs(out[arms] - out0[arms]).mean())
    log(f"  decoder output dim: {out.shape[0]} (expect {N_BODY})")
    log(f"  |action| mean={np.abs(out).mean():.4f} max={np.abs(out).max():.4f}")
    log(f"  token sensitivity (mean|d| all)={sens:.4f} rad, arms={sens_arms:.4f} rad")
    ok = out.shape[0] == N_BODY and sens > 0.05
    log(f"  decoder consumes token & outputs 29-dim body action: "
        f"{'PASS' if ok else 'FAIL'} "
        f"(Phase-0.5 saw ~0.47 all / ~0.51-0.57 arms)")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    log("Phase C1 sanity -- stock MuJoCo substrate + SONIC token decoder (open-loop)")
    log(f"repo={REPO}")
    log("NOTE: stock run_sim_loop.py = DECOUPLED WBC (arm-interp + RL legs), needs "
        "unitree_sdk2py (absent here) and does NOT use the SONIC token decoder.")

    results = {}
    try:
        results["mujoco"] = check_mujoco(args.steps, render=not args.no_render)
    except Exception:
        log("  MuJoCo check raised:\n" + traceback.format_exc())
        results["mujoco"] = False
    try:
        results["decoder"] = check_decoder()
    except Exception:
        log("  decoder check raised:\n" + traceback.format_exc())
        results["decoder"] = False

    log("\n=== SUMMARY ===")
    for k, v in results.items():
        log(f"  {k}: {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)

if __name__ == "__main__":
    main()
