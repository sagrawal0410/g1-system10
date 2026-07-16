#!/usr/bin/env python3
"""
Phase C3 -- executed-motion metrics for the hierarchy eval.

Per task (held-out episode) it compares the EXECUTED motion against the DATASET
reference trajectory and emits, for each row it can find:

  rows  = {base-proxy | GR00T-A(hands-out) | GR00T-A+Phase-D | ACT | GT-targets}
          + an always-available real-robot floor row computed here on the cluster:
          `real-robot(q_current)` = the demo's own commanded-vs-achieved tracking.

  metrics per (row, task):
    - ee_wrist_err_l/r_m       : L2 error (metres) between executed and reference
                                 wrist positions, via MuJoCo forward kinematics
    - joint_rmse_rad           : body-joint (29) tracking RMSE
    - joint_maxabs_rad
    - fall / terminated        : stability events (from physics rollout logs)
    - root_roll_max / root_pitch_max_rad
    - decoder_peak_abs_rad / clipped_frac : divergence diagnostic (kinematic logs)

Outputs:
    results/executed_metrics.csv     (long format: row,task,episode,metric,value,physics)
    results/executed_videos/<task>.mp4  (MuJoCo playback of the reference motion,
                                          the demo each policy must reproduce)

DESIGN: "consume whatever rollout logs exist."
  - The physics rows (base-proxy / GR00T-A / +Phase-D / ACT) are produced by the
    CLOSED-LOOP eval in Isaac Lab on the RTX 5090 (see results/eval_on_5090_runbook.md).
    Drop their rollout npz into a dir and pass `--row LABEL=DIR`.
  - GT-targets is produced on the cluster now by
    `run_hierarchy.py --token-source dataset_gt` (SONIC's kinematic tracking floor).
  - Any metric a given log can't support (e.g. EE for a kinematic log that lacks
    named-order absolute joint positions; stability for a no-physics log) is written
    as NaN rather than guessed.

EXECUTED-LOG CONTRACT for FK/EE/stability (what the 5090 export must save in the npz):
    body_qpos_named : (N,29) float  ABSOLUTE joint angles (rad), in the named order
                      [L-leg6, R-leg6, waist3, L-arm7, R-arm7] (== MuJoCo g1 joint
                      order == dataset robot_q_desired[7:36] order).
    root_pose       : (N,7)  float  [x,y,z, qw,qx,qy,qz]  (optional; for stability +
                      root-frame EE). If absent, FK uses identity root.
  Kinematic cluster logs from run_hierarchy store `executed_body` (relative, decoder
  order) instead -> EE/joint metrics are NaN for those (divergence diagnostics only).
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

REPO = "/lambdafs/shaurya/g1_sonic_system1/repos/GR00T-WholeBodyControl"
G1_XML = f"{REPO}/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
RAW = "/lambdafs/shaurya/g1_sonic_system1/data/g1_raw_full"
DEF_EVAL_SPLIT = os.path.expanduser("~/g1_sonic_system1/results/eval_split.json")

NAMED_29 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
LEFT_WRIST_BODY = "left_wrist_yaw_link"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"

class G1FK:
    def __init__(self, xml=G1_XML):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.qadr = {}
        for name in NAMED_29:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"joint {name} not in model")
            self.qadr[name] = int(self.model.jnt_qposadr[jid])
        fb = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self.free_adr = int(self.model.jnt_qposadr[fb]) if fb >= 0 else None

    def wrists(self, q29_abs, root7=None):
        self.data.qpos[:] = self.model.qpos0
        if root7 is not None and self.free_adr is not None:
            self.data.qpos[self.free_adr:self.free_adr + 7] = root7
        for i, name in enumerate(NAMED_29):
            self.data.qpos[self.qadr[name]] = q29_abs[i]
        self.mj.mj_forward(self.model, self.data)
        lw = self.data.body(LEFT_WRIST_BODY).xpos.copy()
        rw = self.data.body(RIGHT_WRIST_BODY).xpos.copy()
        return lw, rw

    def wrist_traj(self, q29_abs_seq, root7_seq=None):
        L, R = [], []
        for t in range(len(q29_abs_seq)):
            r7 = root7_seq[t] if root7_seq is not None else None
            lw, rw = self.wrists(q29_abs_seq[t], r7)
            L.append(lw); R.append(rw)
        return np.asarray(L), np.asarray(R)

def load_reference(task, session):
    """Return dict with q_desired[N,36], q_current[N,36], fps."""
    import pyarrow.parquet as pq
    ep_dir = os.path.join(RAW, task, session)
    files = sorted(glob.glob(f"{ep_dir}/data/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {ep_dir}")
    cols = ["action.robot_q_desired", "observation.state.robot_q_current"]
    arrs = {c: [] for c in cols}
    for f in files:
        tbl = pq.read_table(f, columns=[c for c in cols if c])
        for c in cols:
            arrs[c].append(np.asarray(tbl.column(c).to_pylist(), dtype=np.float32))
    qd = np.concatenate(arrs["action.robot_q_desired"])
    qc = np.concatenate(arrs["observation.state.robot_q_current"])
    return {"q_desired": qd, "q_current": qc, "fps": 30.0}

def resample(x, n):
    """Linear-resample sequence x (T,D) to length n."""
    T = len(x)
    if T == n:
        return x
    idx = np.linspace(0, T - 1, n)
    lo = np.floor(idx).astype(int); hi = np.ceil(idx).astype(int)
    w = (idx - lo)[:, None]
    return (1 - w) * x[lo] + w * x[hi]

def load_rollout(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    return out

def load_executed_raw(ep_dir):
    """Load a physics rollout recorded by run_data_exporter.py on the 5090 (raw
    dataset format). robot_q_current == achieved whole-body state each step:
    [0:7] root pose (x,y,z,qw,qx,qy,qz), [7:36] the 29 named body joints.
    Returns a rollout-like dict FK/stability can consume directly."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(f"{ep_dir}/data/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"no recorded parquet under {ep_dir}")
    qc = []
    for f in files:
        tbl = pq.read_table(f, columns=["observation.state.robot_q_current"])
        qc.append(np.asarray(tbl.column("observation.state.robot_q_current").to_pylist(),
                             dtype=np.float32))
    qc = np.concatenate(qc)
    return {"body_qpos_named": qc[:, 7:36].astype(np.float32),
            "root_pose": qc[:, 0:7].astype(np.float32), "physics": np.bool_(True)}

def executed_named_qpos(roll):
    """Return (q29_abs[N,29], root7[N,7] or None) if the log supports FK, else (None,None)."""
    if "body_qpos_named" in roll:
        q = np.asarray(roll["body_qpos_named"], dtype=np.float32)
        root = np.asarray(roll["root_pose"], dtype=np.float32) if "root_pose" in roll else None
        return q, root
    return None, None

def _f(x):
    return float(x) if np.isfinite(x) else float("nan")

def wrist_err(exec_L, exec_R, ref_L, ref_R):
    n = min(len(exec_L), len(ref_L))
    el = np.linalg.norm(exec_L[:n] - ref_L[:n], axis=1)
    er = np.linalg.norm(exec_R[:n] - ref_R[:n], axis=1)
    return el, er

def stability(root7_seq):
    if root7_seq is None:
        return {"fall": np.nan, "root_roll_max_rad": np.nan, "root_pitch_max_rad": np.nan,
                "min_root_z_m": np.nan}
    z = root7_seq[:, 2]
    quat = root7_seq[:, 3:7]
    w, x, y, zc = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = np.arctan2(2 * (w * x + y * zc), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - zc * x), -1, 1))
    return {"fall": float(z.min() < 0.4), "root_roll_max_rad": float(np.abs(roll).max()),
            "root_pitch_max_rad": float(np.abs(pitch).max()), "min_root_z_m": float(z.min())}

def metrics_for_rollout(fk, roll, ref, physics):
    """Compute all supported metrics; NaN where unsupported."""
    m = {}
    q29, root = executed_named_qpos(roll)
    ref_qd = ref["q_desired"]
    if q29 is not None:
        n = len(q29)
        ref_body = resample(ref_qd[:, 7:36], n)
        ref_root = resample(ref_qd[:, 0:7], n)

        diff = q29 - ref_body
        m["joint_rmse_rad"] = _f(np.sqrt(np.mean(diff ** 2)))
        m["joint_maxabs_rad"] = _f(np.abs(diff).max())

        eL, eR = fk.wrist_traj(q29)
        rL, rR = fk.wrist_traj(ref_body)
        el, er = wrist_err(eL, eR, rL, rR)
        m["ee_wrist_err_l_m"] = _f(el.mean()); m["ee_wrist_err_l_max_m"] = _f(el.max())
        m["ee_wrist_err_r_m"] = _f(er.mean()); m["ee_wrist_err_r_max_m"] = _f(er.max())
        m.update({f"stab_{k}": v for k, v in stability(root).items()})
    else:
        for k in ["joint_rmse_rad", "joint_maxabs_rad", "ee_wrist_err_l_m",
                  "ee_wrist_err_l_max_m", "ee_wrist_err_r_m", "ee_wrist_err_r_max_m",
                  "stab_fall", "stab_root_roll_max_rad", "stab_root_pitch_max_rad",
                  "stab_min_root_z_m"]:
            m[k] = float("nan")

    m["decoder_peak_abs_rad"] = _f(roll["decoder_peak_abs_target_rad"]) \
        if "decoder_peak_abs_target_rad" in roll else float("nan")
    if "executed_body" in roll and "decoder_peak_abs_target_rad" not in roll:
        m["decoder_peak_abs_rad"] = _f(np.abs(np.asarray(roll["executed_body"])).max())
    m["physics"] = bool(physics)
    return m

def render_reference_video(fk, ref, out_mp4, max_frames=300, size=(480, 640)):
    import mujoco, cv2
    os.environ.setdefault("MUJOCO_GL", "egl")
    qd = ref["q_desired"]
    n = len(qd)
    stride = max(1, n // max_frames)
    idxs = list(range(0, n, stride))
    try:
        renderer = mujoco.Renderer(fk.model, size[0], size[1])
    except Exception as e:
        print(f"  [video] renderer unavailable ({type(e).__name__}); skipping video")
        return False
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, 15.0, (size[1], size[0]))
    for i in idxs:
        fk.data.qpos[:] = fk.model.qpos0
        if fk.free_adr is not None:
            fk.data.qpos[fk.free_adr:fk.free_adr + 7] = qd[i, 0:7]
        for j, name in enumerate(NAMED_29):
            fk.data.qpos[fk.qadr[name]] = qd[i, 7 + j]
        mujoco.mj_forward(fk.model, fk.data)
        renderer.update_scene(fk.data)
        img = renderer.render()
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    vw.release()
    del renderer
    print(f"  [video] wrote {out_mp4} ({len(idxs)} frames)")
    return True

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-split", default=DEF_EVAL_SPLIT)
    ap.add_argument("--row", action="append", default=[],
                    help="LABEL=DIR (repeatable). DIR holds rollout_ep{ep}.npz per eval "
                         "episode. e.g. --row GR00T-A=results/rollouts_5090/ft")
    ap.add_argument("--raw-row", action="append", default=[],
                    help="LABEL=EP=DIR (repeatable). DIR is a run_data_exporter recording "
                         "(raw format) of the closed-loop physics rollout for eval episode "
                         "EP. e.g. --raw-row GR00T-A=0=/data/rec/bottle_ft")
    ap.add_argument("--auto-discover", action="store_true",
                    help="also add rows from results/hierarchy/*/ (dir name = label)")
    ap.add_argument("--out-csv", default=os.path.expanduser("~/g1_sonic_system1/results/executed_metrics.csv"))
    ap.add_argument("--video-dir", default=os.path.expanduser("~/g1_sonic_system1/results/executed_videos"))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--proj", default=os.path.expanduser("~/g1_sonic_system1"))
    args = ap.parse_args()

    with open(args.eval_split) as f:
        split = json.load(f)
    eval_eps = split["eval"]

    rows = {}
    eval_ep_ids = [e["episode_index"] for e in eval_eps]

    def add_dir_row(label, d):
        base = d if os.path.isabs(d) else os.path.join(args.proj, d)
        ep_map = rows.setdefault(label, {})
        for ep in eval_ep_ids:
            cand = glob.glob(f"{base}/**/rollout_ep{ep}.npz", recursive=True)
            if cand:
                ep_map[ep] = sorted(cand)[0]

    for spec in args.row:
        if "=" not in spec:
            sys.exit(f"--row expects LABEL=DIR, got {spec}")
        label, d = spec.split("=", 1)
        add_dir_row(label, d)

    if args.auto_discover or not rows:

        import re
        hier = os.path.join(args.proj, "results/hierarchy")
        for npz in sorted(glob.glob(hier + "/*/rollout_ep*.npz")):
            base = os.path.basename(os.path.dirname(npz))
            mlabel = re.sub(r"_ep\d+$", "", base)
            mep = re.search(r"rollout_ep(\d+)\.npz$", npz)
            if not mep:
                continue
            ep = int(mep.group(1))
            label = {"gt_targets": "GT-targets"}.get(mlabel, f"hierarchy-{mlabel}")
            rows.setdefault(label, {})[ep] = npz

    raw_rows = {}
    for spec in args.raw_row:
        parts = spec.split("=", 2)
        if len(parts) != 3:
            sys.exit(f"--raw-row expects LABEL=EP=DIR, got {spec}")
        label, ep_s, d = parts
        raw_rows.setdefault(label, {})[int(ep_s)] = d if os.path.isabs(d) \
            else os.path.join(args.proj, d)

    print("Rows discovered:")
    for k, v in rows.items():
        print(f"  {k} (npz): episodes {sorted(v.keys())}")
    for k, v in raw_rows.items():
        print(f"  {k} (raw recording): episodes {sorted(v.keys())}")
    fk = G1FK()
    os.makedirs(args.video_dir, exist_ok=True)

    records = []

    ref_cache = {}
    for ev in eval_eps:
        task, session, ep = ev["task"], ev["session"], ev["episode_index"]
        try:
            ref = load_reference(task, session)
        except Exception as e:
            print(f"[ref] episode {ep} ({task}): FAILED to load reference: {e}")
            continue
        ref_cache[ep] = (task, ref)

        n = len(ref["q_desired"])
        eL, eR = fk.wrist_traj(ref["q_current"][:, 7:36])
        rL, rR = fk.wrist_traj(ref["q_desired"][:, 7:36])
        el, er = wrist_err(eL, eR, rL, rR)
        jd = ref["q_current"][:, 7:36] - ref["q_desired"][:, 7:36]
        floor = {
            "ee_wrist_err_l_m": _f(el.mean()), "ee_wrist_err_l_max_m": _f(el.max()),
            "ee_wrist_err_r_m": _f(er.mean()), "ee_wrist_err_r_max_m": _f(er.max()),
            "joint_rmse_rad": _f(np.sqrt(np.mean(jd ** 2))),
            "joint_maxabs_rad": _f(np.abs(jd).max()), "physics": True,
        }
        for k, v in floor.items():
            records.append(("real-robot(q_current)", task, ep, k, v, True))
        print(f"[floor] {task} ep{ep}: real-robot wrist track err "
              f"L={floor['ee_wrist_err_l_m']*1000:.1f}mm R={floor['ee_wrist_err_r_m']*1000:.1f}mm")

        if not args.no_video:
            render_reference_video(fk, ref, os.path.join(args.video_dir, f"{task}_ep{ep}.mp4"))

    for label, ep_map in rows.items():
        for ev in eval_eps:
            ep = ev["episode_index"]
            if ep not in ref_cache:
                continue
            task, ref = ref_cache[ep]
            if ep not in ep_map:
                print(f"[{label}] ep{ep} ({task}): no rollout npz -> row absent")
                continue
            roll = load_rollout(ep_map[ep])
            physics = bool(roll["physics"]) if "physics" in roll else False
            m = metrics_for_rollout(fk, roll, ref, physics)
            for k, v in m.items():
                if k == "physics":
                    continue
                records.append((label, task, ep, k, v, physics))
            print(f"[{label}] ep{ep} ({task}): physics={physics} "
                  f"ee_L={m['ee_wrist_err_l_m']} joint_rmse={m['joint_rmse_rad']} "
                  f"peak_abs={m['decoder_peak_abs_rad']}")

    for label, ep_map in raw_rows.items():
        for ep, d in ep_map.items():
            if ep not in ref_cache:
                print(f"[{label}] ep{ep}: not in eval split -> skipped")
                continue
            task, ref = ref_cache[ep]
            try:
                roll = load_executed_raw(d)
            except Exception as e:
                print(f"[{label}] ep{ep} ({task}): failed to load recording: {e}")
                continue
            m = metrics_for_rollout(fk, roll, ref, physics=True)
            for k, v in m.items():
                if k == "physics":
                    continue
                records.append((label, task, ep, k, v, True))
            print(f"[{label}] ep{ep} ({task}): physics=True "
                  f"ee_L={m['ee_wrist_err_l_m']} ee_R={m['ee_wrist_err_r_m']} "
                  f"joint_rmse={m['joint_rmse_rad']} fall={m['stab_fall']}")

    import csv
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "task", "episode", "metric", "value", "physics"])
        for r in records:
            w.writerow(r)
    print(f"\nWrote {len(records)} metric rows -> {args.out_csv}")
    print("Rows present:", sorted(set(r[0] for r in records)))
    print("NOTE: physics rows (base-proxy/GR00T-A/+Phase-D/ACT) come from the 5090 "
          "closed-loop eval; see results/eval_on_5090_runbook.md.")

if __name__ == "__main__":
    main()
