#!/usr/bin/env python3
"""
Phase C (executed eval) -- FAITHFUL MuJoCo physics rollout of the SONIC token
decoder.  camera->VLA is upstream (open-loop tokens); this closes SONIC decoder
-> G1 in REAL physics (PD + gravity + ground contact) on CPU (sonic env), the
substitute for the root-blocked Isaac Sim path.

Loop (all fidelity params extracted from the repo, see below):
  * MuJoCo scene_43dof.xml (free base + ground plane, <motor> torque actuators).
  * sim 200 Hz (dt 0.005), decimation 4 -> decoder/control at 50 Hz.
  * At each 50 Hz step: assemble the 994-dim decoder obs (IsaacLab order),
    run model_decoder.onnx -> 29-dim action (IsaacLab order),
    q_target = default + action (use_default_offset, scale 1.0), convert to
    MuJoCo order, then 4x PD substeps: tau = KP*(q_target-q) - KD*qvel (MOTOR_KP/KD),
    clamped to actuator ctrlrange, set on data.ctrl.
  * Feed back MEASURED joint pos/vel + base ang-vel + projected gravity (real
    proprio -> keeps the decoder in-distribution, unlike the C2 kinematic loop).

Obs order (sonic_release/config.yaml actor `policy` group, the ONNX's training
order): token(64) | gravity_dir | base_ang_vel | joint_pos_rel | joint_vel_rel |
last_actions ; each 29-dim term (gravity/angvel are 3) with 10-frame history; NO
obs scales.  Perm from gear_sonic/envs/manager_env/robots/g1.py.

Outputs a rollout npz (body_qpos_named[N,29] + root_pose[N,7] + physics=True +
fall/peak) that executed_metrics.py --row consumes, and an executed mp4 (EGL).
"""
import argparse, glob, json, os, time
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

REPO = "/lambdafs/shaurya/g1_sonic_system1/repos/GR00T-WholeBodyControl"
SCENE = f"{REPO}/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
DECODER = f"{REPO}/gear_sonic_deploy/policy/release/model_decoder.onnx"
RAW = "/lambdafs/shaurya/g1_sonic_system1/data/g1_raw_full"
EVAL_SPLIT = os.path.expanduser("~/g1_sonic_system1/results/eval_split.json")

# ---- authoritative constants (repo) ----
NAMED_29 = [
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint",
    "left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint",
    "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
]
HAND_14 = [
    "left_hand_thumb_0_joint","left_hand_thumb_1_joint","left_hand_thumb_2_joint",
    "left_hand_middle_0_joint","left_hand_middle_1_joint","left_hand_index_0_joint","left_hand_index_1_joint",
    "right_hand_thumb_0_joint","right_hand_thumb_1_joint","right_hand_thumb_2_joint",
    "right_hand_middle_0_joint","right_hand_middle_1_joint","right_hand_index_0_joint","right_hand_index_1_joint",
]
MJ2ISAAC = np.array([0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28])
ISAAC2MJ = np.array([0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28])

# ---- SONIC TRAINING actuator model (gear_sonic/envs/manager_env/robots/g1.py) ----
# stiffness = armature * (2*pi*10)^2 ; damping = 2*damping_ratio(2) * armature * (2*pi*10)
_NF = 10.0 * 2.0 * np.pi
_A5020, _A7514, _A7522, _A4010 = 0.003609725, 0.010177520, 0.025101925, 0.00425
_S = lambda a: a * _NF**2
_D = lambda a: 2.0 * 2.0 * a * _NF
# per-joint (NAMED order) stiffness/damping/armature
_KJ = {  # name-substring -> (stiffness, damping, armature)
    "hip_pitch": (_S(_A7522), _D(_A7522), _A7522), "hip_roll": (_S(_A7522), _D(_A7522), _A7522),
    "hip_yaw": (_S(_A7514), _D(_A7514), _A7514), "knee": (_S(_A7522), _D(_A7522), _A7522),
    "ankle": (2*_S(_A5020), 2*_D(_A5020), 2*_A5020),
    "waist_yaw": (_S(_A7514), _D(_A7514), _A7514),
    "waist_roll": (2*_S(_A5020), 2*_D(_A5020), 2*_A5020),
    "waist_pitch": (2*_S(_A5020), 2*_D(_A5020), 2*_A5020),
    "shoulder": (_S(_A5020), _D(_A5020), _A5020), "elbow": (_S(_A5020), _D(_A5020), _A5020),
    "wrist_roll": (_S(_A5020), _D(_A5020), _A5020),
    "wrist_pitch": (_S(_A4010), _D(_A4010), _A4010), "wrist_yaw": (_S(_A4010), _D(_A4010), _A4010),
}
def _gain(name, i):
    for k in ("hip_pitch","hip_roll","hip_yaw","knee","ankle","waist_yaw","waist_roll",
              "waist_pitch","shoulder","elbow","wrist_roll","wrist_pitch","wrist_yaw"):
        if k in name:
            return _KJ[k][i]
    raise KeyError(name)
_NAMES = ["left_hip_pitch","left_hip_roll","left_hip_yaw","left_knee","left_ankle_pitch","left_ankle_roll",
          "right_hip_pitch","right_hip_roll","right_hip_yaw","right_knee","right_ankle_pitch","right_ankle_roll",
          "waist_yaw","waist_roll","waist_pitch",
          "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow",
          "left_wrist_roll","left_wrist_pitch","left_wrist_yaw",
          "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow",
          "right_wrist_roll","right_wrist_pitch","right_wrist_yaw"]
KP = np.array([_gain(n,0) for n in _NAMES], np.float64)
KD = np.array([_gain(n,1) for n in _NAMES], np.float64)
ARMATURE = np.array([_gain(n,2) for n in _NAMES], np.float64)
# SONIC training default joint pose (init_state.joint_pos), NAMED order
DEFAULT = np.zeros(29, np.float64)
for _i,_n in enumerate(_NAMES):
    if "hip_pitch" in _n: DEFAULT[_i] = -0.312
    elif "knee" in _n: DEFAULT[_i] = 0.669
    elif "ankle_pitch" in _n: DEFAULT[_i] = -0.363
    elif "elbow" in _n: DEFAULT[_i] = 0.6
    elif _n == "left_shoulder_roll": DEFAULT[_i] = 0.2
    elif _n == "left_shoulder_pitch": DEFAULT[_i] = 0.2
    elif _n == "right_shoulder_roll": DEFAULT[_i] = -0.2
    elif _n == "right_shoulder_pitch": DEFAULT[_i] = 0.2
N_HIST, N_BODY, TOKD = 10, 29, 64
LWRIST, RWRIST = "left_wrist_yaw_link", "right_wrist_yaw_link"


def quat_rot_inv(q, v):
    """Rotate vector v from world into body frame (q = wxyz)."""
    w, x, y, z = q
    # R^T @ v  ; build R from quat
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])
    return R.T @ v


def load_session(episode):
    with open(EVAL_SPLIT) as f:
        split = json.load(f)
    ev = next(e for e in split["eval"] if e["episode_index"] == episode)
    task, session = ev["task"], ev["session"]
    import pyarrow.parquet as pq
    files = sorted(glob.glob(f"{RAW}/{task}/{session}/data/**/*.parquet", recursive=True))
    cols = ["observation.state.robot_q_current", "action.robot_q_desired", "action.token_state"]
    d = {c: [] for c in cols}
    for f in files:
        t = pq.read_table(f, columns=cols)
        for c in cols:
            d[c].append(np.asarray(t.column(c).to_pylist(), dtype=np.float32))
    return (task, session,
            np.concatenate(d["observation.state.robot_q_current"]),
            np.concatenate(d["action.robot_q_desired"]),
            np.concatenate(d["action.token_state"]))


def load_pred_tokens(npz_path, n_expect):
    d = np.load(npz_path, allow_pickle=True)
    key = "pred_action" if "pred_action" in d.files else (
        "pred_actions" if "pred_actions" in d.files else None)
    if key is None:
        raise KeyError(f"no pred_action in {npz_path}; keys={d.files}")
    pa = np.asarray(d[key], dtype=np.float32)
    if pa.ndim == 3:  # (T, horizon, D) -> take first action of each chunk
        pa = pa[:, 0, :]
    return pa[:, :TOKD]


class Sim:
    def __init__(self, history_order="oldest_first", obs_order="gravity_first"):
        import mujoco, onnxruntime as ort
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.m.opt.timestep = 0.005
        # semi-implicit integration is far more stable for stiff PD than the
        # MJCF default (Euler) and closer to IsaacLab's implicit actuators.
        self.m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.d = mujoco.MjData(self.m)
        self.sess = ort.InferenceSession(DECODER, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.history_order = history_order
        self.obs_order = obs_order
        jid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.body_qadr = np.array([self.m.jnt_qposadr[jid(n)] for n in NAMED_29])
        self.body_vadr = np.array([self.m.jnt_dofadr[jid(n)] for n in NAMED_29])
        self.body_act = np.array([aid(n.replace("_joint","")) for n in NAMED_29])
        self.hand_qadr = np.array([self.m.jnt_qposadr[jid(n)] for n in HAND_14])
        self.hand_vadr = np.array([self.m.jnt_dofadr[jid(n)] for n in HAND_14])
        self.hand_act = np.array([aid(n.replace("_joint","")) for n in HAND_14])
        fb = jid("floating_base_joint")
        self.free_qadr = int(self.m.jnt_qposadr[fb])
        self.free_vadr = int(self.m.jnt_dofadr[fb])
        self.pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.lw = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, LWRIST)
        self.rw = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, RWRIST)
        self.ctrlrange = self.m.actuator_ctrlrange.copy()
        self.default_isaac = DEFAULT[MJ2ISAAC]
        # match IsaacLab implicit-actuator rotor inertia (stabilises the PD)
        for i, vadr in enumerate(self.body_vadr):
            self.m.dof_armature[vadr] = ARMATURE[i]

    def set_state(self, root7, body_named, hand_named=None):
        self.d.qpos[:] = 0.0
        self.d.qvel[:] = 0.0
        self.d.qpos[self.free_qadr:self.free_qadr+7] = root7
        self.d.qpos[self.body_qadr] = body_named
        if hand_named is not None:
            self.d.qpos[self.hand_qadr] = hand_named
        self.mj.mj_forward(self.m, self.d)

    def proprio(self):
        q = self.d.qpos[self.body_qadr].copy()                     # named
        qv = self.d.qvel[self.body_vadr].copy()
        q_isaac = q[MJ2ISAAC]; qv_isaac = qv[MJ2ISAAC]
        jpr = (q_isaac - self.default_isaac).astype(np.float32)    # joint_pos_rel isaac
        quat = self.d.qpos[self.free_qadr+3:self.free_qadr+7].copy()
        grav = quat_rot_inv(quat, np.array([0.0, 0.0, -1.0])).astype(np.float32)
        res = np.zeros(6)
        self.mj.mj_objectVelocity(self.m, self.d, self.mj.mjtObj.mjOBJ_BODY,
                                  self.pelvis, res, 1)             # local frame
        angvel = res[0:3].astype(np.float32)
        return grav, angvel, jpr, qv_isaac.astype(np.float32)

    def build_obs(self, token, hist):
        # hist: dict of deques (each list of frames, oldest..newest as appended)
        def stack(name):
            frames = list(hist[name])
            if self.history_order == "newest_first":
                frames = frames[::-1]
            return np.concatenate(frames).astype(np.float32)
        g, a, jp, jv, la = (stack("grav"), stack("angvel"),
                            stack("jpr"), stack("jvel"), stack("lastact"))
        if self.obs_order == "gravity_first":
            body = np.concatenate([g, a, jp, jv, la])
        else:  # angvel_first (deploy-yaml order)
            body = np.concatenate([a, jp, jv, la, g])
        obs = np.concatenate([token.astype(np.float32), body])[None]
        assert obs.shape[1] == 994, obs.shape
        return obs

    def pd(self, q_target_named, hand_target):
        q = self.d.qpos[self.body_qadr]; qv = self.d.qvel[self.body_vadr]
        tau = KP*(q_target_named - q) - KD*qv
        lo = self.ctrlrange[self.body_act, 0]; hi = self.ctrlrange[self.body_act, 1]
        self.d.ctrl[self.body_act] = np.clip(tau, lo, hi)
        hq = self.d.qpos[self.hand_qadr]; hv = self.d.qvel[self.hand_vadr]
        htau = 5.0*(hand_target - hq) - 0.5*hv
        hlo = self.ctrlrange[self.hand_act, 0]; hhi = self.ctrlrange[self.hand_act, 1]
        self.d.ctrl[self.hand_act] = np.clip(htau, hlo, hhi)


def run(args):
    import mujoco
    os.makedirs(args.out_dir, exist_ok=True)
    task, session, qc, qd, gt_tok = load_session(args.episode)
    T = len(qc); fps = 30.0
    dur = T / fps
    n_ctrl = int(round(dur * 50)) if args.max_seconds <= 0 else int(args.max_seconds*50)
    n_ctrl = min(n_ctrl, int(round(dur*50)))

    if args.token_source == "gt":
        tok30 = gt_tok[:, :TOKD]
    else:
        tok30 = load_pred_tokens(args.npz_path, T)
    print(f"[data] ep{args.episode} {task} | {T} frames @30Hz -> {n_ctrl} ctrl steps @50Hz | "
          f"tokens {args.token_source} shape {tok30.shape}", flush=True)

    sim = Sim(history_order=args.history_order, obs_order=args.obs_order)
    # init from recorded start pose (convert recorded root quat -> MuJoCo wxyz)
    root0 = qc[0, 0:7].astype(np.float64).copy()
    if args.root_quat == "xyzw":
        qx, qy, qz, qw = root0[3:7]
        root0[3:7] = [qw, qx, qy, qz]
    sim.set_state(root0, qc[0, 7:36].astype(np.float64), np.zeros(14))
    import collections
    hist = {k: collections.deque(maxlen=N_HIST) for k in ["grav","angvel","jpr","jvel","lastact"]}
    g, a, jp, jv = sim.proprio()
    for _ in range(N_HIST):
        hist["grav"].append(g); hist["angvel"].append(a)
        hist["jpr"].append(jp); hist["jvel"].append(jv)
        hist["lastact"].append(np.zeros(N_BODY, np.float32))

    log_body, log_root, log_lw, log_rw = [], [], [], []
    peak = 0.0; fell = False; last_action_isaac = np.zeros(N_BODY, np.float32)
    frames = []
    renderer = None
    if args.render:
        try:
            renderer = mujoco.Renderer(sim.m, 360, 480)
        except Exception as e:
            print(f"[video] renderer unavailable: {e}"); renderer = None

    for k in range(n_ctrl):
        f30 = min(int(k / 50.0 * fps), tok30.shape[0]-1)
        token = tok30[f30]
        if args.hold_default:
            act_isaac = np.zeros(N_BODY, np.float32)      # q_target == default
        else:
            obs = sim.build_obs(token, hist)
            act_isaac = sim.sess.run(None, {sim.iname: obs})[0].ravel().astype(np.float32)
        peak = max(peak, float(np.abs(act_isaac).max()))
        act_named = act_isaac[ISAAC2MJ]
        q_target = DEFAULT + act_named
        for _ in range(4):  # decimation
            sim.pd(q_target, np.zeros(14))
            mujoco.mj_step(sim.m, sim.d)
        # log + history update
        root7 = sim.d.qpos[sim.free_qadr:sim.free_qadr+7].copy()
        body_named = sim.d.qpos[sim.body_qadr].copy()
        log_root.append(root7.astype(np.float32))
        log_body.append(body_named.astype(np.float32))
        log_lw.append(sim.d.body(sim.lw).xpos.copy().astype(np.float32))
        log_rw.append(sim.d.body(sim.rw).xpos.copy().astype(np.float32))
        if root7[2] < 0.4 and not fell:
            fell = True
        g, a, jp, jv = sim.proprio()
        hist["grav"].append(g); hist["angvel"].append(a)
        hist["jpr"].append(jp); hist["jvel"].append(jv)
        hist["lastact"].append(act_isaac.copy())
        if renderer is not None and k % 3 == 0:
            renderer.update_scene(sim.d)
            frames.append(renderer.render().copy())
        if k % 200 == 0:
            print(f"  step {k}/{n_ctrl} rootz={root7[2]:.3f} peak={peak:.2f}", flush=True)

    log_body = np.asarray(log_body); log_root = np.asarray(log_root)
    log_lw = np.asarray(log_lw); log_rw = np.asarray(log_rw)

    # quick validation metrics vs reference (resampled to n_ctrl)
    def resamp(x):
        idx = np.linspace(0, len(x)-1, len(log_body)); lo=np.floor(idx).astype(int)
        hi=np.ceil(idx).astype(int); w=(idx-lo)[:,None]
        return (1-w)*x[lo]+w*x[hi]
    ref_body = resamp(qc[:, 7:36]); ref_root = resamp(qc[:, 0:7])
    jrmse = float(np.sqrt(np.mean((log_body - ref_body)**2)))
    jmax = float(np.abs(log_body - ref_body).max())
    out = os.path.join(args.out_dir, f"rollout_ep{args.episode}.npz")
    np.savez(out, backend="mujoco_physics", row=args.row_label, episode_index=args.episode,
             task=task, token_source=args.token_source, physics=np.bool_(True),
             body_qpos_named=log_body.astype(np.float32), root_pose=log_root.astype(np.float32),
             wrist_l=log_lw, wrist_r=log_rw,
             history_order=args.history_order, obs_order=args.obs_order,
             decoder_peak_abs_rad=peak, fell=np.bool_(fell), control_hz=50.0)
    summary = {"row": args.row_label, "episode": args.episode, "task": task,
               "token_source": args.token_source, "steps": int(len(log_body)),
               "joint_rmse_vs_qcurrent_rad": round(jrmse,4), "joint_maxabs_rad": round(jmax,3),
               "decoder_peak_abs_rad": round(peak,3), "fell": fell,
               "min_root_z_m": round(float(log_root[:,2].min()),3),
               "final_root_z_m": round(float(log_root[-1,2]),3),
               "history_order": args.history_order, "obs_order": args.obs_order, "npz": out}
    with open(os.path.join(args.out_dir, f"summary_ep{args.episode}.json"),"w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY:", json.dumps(summary), flush=True)
    if renderer is not None and frames:
        import cv2
        vp = os.path.join(args.out_dir, f"executed_ep{args.episode}.mp4")
        vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 16.0,
                             (frames[0].shape[1], frames[0].shape[0]))
        for fr in frames: vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release(); del renderer
        print(f"[video] wrote {vp}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=2)
    ap.add_argument("--token-source", choices=["gt","npz"], default="gt")
    ap.add_argument("--npz-path", default=None)
    ap.add_argument("--row-label", default="GT-targets")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--history-order", choices=["oldest_first","newest_first"], default="oldest_first")
    ap.add_argument("--obs-order", choices=["gravity_first","angvel_first"], default="angvel_first")
    ap.add_argument("--root-quat", choices=["wxyz","xyzw"], default="xyzw",
                    help="convention of recorded robot_q_current[3:7] at init (validated xyzw)")
    ap.add_argument("--hold-default", action="store_true",
                    help="isolation test: ignore decoder, PD-hold default pose")
    ap.add_argument("--max-seconds", type=float, default=-1)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.expanduser(f"~/g1_sonic_system1/results/physics/{args.row_label}")
    run(args)


if __name__ == "__main__":
    main()
