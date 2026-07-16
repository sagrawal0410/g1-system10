#!/usr/bin/env python3
"""Build the GR00T-LeRobot (v2.1) encoded dataset for embodiment unitree_g1_sonic
from the 3 token-bearing tasks (22 episodes).

Usage:
  build_encoded_dataset.py                       # default: hands ZEROED  -> g1_encoded_sonic
  build_encoded_dataset.py --hands-mode pico --out <dir>   # Option B: hands from hand_cmd_pico

action(78) = motion_token(64)<-action.token_state | left_hand_joints(7) | right_hand_joints(7)
  hands-mode=zero : hand dims all 0 (Option A, body/latent-only)
  hands-mode=pico : hand dims = per-task per-dim MIN-MAX normalized action.hand_cmd_pico
                    (pico[0:7]->left_hand_joints, pico[7:14]->right_hand_joints);
                    inactive (constant) dims -> 0; normalization params + active masks
                    stored in meta/hand_normalization.json (invertible + loss-mask ready).
state(46)  = left_leg6|right_leg6|waist3|left_arm7|right_arm7|left_hand7|right_hand7|proj_grav3
  (hand-state groups left zeroed in both modes -- see meta note.)
Hands verified meaningful: pico active dims correlate r~0.99 w/ physical hand_cmd per task.
Videos COPIED (H.264). Episode order = task-then-session alphabetical (identical across modes).
"""
import json, glob, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC = "/lambdafs/shaurya/g1_sonic_system1/data/g1_raw_full"
TASKS = ["bottle_cupnoodles_shelf", "cup_wipe_sponge_dryingrack", "floor_box_table"]
VIDEO_KEYS = ["observation.images.head_cam", "observation.images.left_wrist_cam", "observation.images.right_wrist_cam"]
FPS = 30
ACTIVE_STD_THRESH = 1e-3

hands_mode = "zero"; OUT = Path("/lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic")
exclude = set()
args = sys.argv[1:]
for i, a in enumerate(args):
    if a == "--hands-mode": hands_mode = args[i + 1]
    if a == "--out": OUT = Path(args[i + 1])
    if a == "--exclude": exclude = {int(x) for x in args[i + 1].split(",") if x != ""}
assert hands_mode in ("zero", "pico")
print(f"hands_mode={hands_mode}  OUT={OUT}  exclude_original_idx={sorted(exclude)}")

def projected_gravity(q):
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    return np.stack([-(2*(x*z-w*y)), -(2*(y*z+w*x)), -(1-2*(x*x+y*y))], axis=1).astype(np.float32)

def build_state(q):
    j = q[:, 7:36]
    return np.concatenate([j[:,0:6], j[:,6:12], j[:,12:15], j[:,15:22], j[:,22:29],
                           np.zeros((len(q),7),np.float32), np.zeros((len(q),7),np.float32),
                           projected_gravity(q[:,3:7])], axis=1).astype(np.float32)

ep_dirs = []
for t in TASKS:
    for d in sorted(glob.glob(f"{SRC}/{t}/*/")):
        ep_dirs.append((t, Path(d)))

hand_norm = {}
if hands_mode == "pico":
    for task in TASKS:
        arrs = [np.stack(pd.read_parquet(p)["action.hand_cmd_pico"].to_numpy()).astype(np.float32)
                for p in sorted(glob.glob(f"{SRC}/{task}/*/data/chunk-*/file-*.parquet"))]
        A = np.concatenate(arrs)
        std = A.std(0); active = (std > ACTIVE_STD_THRESH)
        mn = A.min(0); mx = A.max(0)
        hand_norm[task] = {"active_dims": [int(i) for i in np.where(active)[0]],
                           "min": mn.tolist(), "max": mx.tolist(),
                           "normalization": "per-dim min-max to [0,1] on active dims; inactive->0"}
    print("hand normalization (active dims per task):",
          {t: hand_norm[t]["active_dims"] for t in TASKS})

def norm_pico(task, pico):
    hn = hand_norm[task]; mn = np.array(hn["min"], np.float32); mx = np.array(hn["max"], np.float32)
    out = np.zeros_like(pico)
    for d in hn["active_dims"]:
        rng = mx[d] - mn[d]
        out[:, d] = np.clip((pico[:, d] - mn[d]) / (rng if rng > 1e-9 else 1.0), 0.0, 1.0)
    return out.astype(np.float32)

if OUT.exists(): shutil.rmtree(OUT)
(OUT/"data"/"chunk-000").mkdir(parents=True)
for vk in VIDEO_KEYS: (OUT/"videos"/"chunk-000"/vk).mkdir(parents=True)
(OUT/"meta").mkdir(parents=True, exist_ok=True)

task_to_idx = {t: i for i, t in enumerate(TASKS)}; task_desc = {}
episodes_jsonl = []; mapping = []; gidx = 0; all_state = []; all_action = []
e = -1
for oi, (task, d) in enumerate(ep_dirs):
    if oi in exclude:
        print(f"  [skip original idx {oi}] {task}/{d.name} (held-out eval)")
        continue
    e += 1
    df = pd.read_parquet(sorted(glob.glob(str(d/"data"/"chunk-*"/"file-*.parquet")))[0]); T = len(df)
    q = np.stack(df["observation.state.robot_q_current"].to_numpy()).astype(np.float32)
    tok = np.stack(df["action.token_state"].to_numpy()).astype(np.float32)
    if hands_mode == "zero":
        hands = np.zeros((T, 14), np.float32)
    else:
        pico = np.stack(df["action.hand_cmd_pico"].to_numpy()).astype(np.float32)
        hands = norm_pico(task, pico)
    action = np.concatenate([tok, hands], axis=1).astype(np.float32)
    state = build_state(q)
    all_state.append(state); all_action.append(action)
    desc = str(pd.read_parquet(d/"meta"/"tasks.parquet")["task"].iloc[0]); task_desc[task_to_idx[task]] = desc
    out_df = pd.DataFrame({
        "observation.state": list(state), "action": list(action),
        "timestamp": (np.arange(T)/FPS).astype(np.float32), "frame_index": np.arange(T, dtype=np.int64),
        "episode_index": np.full(T, e, np.int64), "index": np.arange(gidx, gidx+T, dtype=np.int64),
        "task_index": np.full(T, task_to_idx[task], np.int64)})
    out_df.to_parquet(OUT/"data"/"chunk-000"/f"episode_{e:06d}.parquet")
    for vk in VIDEO_KEYS:
        shutil.copy(sorted(glob.glob(str(d/"videos"/vk/"chunk-*"/"file-*.mp4")))[0],
                    OUT/"videos"/"chunk-000"/vk/f"episode_{e:06d}.mp4")
    episodes_jsonl.append({"episode_index": e, "tasks": [desc], "length": T})
    mapping.append({"episode_index": e, "original_episode_index": oi, "task": task, "session": d.name, "length": T})
    gidx += T
    print(f"  [{e:02d}] {task}/{d.name} T={T}")

with open(OUT/"meta"/"episodes.jsonl","w") as f:
    for r in episodes_jsonl: f.write(json.dumps(r)+"\n")
with open(OUT/"meta"/"tasks.jsonl","w") as f:
    for i in sorted(task_desc): f.write(json.dumps({"task_index": i, "task": task_desc[i]})+"\n")

n_ep = len(mapping)
per_task_counts = {t: sum(1 for m in mapping if m["task"] == t) for t in TASKS}
print("per-task train-episode counts:", per_task_counts)
src_info = json.load(open(sorted(glob.glob(f"{SRC}/{TASKS[0]}/*/meta/info.json"))[0]))
features = {
    "observation.state": {"dtype":"float32","shape":[46],"names":{"axes":[
        *[f"left_leg_{i}" for i in range(6)],*[f"right_leg_{i}" for i in range(6)],"waist_yaw","waist_roll","waist_pitch",
        *[f"left_arm_{i}" for i in range(7)],*[f"right_arm_{i}" for i in range(7)],
        *[f"left_hand_{i}" for i in range(7)],*[f"right_hand_{i}" for i in range(7)],"grav_x","grav_y","grav_z"]}},
    "action": {"dtype":"float32","shape":[78],"names":{"axes":[
        *[f"motion_token_{i}" for i in range(64)],*[f"left_hand_joint_{i}" for i in range(7)],*[f"right_hand_joint_{i}" for i in range(7)]]}},
    "timestamp":{"dtype":"float32","shape":[1]},"frame_index":{"dtype":"int64","shape":[1]},
    "episode_index":{"dtype":"int64","shape":[1]},"index":{"dtype":"int64","shape":[1]},"task_index":{"dtype":"int64","shape":[1]}}
for vk in VIDEO_KEYS: features[vk] = src_info["features"][vk]
info = {"codebase_version":"v2.1","robot_type":"unitree_g1","total_episodes":n_ep,
    "total_frames":int(gidx),"total_tasks":len(TASKS),"total_videos":n_ep*len(VIDEO_KEYS),
    "total_chunks":1,"chunks_size":1000,"fps":FPS,"splits":{"train":f"0:{n_ep}"},
    "data_path":"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path":"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4","features":features}
json.dump(info, open(OUT/"meta"/"info.json","w"), indent=2)

S=np.concatenate(all_state); A=np.concatenate(all_action)
def st(a): return {"mean":a.mean(0).tolist(),"std":(a.std(0)+1e-8).tolist(),"min":a.min(0).tolist(),
                   "max":a.max(0).tolist(),"q01":np.quantile(a,0.01,0).tolist(),"q99":np.quantile(a,0.99,0).tolist()}
json.dump({"observation.state":st(S),"action":st(A)}, open(OUT/"meta"/"stats.json","w"), indent=2)

modality = {
    "state":{"left_leg":{"start":0,"end":6},"right_leg":{"start":6,"end":12},"waist":{"start":12,"end":15},
             "left_arm":{"start":15,"end":22},"right_arm":{"start":22,"end":29},"left_hand":{"start":29,"end":36},
             "right_hand":{"start":36,"end":43},"projected_gravity":{"start":43,"end":46}},
    "action":{"motion_token":{"start":0,"end":64},"left_hand_joints":{"start":64,"end":71},"right_hand_joints":{"start":71,"end":78}},
    "video":{"ego_view":{"original_key":"observation.images.head_cam"},
             "left_wrist_view":{"original_key":"observation.images.left_wrist_cam"},
             "right_wrist_view":{"original_key":"observation.images.right_wrist_cam"}},
    "annotation":{"human.task_description":{"original_key":"task_index"}}}
json.dump(modality, open(OUT/"meta"/"modality.json","w"), indent=2)
thin = {t: c for t, c in per_task_counts.items() if c <= 1}
map_out = {"embodiment_tag":"unitree_g1_sonic","format":"lerobot v2.1 (GR00T N1.7)","hands_mode":hands_mode,
           "ordering":"task-then-session alphabetical","n_episodes":n_ep,
           "excluded_original_indices":sorted(exclude),
           "per_task_episode_counts":per_task_counts,
           "note":("episode_index = index within THIS dataset dir; original_episode_index = index in the "
                   "full 22-ep dataset (g1_encoded_sonic[_handsB]). Excluded indices are the held-out eval "
                   "episodes; the eval harness reads those from the FULL datasets."),
           "episodes":mapping}
if thin:
    map_out["DATA_THIN_WARNING"] = (f"Tasks with <=1 training episode: {thin}. "
        "bottle_cupnoodles_shelf had only 2 total; excluding held-out eval ep leaves 1 training episode "
        "-> expect weak generalization on this task (documented data limitation, not a bug).")
json.dump(map_out, open(OUT/"meta"/"episode_mapping.json","w"), indent=2)

if hands_mode == "pico":
    hm = {"note":"action.left_hand_joints[64:71]<-pico[0:7], right_hand_joints[71:78]<-pico[7:14]. "
                 "Per-task per-dim min-max to [0,1] on active dims; inactive->0. To invert: x = norm*(max-min)+min. "
                 "active_action_dims = dataset action indices (64+pico_dim) that carry real signal (use for loss masking).",
          "tasks":{}}
    for t in TASKS:
        ad = hand_norm[t]["active_dims"]
        hm["tasks"][t] = {"task_index":task_to_idx[t],"pico_active_dims":ad,
                          "active_action_dims":[64+d for d in ad],
                          "min":[hand_norm[t]["min"][d] for d in ad],"max":[hand_norm[t]["max"][d] for d in ad]}
    json.dump(hm, open(OUT/"meta"/"hand_normalization.json","w"), indent=2)

print(f"\nDONE ({hands_mode}). total_frames={int(gidx)}. Dataset at {OUT}")
