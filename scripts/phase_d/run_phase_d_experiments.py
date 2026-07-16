#!/usr/bin/env python3
"""
Phase D — OPEN-LOOP inference ablation on a fine-tuned GR00T checkpoint.

Runs the I0-I3 Phase-D inference stack (receding horizon + RTC stitching +
best-of-N SONIC-aware reranking) on held-out eval episodes, in PREDICTED-TOKEN
space (open-loop: observations always come from the recorded dataset, never from
executing predicted actions). The EXECUTED / physics effect of Phase D is NOT
measured here -- that is the 5090 closed-loop job (eval_on_5090_runbook). Every
row is labeled open_loop=True.

This is the Phase-D-specific ablation (I0-I3 + FSQ on/off + an oracle upper
bound). It deliberately does NOT redo the base-vs-finetuned GATE-A comparison --
that belongs to the Phase-A eval agent that owns scripts/eval_openloop.py. The
obs-assembly + get_action pattern here mirrors that script's tested
run_rollout() (same extract_step_data / parse_observation_gr00t path); metrics
are computed independently and broken out per action block.

Experiments (all on the SAME fine-tuned checkpoint):
  I0  plain independent 40-step chunks (baseline; predict 40, execute 40)
  I1  receding horizon (predict 40, execute 8, no stitch)
  I2  I1 + rtc_style_stitch (body-strong / hand-weak, discrete-latent gate)
  I3  I2 + best-of-4 (candidate_cost + SONIC decoder-roundtrip; per-(step,k) seed)
  each x {fsq_off, fsq_on}   (FSQ manifold-projection ablation)
  I3_oracle_bestK_UPPERBOUND  best-of-4 by closeness to held-out GT -- UPPER
                              BOUND ONLY, never a deployable method.

Metrics per (experiment, fsq, block): vs-GT MSE, mean-squared jerk, boundary
discontinuity at chunk seams, FSQ on-grid fraction (latent), error-vs-horizon.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_phase_d")

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from phase_d.layout import load_layout, ActionLayout, Block
from phase_d.wrappers import RecedingHorizonController, WrapperConfig
from phase_d.stitching import StitchConfig, RecedingHorizonStitcher, rtc_style_stitch
from phase_d.reranker import RerankConfig, candidate_seed, oracle_best_of_k
from phase_d.fsq import fsq_projection_report
from phase_d.sonic_decoder import SonicOnnxDecoder

K = 4
CHUNK_SEED_BASE = 12345

def build_episode_groot(policy, loader, episode_idx, embodiment_tag, layout, steps):
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t
    import torch

    traj = loader[episode_idx]
    actual_steps = min(steps, len(traj))
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = loader.modality_configs["action"].modality_keys
    lang_keys = loader.modality_configs["language"].modality_keys

    modality_no_action = deepcopy(loader.modality_configs)
    modality_no_action.pop("action")

    def extract_cols(df, columns):
        np_dict = {c: np.vstack([np.asarray(a) for a in df[c]]) for c in columns}
        return np.concatenate([np_dict[c] for c in columns], axis=-1)

    gt_action = extract_cols(traj, [f"action.{k}" for k in action_keys])[:actual_steps]
    assert gt_action.shape[1] == layout.total_dim, (gt_action.shape, layout.total_dim)
    state_joints = extract_cols(traj, [f"state.{k}" for k in state_keys])[:actual_steps]

    dp0 = extract_step_data(traj, 0, modality_no_action, embodiment_tag)
    states0 = dict(dp0.states)

    cache: dict[tuple, np.ndarray] = {}

    def _predict(step_count, seed, frozen):

        eff_seed = candidate_seed(int(step_count), 0, CHUNK_SEED_BASE, K) if seed is None else int(seed)
        key = (int(step_count), eff_seed, frozen)
        if key in cache:
            return cache[key]
        torch.manual_seed(eff_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eff_seed)
        data_point = extract_step_data(traj, step_count, modality_no_action, embodiment_tag)
        obs = {}
        states = states0 if frozen else data_point.states
        for k, v in states.items():
            obs[f"state.{k}"] = v
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)
        for lk in lang_keys:
            obs[lk] = data_point.text
        parsed = parse_observation_gr00t(obs, loader.modality_configs)
        raw, _ = policy.get_action(parsed)
        parts = []
        for k in action_keys:
            v = np.asarray(raw[k])[0]
            if v.ndim == 1:
                v = v[:, None]
            parts.append(v)
        chunk = np.concatenate(parts, axis=-1).astype(np.float64)
        assert chunk.shape[1] == layout.total_dim, (chunk.shape, layout.total_dim)
        cache[key] = chunk
        return chunk

    def predict_chunk(step_count, seed=None):
        return _predict(step_count, seed, False)

    def predict_chunk_ablated(step_count, seed=None):
        return _predict(step_count, seed, True)

    return gt_action, actual_steps, predict_chunk, predict_chunk_ablated, state_joints

class ActRunner:
    def __init__(self, ckpt, dataset_path, layout, device="cuda:0", ablate_dims=None):
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.torch = torch
        pm = Path(ckpt)
        if (pm / "pretrained_model").exists():
            pm = pm / "pretrained_model"
        self.pm = pm
        self.device = device
        self.layout = layout
        log.info("Loading ACT policy from %s", pm)
        self.policy = ACTPolicy.from_pretrained(str(pm)).to(device).eval()
        self.cfg = self.policy.config
        self.latent_dim = int(self.cfg.latent_dim)
        self.image_keys = list(self.cfg.image_features)
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.cfg, pretrained_path=str(pm),
            preprocessor_overrides={"device_processor": {"device": str(device)}})

        self.ds = LeRobotDataset(repo_id="local/act_eval", root=dataset_path, video_backend="pyav")
        try:
            self._efrom = self.ds.episode_data_index["from"]
            self._eto = self.ds.episode_data_index["to"]
        except Exception:
            self._efrom = self._eto = None

        self.ablate_dims = ablate_dims

        self.bestn_supported = False

    def _ep_range(self, ep):
        if self._efrom is not None:
            return int(self._efrom[ep]), int(self._eto[ep])

        lengths = [self.ds.meta.episodes[i]["length"] for i in range(self.ds.num_episodes)]
        start = sum(lengths[:ep])
        return start, start + lengths[ep]

    _latent_patch_fires = 0

    def _sampled_latent_ctx(self, seed):
        torch = self.torch
        latent_dim = self.latent_dim
        orig = torch.zeros
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        runner = self

        class _Ctx:
            def __enter__(self_):
                def patched(*a, **kw):
                    shp = a[0] if a else kw.get("size")
                    if shp is None and len(a) >= 2 and all(isinstance(x, int) for x in a[:2]):
                        shp = list(a)
                    if isinstance(shp, (list, tuple)) and len(shp) == 2 and int(shp[1]) == latent_dim:
                        runner._latent_patch_fires += 1
                        b = int(shp[0])
                        return torch.randn(b, latent_dim, generator=gen).to(dtype=kw.get("dtype", torch.float32))
                    return orig(*a, **kw)
                torch.zeros = patched
            def __exit__(self_, *e):
                torch.zeros = orig
        return _Ctx()

    def build_episode(self, ep, steps):
        torch = self.torch
        efrom, eto = self._ep_range(ep)
        n = min(steps, eto - efrom)
        gt = np.stack([np.asarray(self.ds[efrom + t]["action"], dtype=np.float64) for t in range(n)])
        assert gt.shape[1] == self.layout.total_dim, (gt.shape, self.layout.total_dim)
        state_joints = np.stack([np.asarray(self.ds[efrom + t]["observation.state"], dtype=np.float64)
                                 for t in range(n)])
        state0 = self.ds[efrom]["observation.state"].clone()
        pdims = self.ablate_dims if self.ablate_dims is not None else int(state0.shape[0])
        cache = {}

        def _predict(step_count, seed, frozen):

            key = (int(step_count), None if seed is None else int(seed), frozen)
            if key in cache:
                return cache[key]
            item = self.ds[efrom + min(int(step_count), n - 1)]
            st = item["observation.state"].clone()
            if frozen:
                st[:pdims] = state0[:pdims]
            obs = {"observation.state": st.unsqueeze(0)}
            for k in self.image_keys:
                obs[k] = item[k].unsqueeze(0)
            obs = self.pre(obs)
            if seed is None:
                chunk = self.policy.predict_action_chunk(obs)
            else:
                with self._sampled_latent_ctx(seed):
                    chunk = self.policy.predict_action_chunk(obs)
            chunk = np.asarray(chunk[0].float().cpu().numpy(), dtype=np.float64)
            assert chunk.shape[1] == self.layout.total_dim, chunk.shape
            cache[key] = chunk
            return chunk

        if not getattr(self, "_diag_done", False):
            self._diag_done = True
            fires0 = self._latent_patch_fires
            c_zero = _predict(0, None, False)
            c_s1 = _predict(0, candidate_seed(0, 1, CHUNK_SEED_BASE, K), False)
            fired = self._latent_patch_fires - fires0
            d_all = float(np.abs(c_zero - c_s1).mean())
            log.info("[ACT best-of-N diag] latent_patch_fired=%d zero-vs-samp |Δall|=%.6f "
                     "(0 => posterior-collapsed => best-of-N inert => ACT after-D = D1)", fired, d_all)

        def predict_chunk(step_count, seed=None):
            return _predict(step_count, seed, False)

        def predict_chunk_ablated(step_count, seed=None):
            return _predict(step_count, seed, True)

        return gt, n, predict_chunk, predict_chunk_ablated, state_joints

def run_config(predict_chunk, layout, gt_action, actual_steps, *, execute, stitch_cfg,
               use_bestN, decoder, rerank_cfg, oracle=False):
    D = layout.total_dim
    C = layout.chunk_length
    stream = np.zeros((actual_steps, D), dtype=np.float64)

    if not oracle:
        cfg = WrapperConfig(mode="receding_horizon", execute=execute, stitch=stitch_cfg,
                            use_best_of_n=use_bestN, rerank=rerank_cfg)
        ctrl = RecedingHorizonController(predict_chunk, layout, cfg, decoder=decoder)
        ctrl.reset()
        t = 0
        replan = 0
        while t < actual_steps:
            res = ctrl.step(obs=t, step_index=t)
            take = min(execute, actual_steps - t)
            stream[t:t + take] = res.actions[:take]
            t += take
            replan += 1
        return stream

    stitcher = RecedingHorizonStitcher(layout, stitch_cfg, execute=execute, decoder=decoder)
    stitcher.reset()
    t = 0
    while t < actual_steps:
        cands = [predict_chunk(t, candidate_seed(t, k, CHUNK_SEED_BASE, K)) for k in range(K)]
        take = min(execute, actual_steps - t)
        gt_seg = gt_action[t:t + take]
        sel = oracle_best_of_k([c[:take] for c in cands], gt_seg)
        chunk = cands[sel["best_index"]]
        res = stitcher.step(chunk)
        stream[t:t + take] = res.actions[:take]
        t += take
    return stream

def msq_diff(x, n):
    if x.shape[0] <= n:
        return float("nan")
    return float(np.mean(np.diff(x, n=n, axis=0) ** 2))

def on_grid_fraction(x, step=0.0625, atol=1e-6):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.mean(np.abs(x / step - np.round(x / step)) < atol))

def boundary_discontinuity(pred, execute, block_slice):
    """Mean squared jump across chunk seams (t = execute, 2*execute, ...)."""
    jumps = []
    for t in range(execute, pred.shape[0], execute):
        jumps.append(np.mean((pred[t, block_slice] - pred[t - 1, block_slice]) ** 2))
    return float(np.mean(jumps)) if jumps else float("nan")

def error_vs_horizon(pred, gt, period):
    buckets = [[] for _ in range(period)]
    for t in range(pred.shape[0]):
        buckets[t % period].append(np.mean((pred[t] - gt[t]) ** 2))
    return [float(np.mean(b)) if b else float("nan") for b in buckets]

def compute_rows(stream, gt, layout, execute, base, decoder=None):
    rows = []
    blocks = list(layout.blocks) + [Block("all", 0, layout.total_dim, None)]
    for b in blocks:
        sl = b.slice
        p, g = stream[:, sl], gt[:, sl]
        rows.append({**base, "block": b.name, "metric": "mse", "horizon_idx": -1,
                     "value": float(np.mean((p - g) ** 2))})
        rows.append({**base, "block": b.name, "metric": "mse_gt_scale", "horizon_idx": -1,
                     "value": float(np.mean(g ** 2))})
        rows.append({**base, "block": b.name, "metric": "msq_jerk_pred", "horizon_idx": -1,
                     "value": msq_diff(p, 3)})
        rows.append({**base, "block": b.name, "metric": "msq_jerk_gt", "horizon_idx": -1,
                     "value": msq_diff(g, 3)})
        rows.append({**base, "block": b.name, "metric": "msq_vel_pred", "horizon_idx": -1,
                     "value": msq_diff(p, 1)})
        rows.append({**base, "block": b.name, "metric": "boundary_discontinuity", "horizon_idx": -1,
                     "value": boundary_discontinuity(stream, execute, sl)})
        if b.is_latent:
            rows.append({**base, "block": b.name, "metric": "fsq_ongrid_fraction", "horizon_idx": -1,
                         "value": on_grid_fraction(p)})

    for hi, v in enumerate(error_vs_horizon(stream, gt, execute)):
        rows.append({**base, "block": "all", "metric": "mse_at_horizon_idx", "horizon_idx": hi, "value": v})

    if decoder is not None:
        lb = layout.latent_blocks[0].slice
        decoder.reset()
        pose = np.asarray(decoder.decode_chunk(stream[:, lb]), dtype=np.float64)
        rows.append({**base, "block": "decoded_pose", "metric": "pose_msq_jerk", "horizon_idx": -1,
                     "value": msq_diff(pose, 3)})
        rows.append({**base, "block": "decoded_pose", "metric": "pose_msq_vel", "horizon_idx": -1,
                     "value": msq_diff(pose, 1)})
        rows.append({**base, "block": "decoded_pose", "metric": "pose_boundary_discontinuity",
                     "horizon_idx": -1, "value": boundary_discontinuity(pose, execute, slice(0, pose.shape[1]))})
    return rows

def make_plot(streams, gt, layout, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lb = layout.latent_blocks[0].slice
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    ax = axes[0]
    ax.plot(np.linalg.norm(gt[:, lb], axis=-1), color="k", lw=2, label="GT")
    for name, s in streams.items():
        ax.plot(np.linalg.norm(s[:, lb], axis=-1), lw=1, alpha=0.8, label=name, linestyle="--")
    ax.set_title("latent block L2 norm over time (open-loop; NOT decoded xyz)")
    ax.set_ylabel("|latent|_2"); ax.legend(fontsize=7, ncol=3)

    ax = axes[1]
    for name, s in streams.items():
        d = np.linalg.norm(np.diff(s[:, lb], axis=0), axis=-1)
        ax.plot(d, lw=1, alpha=0.8, label=name)
    ax.set_title("per-step latent change |Δ| (lower/smoother = better; seam spikes)")
    ax.set_xlabel("step"); ax.set_ylabel("|Δ latent|"); ax.legend(fontsize=7, ncol=3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint-step", default="unknown")
    p.add_argument("--policy", default="GR00T-A", help="policy label for the CSV 'policy' column")
    p.add_argument("--policy-type", default="groot", choices=["groot", "act"])
    p.add_argument("--dataset-path", default="/lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic")
    p.add_argument("--eval-split", default=str(SCRIPTS_DIR.parent / "results" / "eval_split.json"))
    p.add_argument("--action-layout", default=str(SCRIPTS_DIR.parent / "results" / "action_layout.json"))
    p.add_argument("--embodiment-tag", default="UNITREE_G1_SONIC")
    p.add_argument("--episodes", default="0,2,16")
    p.add_argument("--steps", type=int, default=320)
    p.add_argument("--denoising-steps", type=int, default=4)
    p.add_argument("--decoder-onnx",
                   default="/lambdafs/shaurya/g1_sonic_system1/repos/GR00T-WholeBodyControl/"
                           "gear_sonic_deploy/policy/release/model_decoder.onnx")
    p.add_argument("--output-dir", default=str(SCRIPTS_DIR.parent / "results" / "phase_d"))
    p.add_argument("--npz-dir", default="/lambdafs/shaurya/g1_sonic_system1/results/eval_npz",
                   help="eval-agent NPZ contract dir")
    p.add_argument("--npz-policy", default=None,
                   help="exact POLICY label for NPZ filenames (R1_handsout/R1b_handsin/RA_handsout/RA2_handsin)")
    p.add_argument("--ablate-proprio-dims", type=int, default=-1,
                   help="# leading state dims to freeze at t0 for the ablated rollout; -1 = whole state "
                        "(ACT-hands-in: 36 to keep the trailing task-id one-hot)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--isaac-gr00t-repo",
                   default=os.environ.get("ISAAC_GR00T_REPO",
                                          str(Path.home() / "g1_sonic_system1" / "repos" / "Isaac-GR00T")))
    args = p.parse_args()

    sys.path.insert(0, args.isaac_gr00t_repo)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"

    layout = load_layout(args.action_layout)
    log.info("layout: total=%d chunk=%d blocks=%s latent_continuous=%s",
             layout.total_dim, layout.chunk_length, [b.name for b in layout.blocks], layout.latent_continuous)

    with open(args.eval_split) as f:
        split = json.load(f)
    ep_records = {int(e["episode_index"]): e for e in split["eval"]}
    want_eps = [int(x) for x in args.episodes.split(",")]

    decoder = None
    if Path(args.decoder_onnx).exists():
        try:
            import onnxruntime
            decoder = SonicOnnxDecoder(onnx_path=args.decoder_onnx, mode="fixed_history")
            log.info("SONIC decoder loaded (fixed_history) for I3 roundtrip term.")
        except Exception as e:
            log.warning("onnxruntime/decoder unavailable (%s); I3 roundtrip term disabled.", e)
    else:
        log.warning("decoder ONNX not found at %s; I3 roundtrip term disabled.", args.decoder_onnx)

    bestn_supported = True
    if args.policy_type == "groot":
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag) if hasattr(EmbodimentTag, "resolve") \
            else EmbodimentTag(args.embodiment_tag)
        log.info("Loading GR00T policy from %s", args.checkpoint)
        policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=args.checkpoint, device=args.device)
        policy.model.action_head.num_inference_timesteps = args.denoising_steps
        loader = LeRobotEpisodeLoader(dataset_path=args.dataset_path, modality_configs=policy.get_modality_config())
        episode_builder = lambda ep: build_episode_groot(policy, loader, ep, embodiment_tag, layout, args.steps)
    elif args.policy_type == "act":
        act = ActRunner(args.checkpoint, args.dataset_path, layout, device=args.device,
                        ablate_dims=(None if args.ablate_proprio_dims < 0 else args.ablate_proprio_dims))
        bestn_supported = act.bestn_supported
        episode_builder = lambda ep: act.build_episode(ep, args.steps)
    else:
        raise ValueError(args.policy_type)

    def stitch(freeze, blend, fsq_on):
        return StitchConfig(freeze=freeze, blend_len=blend, latent_strategy="newest_only",
                            hand_alpha_power=0.5, snap_latent_to_grid=fsq_on)

    rerank_cfg = RerankConfig(K=K, base_seed=CHUNK_SEED_BASE,
                              body_smooth_scale=1.0, hand_smooth_scale=0.2,
                              w_roundtrip=1.0 if decoder is not None else 0.0)

    CONFIG_SPECS = [

        ("D1", 8, 3, 20, False, False),
        ("D2", 8, 3, 20, True, True),
    ]
    npz_policy = args.npz_policy or args.policy
    npz_dir = Path(args.npz_dir); npz_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = {"policy": args.policy, "npz_policy": npz_policy, "policy_type": args.policy_type,
               "checkpoint": args.checkpoint, "checkpoint_step": args.checkpoint_step,
               "dataset_path": args.dataset_path, "embodiment_tag": args.embodiment_tag,
               "steps_cap": args.steps, "denoising_steps": args.denoising_steps, "K": K,
               "best_of_n_supported": bestn_supported, "ablate_proprio_dims": args.ablate_proprio_dims,
               "decoder_roundtrip_enabled": decoder is not None, "decoder_mode": "fixed_history",
               "open_loop": True, "execution_horizon_meta": 40,
               "physics_validation": "DEFERRED to 5090 closed-loop (eval_on_5090_runbook)",
               "configs_emitted": [], "episodes": {}, "npz_files": []}
    if not bestn_supported:
        run_cfg["best_of_n_note"] = (
            "D2 (best-of-N) SKIPPED for this policy: the patched discrete-head ACT is "
            "CVAE posterior-collapsed -> sampling the prior latent leaves the argmax output "
            "unchanged (|Δ|=0), so best-of-N is inert and FSQ is a no-op (ACT is natively "
            "on-grid). 'after Phase D' for ACT = D1 stitching. Only D1 NPZ emitted.")

    for ep in want_eps:
        if ep not in ep_records:
            log.warning("episode %d not in eval_split; skipping", ep); continue
        task = ep_records[ep].get("task", "?")
        log.info("=== episode %d (task=%s) ===", ep, task)
        gt_action, actual_steps, predict_chunk, predict_chunk_ablated, state_joints = episode_builder(ep)
        run_cfg["episodes"][str(ep)] = {"task": task, "actual_steps": int(actual_steps)}

        for (cfg, execute, freeze, blend, use_bestN, fsq_on) in CONFIG_SPECS:
            if use_bestN and not bestn_supported:
                continue
            dec = decoder if use_bestN else None
            sc = stitch(freeze, blend, fsq_on)
            pred = run_config(predict_chunk, layout, gt_action, actual_steps,
                              execute=execute, stitch_cfg=sc, use_bestN=use_bestN,
                              decoder=dec, rerank_cfg=rerank_cfg, oracle=False)
            pred_abl = run_config(predict_chunk_ablated, layout, gt_action, actual_steps,
                                  execute=execute, stitch_cfg=sc, use_bestN=use_bestN,
                                  decoder=dec, rerank_cfg=rerank_cfg, oracle=False)
            meta = dict(checkpoint=f"{npz_policy}@{cfg}", task=task, episode_id=str(ep),
                        plot_rank=0, execution_horizon=40)
            fname = f"{npz_policy}@{cfg}__{task}__ep{ep}.npz"
            np.savez(
                npz_dir / fname,
                pred_action=pred.astype(np.float32),
                gt_action=gt_action.astype(np.float32),
                pred_action_ablated=pred_abl.astype(np.float32),
                state_joints=state_joints.astype(np.float32),
                meta=np.array(meta, dtype=object),
            )

            ong = on_grid_fraction(pred[:, layout.latent_blocks[0].slice])
            abl_delta = float(np.abs(pred - pred_abl).mean())
            log.info("  wrote %s  T=%d  mse=%.5f jerk=%.5f ongrid=%.3f  |Δablated|=%.5f",
                     fname, pred.shape[0], float(np.mean((pred - gt_action) ** 2)),
                     msq_diff(pred, 3), ong, abl_delta)
            run_cfg["configs_emitted"].append(cfg)
            run_cfg["npz_files"].append(fname)

    run_cfg["configs_emitted"] = sorted(set(run_cfg["configs_emitted"]))
    with open(out_dir / f"run_config_{npz_policy}.json", "w") as f:
        json.dump(run_cfg, f, indent=2)
    log.info("Wrote %d NPZ files to %s (run_config_%s.json in %s)",
             len(run_cfg["npz_files"]), npz_dir, npz_policy, out_dir)

if __name__ == "__main__":
    main()
