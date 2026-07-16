#!/usr/bin/env python3
"""Phase D3 — derive per-sample RA-BC weights from TOPReward progress.

Input : topreward_progress_video.parquet  (per-frame progress_sparse, keyed by
        VIDEO-dataset episode_index in {1,3,...,21}).
Output: rabc_weights.json  keyed "encoded_ep:frame" (encoded_sonic_train episode
        space 0..18, frame == step_index within episode), plus a distribution
        report to stdout and rabc_weights_report.json.

Weighting reproduces LeRobot's RABCWeights (lerobot/rewards/sarm/rabc.py,
paper Eq. 8-9) exactly:
    delta_t   = progress[min(t+chunk, L-1)] - progress[t]      (per episode)
    mu,sigma  = max(mean(delta),0), max(std(delta), eps)       (global, all frames)
    soft_t    = clip((delta_t - (mu-2 sigma)) / (4 sigma + eps), 0, 1)
    w_t       = 1                       if delta_t > kappa
              = soft_t                  if 0 <= delta_t <= kappa
              = 0                       if delta_t < 0
chunk defaults to the GR00T action_horizon (40) so the weight at step t reflects
progress over exactly the horizon the policy predicts.

VIDEO->ENCODED episode map: both datasets share task-then-session alphabetical
ordering; encoded_train == full minus held-out {0,2,16}. So the ascending list
of video train episodes zips 1:1 with encoded_train 0..18. Verified by matching
per-episode frame counts against meta/episode_mapping.json lengths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HELD_OUT_VIDEO_EPS = {0, 2, 16}

def compute_weights(progress: np.ndarray, chunk: int, mu: float, sigma: float,
                    kappa: float, eps: float) -> np.ndarray:
    L = len(progress)
    fut = np.minimum(np.arange(L) + chunk, L - 1)
    delta = progress[fut] - progress
    lower = mu - 2 * sigma
    soft = np.clip((delta - lower) / (4 * sigma + eps), 0.0, 1.0)
    w = np.zeros(L, dtype=np.float64)
    w[delta > kappa] = 1.0
    mod = (delta >= 0) & (delta <= kappa)
    w[mod] = soft[mod]
    return w, delta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress-parquet", required=True)
    ap.add_argument("--encoded-mapping", required=True,
                    help="meta/episode_mapping.json of g1_encoded_sonic_train")
    ap.add_argument("--out-weights", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--kappa", type=float, default=0.01)
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    df = pd.read_parquet(args.progress_parquet)
    enc = json.loads(Path(args.encoded_mapping).read_text())["episodes"]
    enc_sorted = sorted(enc, key=lambda e: e["episode_index"])

    video_eps = sorted(int(e) for e in df["episode_index"].unique())
    print(f"[map] video train episodes present: {video_eps}")
    assert set(video_eps).isdisjoint(HELD_OUT_VIDEO_EPS), "held-out episode leaked into progress"
    assert len(video_eps) == len(enc_sorted), \
        f"count mismatch: {len(video_eps)} video vs {len(enc_sorted)} encoded"

    prog_by_video = {}
    for vep in video_eps:
        sub = df[df["episode_index"] == vep].sort_values("frame_index")
        prog_by_video[vep] = sub["progress_sparse"].to_numpy(dtype=np.float64)

    vid2enc = {}
    for enc_ep, vep in zip(enc_sorted, video_eps):
        Lv = len(prog_by_video[vep])
        Le = int(enc_ep["length"])
        assert Lv == Le, (f"length mismatch video ep{vep} ({Lv}) vs encoded ep"
                          f"{enc_ep['episode_index']} ({Le}); ordering assumption broken")
        vid2enc[vep] = int(enc_ep["episode_index"])
    print(f"[map] video->encoded episode map: {vid2enc}")

    all_delta = []
    for vep in video_eps:
        p = prog_by_video[vep]
        L = len(p)
        fut = np.minimum(np.arange(L) + args.chunk, L - 1)
        all_delta.append(p[fut] - p)
    all_delta = np.concatenate(all_delta)
    mu = max(float(all_delta.mean()), 0.0)
    sigma = max(float(all_delta.std()), args.eps)
    print(f"[delta] global progress-delta (chunk={args.chunk}): "
          f"mean={all_delta.mean():.4f} std={all_delta.std():.4f} "
          f"-> mu={mu:.4f} sigma={sigma:.4f} "
          f"frac_negative={float((all_delta < 0).mean()):.4f}")

    weights = {}
    per_ep_report = {}
    all_w = []
    for vep in video_eps:
        p = prog_by_video[vep]
        w, delta = compute_weights(p, args.chunk, mu, sigma, args.kappa, args.eps)
        enc_ep = vid2enc[vep]
        for frame in range(len(w)):
            weights[f"{enc_ep}:{frame}"] = float(w[frame])
        all_w.append(w)
        per_ep_report[str(enc_ep)] = {
            "video_ep": vep,
            "num_frames": int(len(w)),
            "weight_mean": float(w.mean()),
            "weight_std": float(w.std()),
            "frac_zero": float((w == 0).mean()),
            "frac_one": float((w == 1.0).mean()),
            "progress_monotonic_frac": float(np.mean(np.diff(p) >= -1e-6)) if len(p) > 1 else 1.0,
        }
    all_w = np.concatenate(all_w)

    report = {
        "chunk": args.chunk, "kappa": args.kappa,
        "delta_mu": mu, "delta_sigma": sigma,
        "delta_frac_negative": float((all_delta < 0).mean()),
        "n_frames_total": int(len(all_w)),
        "weight_mean": float(all_w.mean()),
        "weight_std": float(all_w.std()),
        "weight_min": float(all_w.min()),
        "weight_max": float(all_w.max()),
        "frac_zero": float((all_w == 0).mean()),
        "frac_one": float((all_w == 1.0).mean()),
        "frac_intermediate": float(((all_w > 0) & (all_w < 1.0)).mean()),
        "histogram_bins": np.histogram(all_w, bins=10, range=(0, 1))[0].tolist(),
        "per_episode": per_ep_report,
    }

    Path(args.out_weights).write_text(json.dumps({"weights": weights, "meta": report}, indent=2))
    Path(args.out_report).write_text(json.dumps(report, indent=2))

    print("\n==== RA-BC WEIGHT DISTRIBUTION ====")
    print(f"total frames        : {report['n_frames_total']}")
    print(f"weight mean/std     : {report['weight_mean']:.4f} / {report['weight_std']:.4f}")
    print(f"weight min/max      : {report['weight_min']:.4f} / {report['weight_max']:.4f}")
    print(f"frac weight == 0    : {report['frac_zero']:.4f}")
    print(f"frac weight == 1    : {report['frac_one']:.4f}")
    print(f"frac intermediate   : {report['frac_intermediate']:.4f}")
    print(f"histogram [0..1]/10 : {report['histogram_bins']}")
    print(f"\nwrote weights -> {args.out_weights}")
    print(f"wrote report  -> {args.out_report}")

if __name__ == "__main__":
    main()
