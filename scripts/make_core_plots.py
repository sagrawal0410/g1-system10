#!/usr/bin/env python3
"""CORE summary plots from the master table (results/openloop_metrics.csv).
Reusable: renders whatever policies/blocks are present (GR00T now, +ACT later).
Plots (from AGGREGATE rows, error bars = <metric>__std across held-out eps):
  1. per-block MSE (log) base-vs-FT   2. per-block normalized-MSE
  3. grasp-F1 per hand                4. grasp onset timing (signed)
  5. proprio-leakage gap              6. commanded MS-jerk per block
  7. chunk-boundary discontinuity per block
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "/home/shaurya/g1_sonic_system1/results/openloop_metrics.csv"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/shaurya/g1_sonic_system1/results/openloop_plots/summary")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
agg = df[df["scope"] == "AGGREGATE"].copy()
policies = list(dict.fromkeys(agg["checkpoint"].tolist()))
BLOCKS = [b for b in ["motion_token", "left_hand_joints", "right_hand_joints", "all"] if b in set(agg["block"])]

def _grouped_bar(metric, title, fname, blocks=None, logy=False, std=True):
    blocks = blocks or BLOCKS
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(policies) * len(blocks) / 2), 5))
    x = np.arange(len(blocks))
    w = 0.8 / max(1, len(policies))
    for i, pol in enumerate(policies):
        vals, errs = [], []
        for b in blocks:
            row = agg[(agg["checkpoint"] == pol) & (agg["block"] == b)]
            v = float(row[metric].iloc[0]) if len(row) and metric in row else np.nan
            e = float(row[metric + "__std"].iloc[0]) if std and len(row) and (metric + "__std") in row else 0.0
            vals.append(v)
            errs.append(e if np.isfinite(e) else 0.0)
        ax.bar(x + i * w, vals, w, yerr=errs, capsize=3, label=pol)
    ax.set_xticks(x + w * (len(policies) - 1) / 2)
    ax.set_xticklabels(blocks, rotation=15)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(metric + (" (log)" if logy else ""))
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=120)
    plt.close(fig)
    print("wrote", OUT / fname)

_grouped_bar("mse", "Per-block MSE (AGGREGATE) — base vs fine-tuned", "core1_mse_per_block.png", logy=True)
_grouped_bar("normalized_mse", "Per-block NORMALIZED MSE (÷GT var; 1.0=mean-predictor)", "core2_normalized_mse.png")
_grouped_bar("grasp_f1", "Grasp-event F1 per hand", "core3_grasp_f1.png",
             blocks=[b for b in ["left_hand_joints", "right_hand_joints"] if b in BLOCKS])
_grouped_bar("grasp_onset_timing_error_signed_steps", "Grasp onset timing error (signed frames)",
             "core4_grasp_onset.png", blocks=[b for b in ["left_hand_joints", "right_hand_joints"] if b in BLOCKS], std=False)
_grouped_bar("proprio_leakage_gap", "Proprio-leakage gap (MSE_ablated - MSE); ~0 = no state-copying",
             "core5_proprio_leakage.png")
_grouped_bar("mean_squared_jerk_pred", "Commanded mean-squared jerk per block", "core6_jerk.png", logy=True)
_grouped_bar("chunk_boundary_discontinuity", "Chunk-boundary discontinuity per block", "core7_boundary_discontinuity.png", logy=True)
print("DONE core plots ->", OUT)
