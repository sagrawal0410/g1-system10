#!/usr/bin/env python3
"""Produce CURATED, human-readable summary CSVs from the master table.
GitHub renders CSV as sortable tables (no .md needed). Writes:
  results/summary/gate_a_summary.csv   — base vs fine-tuned headline (per policy)
  results/summary/phase_d_summary.csv  — plain vs D1 vs D2 (per policy)
  results/summary/per_task_summary.csv — per-task headline metrics (per policy, FT)
Rounds to sane precision; rows clearly labeled.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

MASTER = sys.argv[1] if len(sys.argv) > 1 else "/home/shaurya/g1_sonic_system1/results/openloop_metrics.csv"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/shaurya/g1_sonic_system1/results/summary")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MASTER)
agg = df[df["scope"] == "AGGREGATE"].copy()

POLICIES = [
    ("GR00T-A (hands-out)", "R1_handsout", "R0_base_handsout"),
    ("GR00T-B (hands-in)", "R1b_handsin", "R0_base_handsin"),
    ("ACT-A (hands-out)", "RA_handsout", "R0_base_ACT_handsout"),
    ("ACT-B (hands-in)", "RA2_handsin", "R0_base_ACT_handsin"),
]

def val(policy_label, config, block, metric):
    r = agg[(agg["policy"] == policy_label) & (agg["config"] == config) & (agg["block"] == block)]
    if not len(r) or metric not in r:
        return np.nan
    return float(r[metric].iloc[0])

def rnd(x, n=4):
    return round(x, n) if x == x and np.isfinite(x) else ""

rows = []
for disp, ft, base in POLICIES:
    lat_base = val(base, "base", "motion_token", "mse")
    lat_ft = val(ft, "plain", "motion_token", "mse")
    rows.append({
        "policy": disp,
        "latent_MSE_base": rnd(lat_base), "latent_MSE_finetuned": rnd(lat_ft),
        "latent_MSE_improvement_x": rnd(lat_base / lat_ft, 1) if (lat_ft and lat_ft == lat_ft and lat_ft != 0) else "",
        "latent_normMSE_base": rnd(val(base, "base", "motion_token", "normalized_mse"), 2),
        "latent_normMSE_finetuned": rnd(val(ft, "plain", "motion_token", "normalized_mse"), 2),
        "latent_finalPosErr_base": rnd(val(base, "base", "motion_token", "final_position_error"), 3),
        "latent_finalPosErr_finetuned": rnd(val(ft, "plain", "motion_token", "final_position_error"), 3),
        "Rhand_graspF1_base": rnd(val(base, "base", "right_hand_joints", "grasp_f1"), 3),
        "Rhand_graspF1_finetuned": rnd(val(ft, "plain", "right_hand_joints", "grasp_f1"), 3),
        "proprio_leakage_gap_finetuned": rnd(val(ft, "plain", "motion_token", "proprio_leakage_gap"), 5),
        "gate_A": "PASS" if (lat_ft == lat_ft and lat_base == lat_base and lat_ft < lat_base) else "?",
    })
pd.DataFrame(rows).to_csv(OUT / "gate_a_summary.csv", index=False)
print("wrote", OUT / "gate_a_summary.csv")

rows = []
for disp, ft, base in POLICIES:
    for config in ["plain", "D1", "D2"]:
        present = len(agg[(agg["policy"] == ft) & (agg["config"] == config) & (agg["block"] == "motion_token")])
        note = ""
        if not present:
            if config == "D2" and ft.startswith("RA"):
                note = "N/A (best-of-N inert on ACT discrete-argmax head; FSQ no-op)"
            else:
                note = "pending" if config != "plain" else ""
        rows.append({
            "policy": disp, "config": config,
            "latent_MSE": rnd(val(ft, config, "motion_token", "mse")),
            "chunk_boundary_discontinuity": rnd(val(ft, config, "motion_token", "chunk_boundary_discontinuity"), 6),
            "latent_MS_jerk": rnd(val(ft, config, "motion_token", "mean_squared_jerk_pred"), 5),
            "latent_on_grid_fraction": rnd(val(ft, config, "motion_token", "on_grid_fraction"), 3),
            "note": note,
        })
pd.DataFrame(rows).to_csv(OUT / "phase_d_summary.csv", index=False)
print("wrote", OUT / "phase_d_summary.csv")

rows = []
TASKS = ["bottle_cupnoodles_shelf", "cup_wipe_sponge_dryingrack", "floor_box_table"]
for disp, ft, base in POLICIES:
    for task in TASKS:
        def tval(policy, config, block, metric):
            r = df[(df["policy"] == policy) & (df["config"] == config) & (df["scope"] == task) & (df["block"] == block)]
            return float(r[metric].iloc[0]) if len(r) and metric in r and len(r[metric]) else np.nan
        rows.append({
            "policy": disp, "task": task,
            "latent_MSE_base": rnd(tval(base, "base", "motion_token", "mse")),
            "latent_MSE_finetuned": rnd(tval(ft, "plain", "motion_token", "mse")),
            "latent_normMSE_finetuned": rnd(tval(ft, "plain", "motion_token", "normalized_mse"), 2),
            "Rhand_MSE_finetuned": rnd(tval(ft, "plain", "right_hand_joints", "mse")),
            "Rhand_graspF1_finetuned": rnd(tval(ft, "plain", "right_hand_joints", "grasp_f1"), 3),
        })
pd.DataFrame(rows).to_csv(OUT / "per_task_summary.csv", index=False)
print("wrote", OUT / "per_task_summary.csv")
print("DONE summary tables ->", OUT)
