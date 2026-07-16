#!/usr/bin/env python3
"""Curated summary CSVs + CORE plots from the master open-loop table.
  report.py tables [master.csv] [out_dir]   -> results/summary/{gate_a,phase_d,per_task}_summary.csv
  report.py plots  [master.csv] [out_dir]   -> CORE bar plots (GitHub renders the CSVs as tables)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

POLICIES = [
    ("GR00T-A (hands-out)", "R1_handsout", "R0_base_handsout"),
    ("GR00T-B (hands-in)", "R1b_handsin", "R0_base_handsin"),
    ("ACT-A (hands-out)", "RA_handsout", "R0_base_ACT_handsout"),
    ("ACT-B (hands-in)", "RA2_handsin", "R0_base_ACT_handsin"),
]
TASKS = ["bottle_cupnoodles_shelf", "cup_wipe_sponge_dryingrack", "floor_box_table"]


def _rnd(x, n=4):
    return round(x, n) if x == x and np.isfinite(x) else ""


def cmd_tables(argv):
    master = argv[0] if argv else "/home/shaurya/g1_sonic_system1/results/openloop_metrics.csv"
    out = Path(argv[1] if len(argv) > 1 else "/home/shaurya/g1_sonic_system1/results/summary")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(master)
    agg = df[df["scope"] == "AGGREGATE"].copy()

    def val(policy, config, block, metric):
        r = agg[(agg["policy"] == policy) & (agg["config"] == config) & (agg["block"] == block)]
        return float(r[metric].iloc[0]) if len(r) and metric in r else np.nan

    rows = []
    for disp, ft, base in POLICIES:
        lb = val(base, "base", "motion_token", "mse")
        lf = val(ft, "plain", "motion_token", "mse")
        rows.append({
            "policy": disp,
            "latent_MSE_base": _rnd(lb), "latent_MSE_finetuned": _rnd(lf),
            "latent_MSE_improvement_x": _rnd(lb / lf, 1) if (lf and lf == lf and lf != 0) else "",
            "latent_normMSE_base": _rnd(val(base, "base", "motion_token", "normalized_mse"), 2),
            "latent_normMSE_finetuned": _rnd(val(ft, "plain", "motion_token", "normalized_mse"), 2),
            "latent_finalPosErr_base": _rnd(val(base, "base", "motion_token", "final_position_error"), 3),
            "latent_finalPosErr_finetuned": _rnd(val(ft, "plain", "motion_token", "final_position_error"), 3),
            "Rhand_graspF1_base": _rnd(val(base, "base", "right_hand_joints", "grasp_f1"), 3),
            "Rhand_graspF1_finetuned": _rnd(val(ft, "plain", "right_hand_joints", "grasp_f1"), 3),
            "proprio_leakage_gap_finetuned": _rnd(val(ft, "plain", "motion_token", "proprio_leakage_gap"), 5),
            "gate_A": "PASS" if (lf == lf and lb == lb and lf < lb) else "?",
        })
    pd.DataFrame(rows).to_csv(out / "gate_a_summary.csv", index=False); print("wrote", out / "gate_a_summary.csv")

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
                "latent_MSE": _rnd(val(ft, config, "motion_token", "mse")),
                "chunk_boundary_discontinuity": _rnd(val(ft, config, "motion_token", "chunk_boundary_discontinuity"), 6),
                "latent_MS_jerk": _rnd(val(ft, config, "motion_token", "mean_squared_jerk_pred"), 5),
                "latent_on_grid_fraction": _rnd(val(ft, config, "motion_token", "on_grid_fraction"), 3),
                "note": note,
            })
    pd.DataFrame(rows).to_csv(out / "phase_d_summary.csv", index=False); print("wrote", out / "phase_d_summary.csv")

    rows = []
    for disp, ft, base in POLICIES:
        for task in TASKS:
            def tval(policy, config, block, metric):
                r = df[(df["policy"] == policy) & (df["config"] == config) & (df["scope"] == task) & (df["block"] == block)]
                return float(r[metric].iloc[0]) if len(r) and metric in r and len(r[metric]) else np.nan
            rows.append({
                "policy": disp, "task": task,
                "latent_MSE_base": _rnd(tval(base, "base", "motion_token", "mse")),
                "latent_MSE_finetuned": _rnd(tval(ft, "plain", "motion_token", "mse")),
                "latent_normMSE_finetuned": _rnd(tval(ft, "plain", "motion_token", "normalized_mse"), 2),
                "Rhand_MSE_finetuned": _rnd(tval(ft, "plain", "right_hand_joints", "mse")),
                "Rhand_graspF1_finetuned": _rnd(tval(ft, "plain", "right_hand_joints", "grasp_f1"), 3),
            })
    pd.DataFrame(rows).to_csv(out / "per_task_summary.csv", index=False); print("wrote", out / "per_task_summary.csv")


def cmd_plots(argv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    csv = argv[0] if argv else "/home/shaurya/g1_sonic_system1/results/openloop_metrics.csv"
    out = Path(argv[1] if len(argv) > 1 else "/home/shaurya/g1_sonic_system1/results/openloop_plots/summary")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)
    agg = df[df["scope"] == "AGGREGATE"].copy()
    policies = list(dict.fromkeys(agg["checkpoint"].tolist()))
    BLOCKS = [b for b in ["motion_token", "left_hand_joints", "right_hand_joints", "all"] if b in set(agg["block"])]

    def bar(metric, title, fname, blocks=None, logy=False, std=True):
        blocks = blocks or BLOCKS
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(policies) * len(blocks) / 2), 5))
        x = np.arange(len(blocks)); w = 0.8 / max(1, len(policies))
        for i, pol in enumerate(policies):
            vals, errs = [], []
            for b in blocks:
                row = agg[(agg["checkpoint"] == pol) & (agg["block"] == b)]
                vals.append(float(row[metric].iloc[0]) if len(row) and metric in row else np.nan)
                e = float(row[metric + "__std"].iloc[0]) if std and len(row) and (metric + "__std") in row else 0.0
                errs.append(e if np.isfinite(e) else 0.0)
            ax.bar(x + i * w, vals, w, yerr=errs, capsize=3, label=pol)
        ax.set_xticks(x + w * (len(policies) - 1) / 2); ax.set_xticklabels(blocks, rotation=15)
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(metric + (" (log)" if logy else "")); ax.set_title(title); ax.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(out / fname, dpi=120); plt.close(fig); print("wrote", out / fname)

    hb = [b for b in ["left_hand_joints", "right_hand_joints"] if b in BLOCKS]
    bar("mse", "Per-block MSE (AGGREGATE) - base vs fine-tuned", "core1_mse_per_block.png", logy=True)
    bar("normalized_mse", "Per-block NORMALIZED MSE (div GT var; 1.0=mean-predictor)", "core2_normalized_mse.png")
    bar("grasp_f1", "Grasp-event F1 per hand", "core3_grasp_f1.png", blocks=hb)
    bar("grasp_onset_timing_error_signed_steps", "Grasp onset timing error (signed frames)", "core4_grasp_onset.png", blocks=hb, std=False)
    bar("proprio_leakage_gap", "Proprio-leakage gap (MSE_ablated - MSE); ~0 = no state-copying", "core5_proprio_leakage.png")
    bar("mean_squared_jerk_pred", "Commanded mean-squared jerk per block", "core6_jerk.png", logy=True)
    bar("chunk_boundary_discontinuity", "Chunk-boundary discontinuity per block", "core7_boundary_discontinuity.png", logy=True)


CMDS = {"tables": cmd_tables, "plots": cmd_plots}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print("usage: report.py {tables,plots} [master.csv] [out_dir]"); sys.exit(2)
    CMDS[sys.argv[1]](sys.argv[2:])
