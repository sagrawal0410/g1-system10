#!/usr/bin/env python3
"""Summarize results/phase_d/phase_d_metrics.csv into the per-policy
before(I0)->after(I2/I3) Phase-D comparison (mean over held-out eval eps).

Usage:  python -m phase_d.summarize_metrics [csv_path]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else str(
    Path(__file__).resolve().parent.parent.parent / "results" / "phase_d" / "phase_d_metrics.csv")

METRICS = [
    ("latent MSE vs GT",           "motion_token", "mse"),
    ("token msq-jerk(all)",        "all",          "msq_jerk_pred"),
    ("latent boundary-disc",       "motion_token", "boundary_discontinuity"),
    ("decoded-pose msq-jerk",      "decoded_pose", "pose_msq_jerk"),
    ("decoded-pose boundary-disc", "decoded_pose", "pose_boundary_discontinuity"),
    ("latent on-grid frac",        "motion_token", "fsq_ongrid_fraction"),
]
AFTER_EXPS = ["I2_rtc_stitch", "I3_bestof4_rerank"]


def val(df, policy, exp, fsq, block, metric):
    s = df[(df.policy == policy) & (df.experiment == exp) & (df.fsq == fsq)
           & (df.block == block) & (df.metric == metric)]["value"]
    return float(s.mean()) if len(s) else float("nan")


def main():
    df = pd.read_csv(CSV)
    policies = sorted(df.policy.unique())
    print(f"# Phase D OPEN-LOOP before->after comparison  ({CSV})")
    print(f"# policies={policies}  eps={sorted(df.episode.unique())}")
    print("# OPEN-LOOP (predicted-token space). Executed/physics effect deferred to 5090.")
    print("# 'before' = I0 plain (fsq=off).  'after' = I2 stitch / I3 best-of-N (fsq=on).")
    print("# lower is better for all except on-grid frac (higher better).\n")

    for pol in policies:
        has_i3 = len(df[(df.policy == pol) & (df.experiment == "I3_bestof4_rerank")]) > 0
        after_exps = AFTER_EXPS if has_i3 else ["I2_rtc_stitch"]
        note = "" if has_i3 else "   [D1+FSQ only; best-of-N N/A for this policy]"
        print(f"===== {pol}{note} =====")
        rows = []
        for label, block, metric in METRICS:
            row = {"metric": label, "I0(before)": val(df, pol, "I0_baseline", "off", block, metric)}
            for e in after_exps:
                row[e.split("_")[0] + "(after)"] = val(df, pol, e, "on", block, metric)
            rows.append(row)
        tbl = pd.DataFrame(rows).set_index("metric")
        with pd.option_context("display.float_format", lambda v: f"{v:.5g}", "display.width", 160):
            print(tbl.to_string())
        print("  headline (best 'after' vs I0 before):")
        for label, block, metric in METRICS:
            i0 = val(df, pol, "I0_baseline", "off", block, metric)
            afters = [val(df, pol, e, "on", block, metric) for e in after_exps]
            afters = [a for a in afters if not np.isnan(a)]
            if np.isnan(i0) or not afters:
                continue
            if metric == "fsq_ongrid_fraction":
                print(f"    {label:<28s}: {i0:.4g} -> {max(afters):.4g}  (higher better)")
            else:
                best = min(afters)
                fac = (i0 / best) if best > 0 else float("inf")
                print(f"    {label:<28s}: {i0:.4g} -> {best:.4g}  ({fac:.2f}x {'better' if best < i0 else 'worse'})")
        print()


if __name__ == "__main__":
    main()
