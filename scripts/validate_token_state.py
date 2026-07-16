#!/usr/bin/env python3
"""
Read-only data-quality check: is meta/info.json's `action.token_state` (64-dim)
populated with real, varying latent values, or is it a zero-filled placeholder?

Pure pandas/numpy over already-downloaded parquet files. No GPU, no training,
no MuJoCo, no network calls.
"""
import sys
import glob
import numpy as np
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "/home/shaurya/g1_sonic_system1/data/g1_raw"

parquet_files = sorted(glob.glob(f"{root}/*/*/data/chunk-*/file-*.parquet"))
print(f"Found {len(parquet_files)} episode-session parquet files under {root}")

cols_of_interest = [
    "action.token_state",
    "action.ee_action",
    "action.robot_q_desired",
    "action.planner_cmd",
    "action.hand_cmd",
]

for pf in parquet_files:
    print(f"\n=== {pf} ===")
    try:
        df = pd.read_parquet(pf)
    except Exception as e:
        print(f"  FAILED to read: {e}")
        continue
    print(f"  rows={len(df)} cols={len(df.columns)}")
    for col in cols_of_interest:
        if col not in df.columns:
            print(f"  {col}: NOT PRESENT")
            continue
        arr = np.stack(df[col].to_numpy())
        nonzero_frac = np.mean(np.abs(arr) > 1e-8)
        per_dim_std = arr.std(axis=0)
        n_dims_nonconstant = int(np.sum(per_dim_std > 1e-6))
        print(
            f"  {col}: shape={arr.shape} "
            f"nonzero_frac={nonzero_frac:.4f} "
            f"mean_abs={np.mean(np.abs(arr)):.6f} "
            f"overall_std={arr.std():.6f} "
            f"nonconstant_dims={n_dims_nonconstant}/{arr.shape[1] if arr.ndim>1 else 1} "
            f"min={arr.min():.4f} max={arr.max():.4f}"
        )
