#!/usr/bin/env python3
"""Merge per-policy Phase-D CSVs (results/phase_d/parts/<policy>/phase_d_metrics.csv)
into a single results/phase_d/phase_d_metrics.csv with the `policy` column.

Usage:  python -m phase_d.merge_parts [phase_d_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parent.parent.parent / "results" / "phase_d")


def main():
    parts = sorted((BASE / "parts").glob("*/phase_d_metrics.csv"))
    if not parts:
        print(f"no parts found under {BASE/'parts'}")
        return 1
    dfs = []
    for p in parts:
        df = pd.read_csv(p)
        if "policy" not in df.columns:
            df["policy"] = p.parent.name
        dfs.append(df)
        print(f"  + {p.parent.name}: {len(df)} rows")
    merged = pd.concat(dfs, ignore_index=True)
    out = BASE / "phase_d_metrics.csv"
    merged.to_csv(out, index=False)
    print(f"merged {len(merged)} rows from {len(parts)} policies -> {out}")
    print("policies:", sorted(merged.policy.unique()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
