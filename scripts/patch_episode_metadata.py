#!/usr/bin/env python3
"""
Metadata-only patch for the WBC data-exporter's LeRobot v3.0 output.

The exporter (built against an older pinned lerobot commit) writes
meta/episodes/*.parquet without a few columns that current lerobot (0.6.x,
what `lerobot[groot]` installs) requires to read v3.0 datasets:
  - data/chunk_index, data/file_index
  - videos/{key}/chunk_index, videos/{key}/file_index
  - videos/{key}/from_timestamp, videos/{key}/to_timestamp
  - meta/episodes/chunk_index, meta/episodes/file_index
  - dataset_from_index, dataset_to_index

Every episode-session directory produced by this exporter contains exactly
ONE chunk (chunk-000) and ONE file (file-000) for both `data/` and each
`videos/<key>/` stream (confirmed by directory listing), so these columns
are trivially reconstructable -- this is NOT a guess at unknown data, it's
filling in indices that are fully determined by the on-disk layout.

This does not touch any data/ or videos/ content, only meta/episodes
parquet files. Idempotent: skips files that already have the columns.
"""
import sys
import glob
from pathlib import Path

import pandas as pd

FPS_DEFAULT = 30

def patch_one(ep_dir: Path):
    ep_parquet_candidates = glob.glob(str(ep_dir / "meta" / "episodes" / "chunk-*" / "file-*.parquet"))
    if not ep_parquet_candidates:
        print(f"  SKIP (no meta/episodes parquet found): {ep_dir}")
        return False
    if len(ep_parquet_candidates) != 1:
        print(f"  WARN: multiple episodes-parquet files in {ep_dir}, patching all")

    video_keys = []
    videos_dir = ep_dir / "videos"
    if videos_dir.exists():
        video_keys = [p.name for p in videos_dir.iterdir() if p.is_dir()]

    import json
    info_path = ep_dir / "meta" / "info.json"
    fps = FPS_DEFAULT
    if info_path.exists():
        with open(info_path) as f:
            fps = json.load(f).get("fps", FPS_DEFAULT)

    changed_any = False
    for ep_parquet in ep_parquet_candidates:
        ep_parquet = Path(ep_parquet)
        df = pd.read_parquet(ep_parquet)
        if "data/chunk_index" in df.columns:
            continue

        length = int(df["length"].iloc[0]) if "length" in df.columns else None

        df["data/chunk_index"] = 0
        df["data/file_index"] = 0
        for key in video_keys:
            df[f"videos/{key}/chunk_index"] = 0
            df[f"videos/{key}/file_index"] = 0
            df[f"videos/{key}/from_timestamp"] = 0.0
            df[f"videos/{key}/to_timestamp"] = (length / fps) if length else 0.0
        df["meta/episodes/chunk_index"] = 0
        df["meta/episodes/file_index"] = 0

        if "episode_index" in df.columns and length is not None:
            df["dataset_from_index"] = 0
            df["dataset_to_index"] = length

        df.to_parquet(ep_parquet)
        changed_any = True
    return changed_any

def main():
    data_root = Path(sys.argv[1]).expanduser().resolve()
    ep_dirs = sorted(
        p.parent for p in data_root.glob("*/*/meta/info.json")
    )
    print(f"Found {len(ep_dirs)} episode-session directories under {data_root.parent if False else data_root}")
    n_patched = 0
    for ep_dir in ep_dirs:

        pass

    ep_dirs = sorted(set(p.parent.parent for p in data_root.glob("*/*/meta/info.json")))
    print(f"Patching {len(ep_dirs)} episode-session directories...")
    for ep_dir in ep_dirs:
        print(f"- {ep_dir}")
        if patch_one(ep_dir):
            n_patched += 1
    print(f"\nDone. Patched {n_patched}/{len(ep_dirs)} directories (others already patched or skipped).")

if __name__ == "__main__":
    main()
