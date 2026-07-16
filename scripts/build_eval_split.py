#!/usr/bin/env python3
"""
Build results/eval_split.json: hold out ~10% of episodes per task, fixed seed.

Operates on the "episode-session directories" produced by the WBC data
exporter (each dir == one LeRobotDataset with total_episodes usually == 1).
Excludes any episode-session directory that is missing required files
(videos/, meta/stats.json, meta/tasks.parquet) -- i.e. incomplete downloads.

With very small per-task episode counts (this run: partial dataset, only
2 and 5 raw session dirs per task), a literal 10% rounds to 0. Policy used
here: hold out max(1, round(0.10 * n)) episodes per task so every task has
at least one eval episode, and note this deviation explicitly in the output.
"""
import json
import random
import sys
from pathlib import Path

SEED = 42

def is_complete(ep_dir: Path) -> bool:
    videos_ok = (ep_dir / "videos").exists() and any((ep_dir / "videos").iterdir())
    stats_ok = (ep_dir / "meta" / "stats.json").exists()
    tasks_ok = (ep_dir / "meta" / "tasks.parquet").exists()
    return videos_ok and stats_ok and tasks_ok

def main():
    data_root = Path(sys.argv[1]).expanduser().resolve()
    out_path = Path(sys.argv[2]).expanduser().resolve()

    rng = random.Random(SEED)
    result = {
        "seed": SEED,
        "policy": "hold out max(1, round(0.10 * n_complete_episodes)) per task; "
        "incomplete episode-session dirs (missing videos/stats.json/tasks.parquet) excluded entirely",
        "generated_from_partial_dataset": True,
        "tasks": {},
        "excluded_incomplete": [],
    }

    task_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    for task_dir in task_dirs:
        ep_dirs = sorted(p for p in task_dir.iterdir() if p.is_dir() and (p / "meta" / "info.json").exists())
        complete, incomplete = [], []
        for ep in ep_dirs:
            (complete if is_complete(ep) else incomplete).append(ep)

        for ep in incomplete:
            result["excluded_incomplete"].append(f"{task_dir.name}/{ep.name}")

        names = sorted(p.name for p in complete)
        n = len(names)
        n_eval = max(1, round(0.10 * n)) if n > 0 else 0
        shuffled = names[:]
        rng.shuffle(shuffled)
        eval_ids = sorted(shuffled[:n_eval])
        train_ids = sorted(shuffled[n_eval:])

        result["tasks"][task_dir.name] = {
            "total_complete_episodes": n,
            "n_eval": n_eval,
            "eval_episode_ids": eval_ids,
            "train_episode_ids": train_ids,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
