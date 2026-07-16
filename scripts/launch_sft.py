#!/usr/bin/env python3
"""
Thin wrapper around Isaac-GR00T's `gr00t/experiment/launch_finetune.py` that
adds non-invasive per-block (latent / left_hand / right_hand) loss logging
to W&B — required by the runbook for ALL Phase A training runs ("Log
body/left-hand/right-hand losses separately in all runs regardless").

This does NOT change the training loss or gradients in any way. It only
reads the `action_loss` / `action_mask` tensors that the model's action-head
forward already returns (see gr00t/model/gr00t_n1d7/gr00t_n1d7.py, forward():
returns {"loss", "action_loss", "action_mask", ...}, where action_loss is the
element-wise masked MSE *before* the final `.sum()/mask.sum()` reduction),
slices them by block using `results/action_layout.json`, and logs the
per-block masked means as extra W&B scalars via `Trainer.log(...)`. This
mirrors the existing `train_accuracy` logging pattern already in
`Gr00tTrainer.compute_loss` (gr00t/experiment/trainer.py) — same gather-across-
ranks, same `logging_steps` cadence, same rank-0-only log() call.

Deliberately NOT the runbook draft's `block_balanced_flow_loss` ablation
(hand_weight=2.5 re-weighted loss) — that changes the actual training
objective and per the runbook is an OPTIONAL T2 run gated on "a second GPU
group free AND the flow-matching loss is cleanly patchable". We don't have a
spare GPU group (all 4 allocated GPUs go to T1's DDP), so T2 is skipped; this
script only does the always-required *logging* breakdown, which is purely
additive/observational.

Usage: byte-for-byte the same CLI as `gr00t/experiment/launch_finetune.py`
(all the same flags: --base_model_path, --dataset_path, --embodiment_tag,
--num_gpus, --output_dir, --save_steps, --max_steps, --use_wandb, ...) —
this script re-execs that module's own `__main__` logic via `runpy` after
patching, so config-building stays in exact lockstep with upstream instead
of being duplicated (and drifting) here. Launch it exactly the way you'd
launch launch_finetune.py, e.g. via `examples/finetune.sh` with
LAUNCH_FINETUNE_SCRIPT overridden, or directly:

    torchrun --nproc_per_node=$NUM_GPUS \\
        /path/to/launch_sft.py \\
        --base_model_path nvidia/GR00T-N1.7-3B \\
        --dataset_path /path/to/encoded_dataset \\
        --embodiment_tag UNITREE_G1_SONIC \\
        --num_gpus $NUM_GPUS \\
        --output_dir /path/to/output \\
        --save_steps 1000 --max_steps 20000 \\
        --use_wandb --global_batch_size 32 ...

Set ACTION_LAYOUT_PATH env var to override the default
~/g1_sonic_system1/results/action_layout.json location.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ACTION_LAYOUT_PATH = os.environ.get(
    "ACTION_LAYOUT_PATH",
    str(Path.home() / "g1_sonic_system1" / "results" / "action_layout.json"),
)

_FALLBACK_BLOCKS = {"latent": (0, 64), "left_hand": (64, 71), "right_hand": (71, 78)}

def _load_blocks() -> dict[str, tuple[int, int]]:
    import json

    path = Path(ACTION_LAYOUT_PATH)
    if not path.exists():
        logging.warning(
            "action_layout.json not found at %s; per-block loss logging will use the "
            "fallback layout %s. DO NOT trust this for a real launch without checking "
            "the real file once the Phase 0.5 agent produces it.",
            path, _FALLBACK_BLOCKS,
        )
        return dict(_FALLBACK_BLOCKS)

    with open(path) as f:
        d = json.load(f)

    out: dict[str, tuple[int, int]] = {}

    def _from_list(items):
        o = {}
        for item in items:
            if not isinstance(item, dict):
                return {}
            name = item.get("name") or item.get("modality_key")
            idx = item.get("indices") or item.get("slice")
            if name is None:
                return {}
            if isinstance(idx, (list, tuple)) and len(idx) == 2:
                o[name] = (int(idx[0]), int(idx[1]))
            elif "start" in item and "end" in item:
                o[name] = (int(item["start"]), int(item["end"]))
            else:
                return {}
        return o

    def _from_dict(raw):
        o = {}
        for name, spec in raw.items():
            if isinstance(spec, dict) and "start" in spec and "end" in spec:
                o[name] = (int(spec["start"]), int(spec["end"]))
            elif isinstance(spec, dict) and isinstance(spec.get("indices"), (list, tuple)):
                o[name] = (int(spec["indices"][0]), int(spec["indices"][1]))
            elif isinstance(spec, (list, tuple)) and len(spec) == 2:
                o[name] = (int(spec[0]), int(spec[1]))
            else:
                return {}
        return o

    action_sec = d.get("action") if isinstance(d.get("action"), dict) else {}
    candidates = [
        action_sec.get("block_layout"),
        action_sec.get("blocks"),
        d.get("block_layout"),
        d.get("blocks"),
        d.get("block_indices"),
        d.get("action_blocks"),
    ]
    for raw in candidates:
        if isinstance(raw, list) and raw:
            out = _from_list(raw)
        elif isinstance(raw, dict) and raw:
            out = _from_dict(raw)
        if out:
            break

    if out:
        total = sum(e - s for s, e in out.values())
        expected_total = (
            d.get("total_action_dim") or d.get("total_dim") or d.get("action_dim")
            or action_sec.get("total_dim") or action_sec.get("total_action_dim")
        )
        if expected_total is not None and int(expected_total) != total:
            logging.warning(
                "action_layout.json blocks %s cover %d dims but total_dim=%s. "
                "MISMATCH — flag this to the coordinator before trusting these logs.",
                out, total, expected_total,
            )
        return out

    logging.warning(
        "Could not parse a recognizable block schema from %s (top-level keys: %s); "
        "falling back to %s.",
        path, sorted(d.keys()), _FALLBACK_BLOCKS,
    )
    return dict(_FALLBACK_BLOCKS)

_BLOCKS = _load_blocks()
logging.info("Per-block loss logging active with blocks: %s", _BLOCKS)

def _patch_trainer() -> None:
    import torch
    from gr00t.experiment.trainer import Gr00tTrainer

    orig_compute_loss = Gr00tTrainer.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = orig_compute_loss(
            self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        try:
            should_log = (
                self.state.global_step % self.args.logging_steps == 0
                and model.training
                and isinstance(outputs, dict)
                and "action_loss" in outputs
                and "action_mask" in outputs
            )
            if should_log:
                action_loss = outputs["action_loss"].detach()
                action_mask = outputs["action_mask"].detach()
                log_dict = {}
                for name, (s, e) in _BLOCKS.items():
                    block_loss = action_loss[..., s:e]
                    block_mask = action_mask[..., s:e]
                    denom = block_mask.sum()
                    if denom > 0:
                        val = (block_loss.sum() / denom.clamp(min=1e-6)).item()
                    else:
                        val = float("nan")
                    val_t = torch.tensor(val, device=loss.device)
                    val_mean = self._nested_gather(val_t).mean().item()
                    log_dict[f"train_block_loss/{name}"] = val_mean
                if self.args.local_rank in (-1, 0) and log_dict:
                    self.log(log_dict)
        except Exception:
            logging.exception(
                "Per-block loss logging failed (non-fatal — does not affect the actual "
                "training loss/gradients, only this extra W&B logging)."
            )

        return (loss, outputs) if return_outputs else loss

    Gr00tTrainer.compute_loss = compute_loss
    logging.info(
        "Patched Gr00tTrainer.compute_loss for per-block (latent/left_hand/right_hand) "
        "W&B loss logging. Training objective/gradients are unchanged."
    )

def _force_ddp() -> None:
    """Force config.training.use_ddp=True so the HF Trainer uses plain DDP
    instead of DeepSpeed.

    Why: experiment.run() enables DeepSpeed whenever num_gpus>1 and use_ddp is
    False (the default), and DeepSpeed tries to JIT-compile CUDA ops at init,
    which needs a full CUDA toolkit (nvcc + CUDA_HOME). This cluster's compute
    nodes have only the CUDA *runtime* (via the torch cu12 wheels), no toolkit
    / nvcc, so DeepSpeed dies with 'MissingCUDAException: CUDA_HOME does not
    exist'. use_ddp is NOT exposed on launch_finetune's CLI (not a FinetuneConfig
    field), so we flip it on the Config here.

    DDP is mathematically identical training (same AdamW optimizer, same loss,
    same bf16 — bf16=True is already the training-config default); it only
    changes the parallelism backend. A 3B model replica + AdamW states + grads
    + activations fits comfortably on an 80GB H100, so ZeRO/DeepSpeed sharding
    isn't needed at this scale.

    Patch point: launch_finetune.py does `from gr00t.experiment.experiment
    import run` when runpy re-executes it below, so we wrap the module-level
    `experiment.run` BEFORE that re-import happens; the fresh `from ... import
    run` then binds to this wrapper.
    """
    import gr00t.experiment.experiment as exp

    orig_run = exp.run

    def run(config):
        try:
            config.training.use_ddp = True
            logging.info(
                "Forced config.training.use_ddp=True (bypass DeepSpeed: no CUDA "
                "toolkit/nvcc on this node for JIT op build). Plain DDP, bf16=%s, "
                "identical training math.",
                getattr(config.training, "bf16", "?"),
            )
        except Exception:
            logging.exception("Could not set config.training.use_ddp=True")
        return orig_run(config)

    exp.run = run
    logging.info("Patched experiment.run to force DDP (DeepSpeed disabled).")

_patch_trainer()

if __name__ == "__main__":
    import runpy
    import gr00t.experiment.launch_finetune as lf

    _force_ddp()

    runpy.run_path(lf.__file__, run_name="__main__")
