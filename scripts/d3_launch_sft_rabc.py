#!/usr/bin/env python3
"""Phase D3 (T3) — TOPReward RA-BC weighted GR00T SFT continuation.

Thin non-invasive wrapper around Isaac-GR00T's launch_finetune.py (same CLI),
identical in spirit to scripts/launch_sft.py but ADDS per-sample RA-BC loss
weighting. Unlike launch_sft.py's block-loss logging (purely observational),
this DOES change the training objective: each sample's flow-matching MSE is
scaled by a TOPReward-derived RA-BC weight w in [0,1], per-batch normalised to
mean 1 (so LR/scale are preserved when weights are ~uniform).

Mechanism (validated against this repo's data path):
  * ShardedSingleStepDataset.get_shard iterates (ep_idx, step_index) and builds
    each per-step datapoint dict. We attach dp["rabc_weight"] = w(ep_idx, frame)
    as a float32 numpy scalar. The Gr00tN1d7DataCollator np.stack's arbitrary
    keys, so it arrives as inputs["inputs"]["rabc_weight"] of shape (B,).
  * Gr00tTrainer.compute_loss: we POP rabc_weight (so the model never sees it),
    let the original compute_loss run (it returns outputs with the elementwise
    action_loss [B,H,D] and action_mask [B,H,D]), then recompute
        loss = (action_loss * w[:,None,None]).sum() / (action_mask.sum()+eps)
    with w renormalised to mean 1 across the batch.

Weights come from RABC_WEIGHTS_JSON (produced by d3_derive_weights.py), keyed
"ep:frame" in ENCODED-dataset episode space (0-based within episode == step).
Missing keys fall back to RABC_FALLBACK_WEIGHT (default 1.0).

Also forces DDP (no DeepSpeed / nvcc on nodes) exactly like launch_sft.py.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_WEIGHTS_PATH = os.environ["RABC_WEIGHTS_JSON"]
_FALLBACK = float(os.environ.get("RABC_FALLBACK_WEIGHT", "1.0"))

_raw = json.loads(Path(_WEIGHTS_PATH).read_text())
_wmap = _raw["weights"] if "weights" in _raw else _raw
_WEIGHTS: dict[tuple[int, int], float] = {}
for k, v in _wmap.items():
    ep_s, fr_s = k.split(":")
    _WEIGHTS[(int(ep_s), int(fr_s))] = float(v)
logging.info("Loaded %d RA-BC per-frame weights from %s (fallback=%.3f)",
             len(_WEIGHTS), _WEIGHTS_PATH, _FALLBACK)
if _WEIGHTS:
    _vals = np.array(list(_WEIGHTS.values()), dtype=np.float64)
    logging.info("RA-BC weight stats: mean=%.4f std=%.4f min=%.4f max=%.4f frac_zero=%.4f",
                 _vals.mean(), _vals.std(), _vals.min(), _vals.max(), float((_vals == 0).mean()))


def _patch_dataset() -> None:
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

    orig_get_shard = ShardedSingleStepDataset.get_shard

    def get_shard(self, idx: int):
        episodes = self.sharded_episodes[idx]
        datapoints = []
        n_hit = 0
        n_tot = 0
        for ep_idx, step_indices in episodes:
            episode_data = self.episode_loader[ep_idx]
            for step_index in step_indices:
                dp = self.get_datapoint(episode_data, step_index)
                key = (int(ep_idx), int(step_index))
                w = _WEIGHTS.get(key, _FALLBACK)
                n_hit += int(key in _WEIGHTS)
                n_tot += 1
                # float32 0-d numpy scalar -> collator np.stack -> (B,) tensor
                dp["rabc_weight"] = np.asarray(w, dtype=np.float32)
                datapoints.append(dp)
        if n_tot and idx == 0:
            logging.info("RA-BC get_shard[0]: %d/%d steps had explicit weights", n_hit, n_tot)
        return datapoints

    ShardedSingleStepDataset.get_shard = get_shard
    logging.info("Patched ShardedSingleStepDataset.get_shard to attach per-sample RA-BC weights.")


def _patch_trainer() -> None:
    import torch
    from gr00t.experiment.trainer import Gr00tTrainer

    orig_compute_loss = Gr00tTrainer.compute_loss

    # optional per-block logging layout (mirrors launch_sft.py)
    blocks = {"latent": (0, 64), "left_hand": (64, 71), "right_hand": (71, 78)}
    layout_path = os.environ.get("ACTION_LAYOUT_PATH")
    if layout_path and Path(layout_path).exists():
        try:
            d = json.loads(Path(layout_path).read_text())
            sec = d.get("action", {}) if isinstance(d.get("action"), dict) else {}
            bl = sec.get("block_layout") or d.get("block_layout")
            if isinstance(bl, list) and bl:
                parsed = {}
                for it in bl:
                    name = it.get("name") or it.get("modality_key")
                    idx = it.get("indices") or it.get("slice")
                    if name and isinstance(idx, (list, tuple)) and len(idx) == 2:
                        parsed[name] = (int(idx[0]), int(idx[1]))
                if parsed:
                    blocks = parsed
        except Exception:
            logging.exception("Could not parse ACTION_LAYOUT_PATH; using default block layout.")
    logging.info("Per-block loss logging layout: %s", blocks)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1) Pop the RA-BC weight BEFORE the model runs (model must not see it).
        w = None
        try:
            inner = inputs["inputs"] if ("inputs" in inputs) else inputs
            if hasattr(inner, "pop") and "rabc_weight" in inner:
                w = inner.pop("rabc_weight")
        except Exception:
            logging.exception("RA-BC: failed to pop rabc_weight; running UNWEIGHTED this step.")
            w = None

        loss, outputs = orig_compute_loss(
            self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # 2) Recompute the weighted loss from the elementwise action_loss.
        try:
            if (w is not None and isinstance(outputs, dict)
                    and "action_loss" in outputs and "action_mask" in outputs):
                al = outputs["action_loss"]           # (B, H, D), requires grad
                am = outputs["action_mask"]           # (B, H, D)
                wt = w.to(device=al.device, dtype=al.dtype).reshape(-1)
                B = al.shape[0]
                wt = wt * (B / (wt.sum() + 1e-6))     # normalise mean -> 1
                wt = wt.view(B, *([1] * (al.dim() - 1)))
                weighted = (al * wt).sum() / (am.sum() + 1e-6)
                loss = weighted
                self.loss = loss
                if (self.state.global_step % self.args.logging_steps == 0
                        and model.training and self.args.local_rank in (-1, 0)):
                    wv = w.detach().float()
                    self.log({
                        "rabc/mean_weight": float(wv.mean().item()),
                        "rabc/min_weight": float(wv.min().item()),
                        "rabc/frac_zero": float((wv == 0).float().mean().item()),
                    })
        except Exception:
            logging.exception("RA-BC reweighting failed (non-fatal); using unweighted loss.")

        # 3) Per-block loss logging (observational; mirrors launch_sft.py).
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
                for name, (s, e) in blocks.items():
                    bl_ = action_loss[..., s:e]
                    bm_ = action_mask[..., s:e]
                    denom = bm_.sum()
                    val = (bl_.sum() / denom.clamp(min=1e-6)).item() if denom > 0 else float("nan")
                    val_t = torch.tensor(val, device=loss.device)
                    log_dict[f"train_block_loss/{name}"] = self._nested_gather(val_t).mean().item()
                if self.args.local_rank in (-1, 0) and log_dict:
                    self.log(log_dict)
        except Exception:
            logging.exception("Per-block loss logging failed (non-fatal).")

        return (loss, outputs) if return_outputs else loss

    Gr00tTrainer.compute_loss = compute_loss
    logging.info("Patched Gr00tTrainer.compute_loss for RA-BC weighting + block logging.")


def _force_ddp() -> None:
    import gr00t.experiment.experiment as exp
    orig_run = exp.run

    def run(config):
        try:
            config.training.use_ddp = True
            logging.info("Forced config.training.use_ddp=True (DeepSpeed disabled).")
        except Exception:
            logging.exception("Could not set use_ddp=True")
        return orig_run(config)

    exp.run = run


_patch_dataset()
_patch_trainer()

if __name__ == "__main__":
    import runpy
    import gr00t.experiment.launch_finetune as lf

    _force_ddp()
    runpy.run_path(lf.__file__, run_name="__main__")
