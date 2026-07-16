#!/usr/bin/env python3
"""
Phase B (hands-in): extend the already-applied ACT discrete-latent patch with a
DEFAULT-OFF `hand_active_mask` flag that masks the continuous hand-block L1 loss
to each task's ACTIVE action dims (per-task, per-sample).

When hand_active_mask=True (requires discrete_latent=True + hand_mask_table json):
  - a (num_task_index, n_cont) 0/1 mask table is loaded from `hand_mask_table`
    (json {"table": {"<task_index>": [relative cont dims], ...}}), stored as a
    non-persistent buffer so it follows the model to GPU;
  - the per-sample mask = table[batch["task_index"]] is broadcast over the action
    chunk and combined with the pad mask, so the hand-L1 only penalizes each
    task's active hand dims (bottle: right-hand fingers; cup/floor: gripper).

When hand_active_mask=False (default) behaviour is byte-for-byte the prior
discrete-latent ACT (hand-L1 over all cont dims). Idempotent: aborts if the
marker is already present. Precise anchors; aborts if any anchor is missing.
"""
from pathlib import Path

REPO = Path.home() / "g1_sonic_system1" / "repos" / "lerobot" / "src" / "lerobot" / "policies" / "act"
CFG = REPO / "configuration_act.py"
MOD = REPO / "modeling_act.py"
MARKER = "hand_active_mask"


def patch_config(text: str) -> str:
    if MARKER in text:
        raise SystemExit("configuration_act.py already has hand_active_mask. Aborting.")
    anchor = "    latent_grid_scale: float = 16.0\n"
    if anchor not in text:
        raise SystemExit("config anchor (latent_grid_scale) not found. Aborting.")
    addition = anchor + (
        "\n"
        "    # --- Phase B (hands-in): per-task active-dim hand-loss mask (default OFF) ---\n"
        "    hand_active_mask: bool = False\n"
        "    hand_mask_table: str | None = None\n"
    )
    return text.replace(anchor, addition, 1)


def patch_model(text: str) -> str:
    if MARKER in text:
        raise SystemExit("modeling_act.py already has hand_active_mask. Aborting.")

    # (a) ACTPolicy.__init__: load the per-task hand mask table.
    init_anchor = (
        "        super().__init__(config)\n"
        "        config.validate_features()\n"
        "        self.config = config\n"
        "\n"
        "        self.model = ACT(config)\n"
    )
    if init_anchor not in text:
        raise SystemExit("model __init__ anchor (self.model = ACT(config)) not found. Aborting.")
    init_new = init_anchor + (
        "\n"
        "        # --- Phase B (hands-in): per-task active-dim hand-loss mask table ---\n"
        "        if getattr(config, \"hand_active_mask\", False):\n"
        "            if not config.discrete_latent:\n"
        "                raise ValueError(\"hand_active_mask requires discrete_latent=True.\")\n"
        "            if not config.hand_mask_table:\n"
        "                raise ValueError(\"hand_active_mask requires hand_mask_table=<json path>.\")\n"
        "            import json as _json\n"
        "            ls, le = config.latent_block\n"
        "            n_cont = config.action_feature.shape[0] - (le - ls)\n"
        "            with open(config.hand_mask_table) as _f:\n"
        "                _spec = _json.load(_f)\n"
        "            _tbl = _spec[\"table\"]\n"
        "            _max_ti = max(int(k) for k in _tbl)\n"
        "            _m = torch.ones(_max_ti + 1, n_cont)\n"
        "            for _k, _dims in _tbl.items():\n"
        "                _row = torch.zeros(n_cont)\n"
        "                for _d in _dims:\n"
        "                    _row[int(_d)] = 1.0\n"
        "                _m[int(_k)] = _row\n"
        "            self.register_buffer(\"_hand_mask_table\", _m, persistent=False)\n"
        "        else:\n"
        "            self._hand_mask_table = None\n"
    )
    text = text.replace(init_anchor, init_new, 1)

    # (b) ACTPolicy.forward: mask the hand-L1 to active dims when enabled.
    loss_anchor = (
        "            if extras[\"cont\"] is not None:\n"
        "                cont_target = batch[ACTION][..., le:]\n"
        "                cont_abs = F.l1_loss(cont_target, extras[\"cont\"], reduction=\"none\")\n"
        "                cont_den = (valid_mask.sum() * cont_abs.shape[-1]).clamp_min(1)\n"
        "                hand_l1 = (cont_abs * valid_mask).sum() / cont_den\n"
        "                l1_loss = l1_loss + hand_l1\n"
        "                loss_dict[\"hand_l1_loss\"] = hand_l1.item()\n"
    )
    if loss_anchor not in text:
        raise SystemExit("ACTPolicy.forward hand-L1 anchor not found. Aborting.")
    loss_new = (
        "            if extras[\"cont\"] is not None:\n"
        "                cont_target = batch[ACTION][..., le:]\n"
        "                cont_abs = F.l1_loss(cont_target, extras[\"cont\"], reduction=\"none\")\n"
        "                if self._hand_mask_table is not None:\n"
        "                    ti = batch[\"task_index\"].long().reshape(-1)\n"
        "                    hand_mask = self._hand_mask_table[ti]\n"
        "                    while hand_mask.dim() < cont_abs.dim():\n"
        "                        hand_mask = hand_mask.unsqueeze(1)\n"
        "                    eff_mask = valid_mask * hand_mask\n"
        "                    cont_den = eff_mask.sum().clamp_min(1)\n"
        "                    hand_l1 = (cont_abs * eff_mask).sum() / cont_den\n"
        "                else:\n"
        "                    cont_den = (valid_mask.sum() * cont_abs.shape[-1]).clamp_min(1)\n"
        "                    hand_l1 = (cont_abs * valid_mask).sum() / cont_den\n"
        "                l1_loss = l1_loss + hand_l1\n"
        "                loss_dict[\"hand_l1_loss\"] = hand_l1.item()\n"
    )
    text = text.replace(loss_anchor, loss_new, 1)
    return text


def main():
    for path, fn in [(CFG, patch_config), (MOD, patch_model)]:
        src = path.read_text()
        out = fn(src)
        if out == src:
            raise SystemExit(f"No change applied to {path}. Aborting.")
        bak = path.with_suffix(path.suffix + ".handmask_bak")
        if not bak.exists():
            bak.write_text(src)
        path.write_text(out)
        print(f"patched {path}  (backup: {bak})")
    print("OK: ACT hand_active_mask patch applied (default OFF).")


if __name__ == "__main__":
    main()
