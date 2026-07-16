#!/usr/bin/env python3
"""
Phase B: patch LeRobot's ACT policy to support a discrete-latent (FSQ)
classification head, behind a DEFAULT-OFF `discrete_latent` config flag.

When discrete_latent=True:
  - the first `latent_block=[0,N]` action dims are modeled as per-dim
    classification over `latent_num_classes` FSQ codes (cross-entropy),
    with target code index = round(value*latent_grid_scale) + num_classes//2;
  - the remaining action dims stay continuous (L1);
  - ACTION normalization is forced to IDENTITY so the raw FSQ-grid target
    values survive to index derivation;
  - inference decodes argmax code -> grid value = (idx-half)/scale, concatenated
    with the continuous head -> a plain float action vector (what the SONIC
    decoder ONNX consumes).

When discrete_latent=False (default) behaviour is byte-for-byte the original
ACT (same L1 head, same 2-vs-3-tuple handling is internal only), so other
agents importing lerobot are unaffected.

Idempotent-ish: refuses to patch if markers already present. Precise anchors;
aborts if any anchor is missing (so a repo change can't be silently mis-patched).
"""
import sys
from pathlib import Path

REPO = Path.home() / "g1_sonic_system1" / "repos" / "lerobot" / "src" / "lerobot" / "policies" / "act"
CFG = REPO / "configuration_act.py"
MOD = REPO / "modeling_act.py"

MARKER = "discrete_latent"

def patch_config(text: str) -> str:
    if MARKER in text:
        raise SystemExit("configuration_act.py already patched (marker present). Aborting.")

    anchor = "    # Training and loss computation.\n    dropout: float = 0.1\n    kl_weight: float = 10.0\n"
    if anchor not in text:
        raise SystemExit("config anchor (dropout/kl_weight block) not found. Aborting.")
    addition = (
        "    # Training and loss computation.\n"
        "    dropout: float = 0.1\n"
        "    kl_weight: float = 10.0\n"
        "\n"
        "    # --- Phase B: discrete-latent (FSQ) action head (default OFF) ---\n"
        "    discrete_latent: bool = False\n"
        "    latent_block: list[int] | None = None\n"
        "    latent_num_classes: int = 32\n"
        "    latent_grid_scale: float = 16.0\n"
    )
    text = text.replace(anchor, addition, 1)

    post_anchor = (
        "        if self.n_obs_steps != 1:\n"
        "            raise ValueError(\n"
        "                f\"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`\"\n"
        "            )\n"
    )
    if post_anchor not in text:
        raise SystemExit("config __post_init__ anchor not found. Aborting.")
    post_addition = post_anchor + (
        "\n"
        "        if self.discrete_latent:\n"
        "            if self.latent_block is None or len(self.latent_block) != 2:\n"
        "                raise ValueError(\"discrete_latent requires latent_block=[start, end].\")\n"
        "            if self.latent_block[0] != 0:\n"
        "                raise ValueError(\"discrete_latent currently assumes latent_block starts at 0.\")\n"
        "            # Raw FSQ-grid targets must survive to index derivation -> no action scaling.\n"
        "            self.normalization_mapping = {\n"
        "                **self.normalization_mapping,\n"
        "                \"ACTION\": NormalizationMode.IDENTITY,\n"
        "            }\n"
    )
    text = text.replace(post_anchor, post_addition, 1)
    return text

def patch_model(text: str) -> str:
    if MARKER in text:
        raise SystemExit("modeling_act.py already patched (marker present). Aborting.")

    init_anchor = (
        "        # Final action regression head on the output of the transformer's decoder.\n"
        "        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])\n"
    )
    if init_anchor not in text:
        raise SystemExit("model __init__ action_head anchor not found. Aborting.")
    init_new = (
        "        # Final action head(s) on the output of the transformer's decoder.\n"
        "        action_dim = self.config.action_feature.shape[0]\n"
        "        if config.discrete_latent:\n"
        "            ls, le = config.latent_block\n"
        "            self.n_latent = le - ls\n"
        "            self.n_cont = action_dim - self.n_latent\n"
        "            self.token_head = nn.Linear(config.dim_model, self.n_latent * config.latent_num_classes)\n"
        "            self.cont_head = (\n"
        "                nn.Linear(config.dim_model, self.n_cont) if self.n_cont > 0 else None\n"
        "            )\n"
        "        else:\n"
        "            self.action_head = nn.Linear(config.dim_model, action_dim)\n"
    )
    text = text.replace(init_anchor, init_new, 1)

    fwd_anchor = (
        "        actions = self.action_head(decoder_out)\n"
        "\n"
        "        return actions, (mu, log_sigma_x2)\n"
    )
    if fwd_anchor not in text:
        raise SystemExit("model forward action_head anchor not found. Aborting.")
    fwd_new = (
        "        if self.config.discrete_latent:\n"
        "            b, s, _ = decoder_out.shape\n"
        "            token_logits = self.token_head(decoder_out).view(\n"
        "                b, s, self.n_latent, self.config.latent_num_classes\n"
        "            )\n"
        "            half = self.config.latent_num_classes // 2\n"
        "            idx = token_logits.argmax(dim=-1)\n"
        "            tok_val = (idx - half).to(decoder_out.dtype) / self.config.latent_grid_scale\n"
        "            if self.cont_head is not None:\n"
        "                cont = self.cont_head(decoder_out)\n"
        "                actions = torch.cat([tok_val, cont], dim=-1)\n"
        "            else:\n"
        "                cont = None\n"
        "                actions = tok_val\n"
        "            return actions, (mu, log_sigma_x2), {\"token_logits\": token_logits, \"cont\": cont}\n"
        "\n"
        "        actions = self.action_head(decoder_out)\n"
        "        return actions, (mu, log_sigma_x2), None\n"
    )
    text = text.replace(fwd_anchor, fwd_new, 1)

    loss_anchor = (
        "        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)\n"
        "\n"
        "        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction=\"none\")\n"
        "        valid_mask = ~batch[\"action_is_pad\"].unsqueeze(-1)\n"
        "        num_valid = valid_mask.sum() * abs_err.shape[-1]\n"
        "        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)\n"
        "\n"
        "        loss_dict = {\"l1_loss\": l1_loss.item()}\n"
    )
    if loss_anchor not in text:
        raise SystemExit("ACTPolicy.forward loss anchor not found. Aborting.")
    loss_new = (
        "        actions_hat, (mu_hat, log_sigma_x2_hat), extras = self.model(batch)\n"
        "\n"
        "        valid_mask = ~batch[\"action_is_pad\"].unsqueeze(-1)\n"
        "        if self.config.discrete_latent:\n"
        "            ls, le = self.config.latent_block\n"
        "            half = self.config.latent_num_classes // 2\n"
        "            tok_target = batch[ACTION][..., ls:le]\n"
        "            tok_idx = (\n"
        "                torch.round(tok_target * self.config.latent_grid_scale).long() + half\n"
        "            ).clamp(0, self.config.latent_num_classes - 1)\n"
        "            logits = extras[\"token_logits\"]\n"
        "            ce = F.cross_entropy(\n"
        "                logits.reshape(-1, self.config.latent_num_classes),\n"
        "                tok_idx.reshape(-1),\n"
        "                reduction=\"none\",\n"
        "            ).view(tok_idx.shape)\n"
        "            ce_den = (valid_mask.sum() * ce.shape[-1]).clamp_min(1)\n"
        "            l1_loss = (ce * valid_mask).sum() / ce_den\n"
        "            loss_dict = {\"latent_ce_loss\": l1_loss.item()}\n"
        "            if extras[\"cont\"] is not None:\n"
        "                cont_target = batch[ACTION][..., le:]\n"
        "                cont_abs = F.l1_loss(cont_target, extras[\"cont\"], reduction=\"none\")\n"
        "                cont_den = (valid_mask.sum() * cont_abs.shape[-1]).clamp_min(1)\n"
        "                hand_l1 = (cont_abs * valid_mask).sum() / cont_den\n"
        "                l1_loss = l1_loss + hand_l1\n"
        "                loss_dict[\"hand_l1_loss\"] = hand_l1.item()\n"
        "        else:\n"
        "            abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction=\"none\")\n"
        "            num_valid = valid_mask.sum() * abs_err.shape[-1]\n"
        "            l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)\n"
        "            loss_dict = {\"l1_loss\": l1_loss.item()}\n"
    )
    text = text.replace(loss_anchor, loss_new, 1)

    return text

def main():
    for path, fn in [(CFG, patch_config), (MOD, patch_model)]:
        src = path.read_text()
        out = fn(src)
        if out == src:
            raise SystemExit(f"No change applied to {path} (unexpected). Aborting.")
        bak = path.with_suffix(path.suffix + ".bB_bak")
        if not bak.exists():
            bak.write_text(src)
        path.write_text(out)
        print(f"patched {path}  (backup: {bak})")
    print("OK: ACT discrete-latent head patch applied (default OFF).")

if __name__ == "__main__":
    main()
