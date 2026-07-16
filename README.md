Results + Metrics Analysis:

The CORE metrics (what they mean)

- MSE per block: raw error of the predicted 64-dim SONIC latent (and 7+7 hands) vs the demonstrated tokens. How close the commanded motion is to the demo.
- Normalized MSE: MSE ÷ ground-truth variance. 1.0 = a mean-predictor (no learning) and <1 = real signal.
- Grasp-event F1: did the hand open/close at the right moments, and how many frames early/late.
- Proprio-leakage gap: (MSE with proprioception zeroed) − (MSE)≈0 means the policy is driven by vision, not by copying the replayed state rules out mode collapse.
- Commanded jerk: smoothness (mean-squared 3rd derivative) of the command stream, lower = smoother on robot.
- Chunk-boundary discontinuity: the jump in the command at 40-step chunk seams, this is exactly what D1 stitching + interpolation removes.
- On-grid fraction: fraction of predicted latent dims that are valid FSQ codes, D2/FSQ forces this to 1.0 so every token is executable by SONIC.

The plot files: results/openloop_plots/summary/

- core1_mse_per_block.png: per-block MSE (log scale), base vs fine-tuned. Shows the raw error collapse.
- core2_normalized_mse.png: normalized MSE, the 1.0 line is the mean-predictor. Shows base sitting above 1.0 (no signal) and fine-tuned dropping below.
- core3_grasp_f1.png: grasp-event F1 per hand (remember only hands-in policies actually carry signal for this).
- core4_grasp_onset.png: signed grasp onset timing error.
- core5_proprio_leakage.png: proprio-leakage gap per block, bars near 0 = vision-driven.
- core6_jerk.png: commanded mean-squared jerk per block.
- core7_boundary_discontinuity.png: chunk-boundary discontinuity per block, the metric D1 slashes.

Curated tables: results/summary/gate_a_summary.csv (base→FT headline), phase_d_summary.csv (plain→inference-time improvements), per_task_summary.csv (bottle/cup/floor).

Results — brief evaluation

Stage A: (base-proxy → plain fine-tuned): all 4 pass decisively.
- Latent MSE drops 8.8×–56.9× (GR00T-MH 0.057→0.0057 and ACT-HI 0.35→0.006).
- Normalized MSE goes from ~10–72 (worse than a mean-predictor) to 0.87–0.99 (real signal) so the policies genuinely learned the token manifold.
- Proprio-leakage gap ≈ 0.0001–0.0004 everywhere meaning the gains are vision-driven and not state-copying.
- Hands-in adds real grasp behavior: GR00T-HI right-hand grasp-F1 0.016 to 0.639.

Inference Time Improvements (plain → D1/D2):
- D1 stitching cuts chunk-boundary discontinuity ~20–85× (GR00T-MH 5.7e-3→6.6e-5 and ACT ~5.5e-3→2.4e-4) and trims latent MSE 10–32% (GR00T-HI 0.0063→0.0043).
- D2 forces on-grid fraction 0.05→1.00 for GR00T: every predicted token becomes a valid, executable SONIC code.
- ACT gets D1 only: its argmax head is already 100% on-grid, and best-of-N doesn't do much (CVAE posterior collapse → identical samples).

Per-task: cup and floor generalize well with a normMSE of around 0.58–0.73 and bottle is the weak task with a normMSE of around 1.37–1.63, because only one training episode survives after the held-out split — a documented data limitation, not a modeling failure.
