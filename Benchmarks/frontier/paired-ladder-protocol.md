# Paired-critic ladder — fixed before results, 2026-09-05

The completed paired-critic run (`paired-critic-protocol.md`) is the first candidate to pass the perceptual promotion gate on both development sets and the frozen gate on the bank set. Its raw checkpoint gains 10.05% LPIPS / 17.98% DISTS on the 48 development pairs; the NVIDIA VFX SDK gains 25.00% / 22.22% on the same pairs. That run used adversarial weight 0.005 and 8,000 steps at learning rate 0.00002, small values chosen when the critic was unproven. Two prespecified ladders test whether the remaining gap is training strength rather than architecture. No inference operation changes; the student remains the folded ch32u 2× graph whose native cost is recorded in `paired-native-profile`.

## Ladder r1 — `C:\lucid\paired-ladder-20260905-r1`

Identical data, caches, initialization, seed, optimizer, crop, batch, objective and paired critic as the completed run. Only these change:

| Arm | Adversarial weight | Steps |
|---|---:|---:|
| `w010_s8k` | 0.01 | 8,000 |
| `w005_s24k` | 0.005 | 24,000 |
| `w010_s24k` | 0.01 | 24,000 |

## Ladder r2 — `C:\lucid\paired-ladder-20260905-r2`

Same as r1 except the pixel-regression target is the HR reference (`--intended reference`, teacher mix 0) instead of the fixed 50% shipping/PixRestore mixture, because the mixture target scores only +2.9% / +6.6% over shipping and the student already exceeds it. Arms: `ref_w005_s8k` (0.005, 8,000) and `ref_w010_s24k` (0.01, 24,000). The queued job waits for r1 to finish and refuses to start while any Python process runs.

## Evaluation, fixed

Every arm evaluates its raw final checkpoint and the predetermined 80% candidate / 20% folded-initialization blend on the frozen 48 development pairs and 96 bank-validation patches, with shipping (4× presented at 2×) and Lanczos controls. 24,000-step arms additionally score raw step 8,000 and 16,000 on the development pairs as trajectory information. Both gates are applied: the frozen development gate (`gate_paired_critic.py`) and the perceptual gate (`gate_perceptual.py`). The 960-pair eight-source holdout is scored for survivors. Passing is admission to the native delivery holdout, not release.

No weight blend other than 0.8, no step other than the final, and no arm is chosen after the fact as the "result" without being reported alongside the others.

## Results, r1 and r2 (receipts in `paired-ladder/`)

Source-balanced LPIPS / DISTS improvement over shipping. "pass" is the recalibrated perceptual gate (`perceptual-gate-protocol.md`); the frozen fine-correlation gate is not applied to ladder arms because it rejects the NVIDIA target.

| Arm | Target | Weight | Steps | 48 dev raw | 48 dev 80% | 96 bank raw | 96 bank 80% |
|---|---|---:|---:|---|---|---|---|
| r2 (completed earlier) | mixture | 0.005 | 8,000 | +10.05 / +17.98 pass | +7.85 / +11.81 pass | +8.98 / +10.89 Sintel LPIPS | +11.10 / +7.60 pass |
| `w010_s8k` | mixture | 0.01 | 8,000 | +12.10 / +21.07 pass | +11.09 / +15.16 pass | +8.20 / +11.63 fail | +13.06 / +9.54 pass |
| `w005_s24k` | mixture | 0.005 | 24,000 | +6.86 / +16.32 pass | +7.95 / +12.71 pass | +7.60 / +11.78 fail | +11.18 / +8.15 pass |
| `w010_s24k` | mixture | 0.01 | 24,000 | +8.93 / +17.06 fail | +10.23 / +14.55 pass | +6.62 / +11.12 fail | +11.12 / +9.89 Sintel |
| `ref_w005_s8k` | **reference** | 0.005 | 8,000 | **+15.61 / +20.00 pass** | **+12.22 / +14.13 pass** | +12.05 / +12.01 Sintel LPIPS | **+14.39 / +9.16 pass** |
| `ref_w010_s24k` | reference | 0.01 | 24,000 | +6.30 / +16.60 fail | +12.98 / +15.41 pass | +8.17 / +9.36 fail | +12.59 / +10.08 Sintel |

Two findings. **Longer schedules hurt**: every 24,000-step raw checkpoint scores below its own 8,000-step snapshot (`ref_w010_s24k` raw at 8k: +13.95 / +20.84; at 24k: +6.30 / +16.60 with LPIPS regressions on FourPeople and Johnny). The adversarial term keeps pushing after the reconstruction term has converged and the output drifts into speckle. **The reference target wins**: at matched weight and steps it adds about five LPIPS points on the development set and three on the bank over the mixture target, and its 80% blend is the best blend on every set.

### 960-pair holdout, torch level (`paired-ladder/merged` reports)

| Candidate | LPIPS | DISTS | Perceptual gate |
|---|---:|---:|---|
| NVIDIA VFX ULTRA (`nvvfx-ultra-holdout960.json`) | **+21.60%** | **+21.02%** | pass (below anchor on 4 sources, aggregate above) |
| `w010_s8k` 80% | +7.92% | +12.25% | Sunflower −4.1% LPIPS / −6.6% DISTS |
| `w005_s24k` 80% | +6.47% | +10.90% | Sunflower regresses |
| paired (r2) 80% | +6.10% | +11.92% | pass |
| paired (r2) raw | +6.27% | +14.92% | RushHour, Sunflower regress; energy 1.10 |

NVIDIA gains most exactly where every Lucid candidate loses: RushHour (+45.9% LPIPS) and Sunflower (+22.1%), the grainy 1080p masters. Our stronger candidates amplify grain and codec noise into speckle (RushHour fine energy 1.09 for the doubled-weight blend); NVIDIA reconstructs clean structure there. That is the remaining gap, not aggregate sharpness.

## Ladders r3 and r4, prespecified

**r3** (`C:\lucid\paired-ladder-20260905-r3`): reference target, 8,000 steps, adversarial weight 0.0025 / 0.0075 / 0.01, plus 0.005 at 4,000 steps. **r4** (`-r4`, queued behind r3): reference target, 0.005, 8,000 steps, with per-sample Gaussian noise of random strength up to sigma 0.01 / 0.02 / 0.04 added to the LR input only (`--input-noise`; targets unchanged), the direct test of the grain hypothesis. Same evaluation for every arm; survivors go to the 960-pair holdout and then the native delivery holdout at a candidate-specific presentation sharpness.

### Reference-target blend on the 960-pair holdout and through the native pipeline

Torch level (`paired-ladder/ref_w005_s8k-holdout8*.json`): the raw checkpoint reaches +9.43% / +12.81% but collapses on the grainy masters (Sunflower −25.7% LPIPS / −23.2% DISTS, RushHour −17.2% LPIPS with fine energy 1.22): it paints blocky texture onto out-of-focus bokeh and film grain (`paired-critic-crops/sunflower-*.png`). The 80% blend is the best Lucid candidate measured so far, **+9.29% LPIPS / +11.79% DISTS**, improving seven of eight sources; Sunflower loses 0.9% LPIPS and 11.0% DISTS.

Native delivery holdout (`paired-ladder/native-ref80-s02/`), same Release app and frozen inputs as the earlier blend, presentation sharpness lowered to 0.2 for this candidate: source-balanced LPIPS 0.3288 → 0.2997 (**+8.84%**), DISTS 0.1345 → 0.1171 (**+12.95%**); six sources improve 7–25% on both metrics; RushHour and Sunflower lose 4.6% / 3.8% LPIPS with fine energy 1.21 / 1.09. Lowering sharpness from 0.4 to 0.2 did not remove the grain-scene regression, so it is the weights, not the presentation, that over-texture grain; that is what ladder r4 tests. Not promoted.

Ladder r3 so far: `ref_w0025_s8k` raw +13.13% / +16.16% dev, `ref_w0075_s8k` raw +15.40% / +21.02% dev, against 0.005's +15.61% / +20.00%: the adversarial weight is flat between 0.005 and 0.0075 and weaker at 0.0025.
