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

## Results, r3 (receipts in `paired-ladder/`)

Reference target throughout; LPIPS / DISTS improvement over shipping.

| Arm | Weight | Steps | 48 dev raw | 48 dev 80% | 96 bank raw | 96 bank 80% | 960 holdout 80% |
|---|---:|---:|---|---|---|---|---|
| `ref_w0025_s8k` | 0.0025 | 8,000 | +13.13 / +16.16 | +9.47 / +10.70 | +11.66 / +11.22 Sintel | +12.54 / +7.60 pass | — |
| `ref_w005_s8k` (r2) | 0.005 | 8,000 | +15.61 / +20.00 | +12.22 / +14.13 | +12.05 / +12.01 Sintel | +14.39 / +9.16 pass | +9.29 / +11.79, Sunflower DISTS −11% |
| `ref_w0075_s8k` | 0.0075 | 8,000 | +15.40 / +21.02 | +13.34 / +15.40 | +11.85 / +11.40 Sintel | +14.92 / +9.48 pass | +9.67 / +11.60, Sunflower −7.6 / −16.5 |
| `ref_w010_s8k` | 0.01 | 8,000 | +15.19 / +21.70 | +14.35 / +16.74 | +10.60 / +11.57 Sintel | +15.16 / +10.23 pass | +9.64 / +11.40, RushHour −6.1 LPIPS, Sunflower −14.0 / −22.4 |
| **`ref_w005_s4k`** | 0.005 | **4,000** | +14.79 / +19.10 | +11.49 / +13.91 | +16.22 / +12.60, no Sintel regression | **+15.93 / +9.23 pass** | **+9.48 / +12.65, passes; all eight sources improve** |

The adversarial weight is flat between 0.005 and 0.01 on the development set and each step up costs the grainy holdout sources more. The schedule matters more than the weight: the 4,000-step blend is the first candidate to pass the perceptual gate on every set, with RushHour +13.0% / +24.2% and Sunflower +6.4% / +2.6% on the 960-pair holdout. Its raw checkpoint still over-textures both (RushHour fine energy 1.20), so the 80% blend toward the folded initialization is doing real work; a still-shorter schedule or an intermediate blend was not searched. `ref_w005_s4k` 80% is the native-holdout candidate (`ref4k-native-config-s02.json`, sharpness 0.2 as before).

### Native delivery holdout, 4,000-step blend (`paired-ladder/native-ref4k-s02/`)

Same Release app, frozen inputs and configuration as the 8,000-step blend run (sharpness 0.2, radius 2). Source-balanced LPIPS 0.3288 → 0.2986 (**+9.16%**), DISTS 0.1345 → 0.1156 (**+14.05%**). **Every one of the eight sources improves on both metrics** through the real pipeline, including RushHour (+2.6% / +24.5%) and Sunflower (+2.7% / +11.3%); six sources gain 7–25%. Remaining flags: RushHour fine energy 1.19 (the native detail stage still adds energy on grain; the torch-level blend was at 1.01), and the aggregate correlation floor, which compares native output against a torch-level Lanczos anchor that even native shipping (0.4895) sits below, so it is not informative for native reports. The frozen native gate fails its shipping-relative correlation guard on four sources, as it does for NVIDIA. This is the strongest native result recorded: the earlier best was +7.20% / +10.46% with three sources regressing.

Native graph cost is unchanged from the folded shipping graph (`profile.json`): 5.6 ms at 640×360, 9.9 ms at 864×480 on CPU+GPU.

## Results, r4: input-noise augmentation rejected (receipts in `paired-ladder/ref_noise*`)

| Arm | σ | 48 dev raw | 48 dev 80% | 96 bank 80% | 960 holdout raw | 960 holdout 80% |
|---|---:|---|---|---|---|---|
| `ref_noise010` | 0.01 | +12.69 / +15.53 | +10.25 / +10.12 | +12.46 / +7.59 pass | +8.23 / +7.69, Sunflower −21.6 / −29.1 | +7.42 / +6.82, Sunflower −3.8 / −18.9 |
| `ref_noise020` | 0.02 | +11.72 / +11.43 | +8.85 / +6.71, Johnny DISTS | +10.82 / +5.02 pass | — | — |

Noise on the LR input costs sharpness everywhere (three to four LPIPS points on the development set per step of σ) and does not touch the failure it was meant to fix: the σ 0.01 raw checkpoint still loses 21.6% LPIPS on Sunflower and over-textures RushHour (fine energy 1.21). The grain and bokeh damage is not a denoising problem; the critic is rewarding texture on smooth regions regardless of the input's noise level. Hypothesis rejected; σ 0.04 is recorded for completeness when it completes.

## Ladder r5, prespecified

`C:\lucid\paired-ladder-20260905-r5`: identical to the r3 winner (reference target, 0.005) with one change in the critic, `--paired-negatives shift+smooth`: a quarter of the discriminator's negative mass is now HR with σ 0.03 noise texture painted only where the reference is locally smooth (`paired_dino_adversary.smooth_texture`, 7×7 local fine energy below 0.02). It teaches the critic that invented texture on bokeh and flats is fake, which is the exact failure in the Sunflower and RushHour crops. Arms: 4,000 and 8,000 steps. Same evaluation; the 4,000-step 80% blend is the comparator to beat on the 960-pair holdout (+9.48% / +12.65%, all eight sources up).

## Results, r5: smooth-region negative (receipts `paired-ladder/ref_smooth_s4k*`, `holdout8-s4k*`)

| Arm | 48 dev raw | 48 dev 80% | 96 bank raw | 96 bank 80% | 960 holdout raw | 960 holdout 80% |
|---|---|---|---|---|---|---|
| `ref_smooth_s4k` | +10.95 / +16.95 | +9.06 / +11.98 | +14.51 / +12.50 **pass** | +13.84 / +8.62 pass | +6.95 / +10.26; Sunflower −2.7 / −9.9, RushHour −4.0 LPIPS | +6.30 / +9.93; Sunflower DISTS −2.7 |
| `ref_w005_s4k` (r3, comparator) | +14.79 / +19.10 | +11.49 / +13.91 | +16.22 / +12.60 | +15.93 / +9.23 pass | +10.13 / +13.10; Sunflower −13.6 / −7.6 | **+9.48 / +12.65, all eight up** |

The smooth-texture negative does what it was built to do on the raw checkpoint: Sunflower goes from −13.6% to −2.7% LPIPS and RushHour fine energy from 1.20 to 1.06, and it is the first raw checkpoint to pass the bank set. It pays for that everywhere else (three to four points of aggregate holdout LPIPS/DISTS), and after the 80% blend the plain 4,000-step arm is better on seven of eight sources. Not adopted as is; a lighter weight on the smooth negative (0.05–0.1 instead of 0.125) is the obvious follow-up if grain handling is revisited. `ref_smooth_s8k` and the r6 schedule ladder (2,000 / 6,000 steps, plain negatives) are recorded when they complete.

## Results, r6: the schedule is the knob (receipts `paired-ladder/ref_w005_s2k*`, `ref_w005_s6k*`, `holdout8-s2k*`)

Reference target, weight 0.005, plain shift negatives; only the step count changes.

| Steps | 48 dev raw | 48 dev 80% | 96 bank raw | 96 bank 80% | 960 holdout raw | 960 holdout 80% |
|---:|---|---|---|---|---|---|
| **2,000** | +15.67 / +18.57 | +11.66 / +13.76 | **+18.92 / +11.79 pass** | +15.62 / +9.05 pass | **+10.83 / +14.17, all eight sources up** (Sunflower +1.4 / +0.3) | **+9.41 / +13.04, all eight up; RushHour +15.8 / +25.1, Sunflower +12.9 / +8.7** |
| 4,000 (r3) | +14.79 / +19.10 | +11.49 / +13.91 | +16.22 / +12.60 | +15.93 / +9.23 pass | +10.13 / +13.10, Sunflower −13.6 | +9.48 / +12.65, all eight up (Sunflower +6.4 / +2.6) |
| 6,000 | +15.08 / +19.16 | +11.90 / +13.76 | +12.53 / +12.08 Sintel | +14.86 / +9.06 pass | — | pending |
| 8,000 (r2) | +15.61 / +20.00 | +12.22 / +14.13 | +12.05 / +12.01 Sintel | +14.39 / +9.16 pass | +9.43 / +12.81, Sunflower −25.7 | +9.29 / +11.79, Sunflower −11.0 DISTS |

The development set cannot see this at all: raw LPIPS is +15.1–15.7% at every schedule. The holdout can: the raw checkpoint's Sunflower LPIPS goes +1.4% → −13.6% → −25.7% from 2,000 to 8,000 steps, and RushHour fine energy 1.08 → 1.20 → 1.22. The adversarial term begins inventing texture on bokeh and grain after roughly 2,000 steps, before any aggregate number moves. The 2,000-step 80% blend is the new promotion candidate; its native delivery holdout runs with the same configuration as the 4,000-step one. Prespecified follow-ups: r7 (1,000 and 3,000 steps) and blend weights 0.6 / 0.9 of the 2,000-step raw checkpoint on the holdout.

### Native delivery holdout, 2,000-step blend (`paired-ladder/native-ref2k-s02/`)

Same Release app, frozen inputs and configuration (sharpness 0.2, radius 2). Source-balanced LPIPS 0.3288 → 0.2974 (**+9.54%**), DISTS 0.1345 → 0.1148 (**+14.66%**). Every source improves on both metrics; the smallest gain is DucksTakeOff DISTS +4.9%, and the grainy masters are now among the larger ones: RushHour +9.6% / +27.4%, Sunflower +9.2% / +15.8%. Remaining advisory flags are the same as before (RushHour fine energy 1.15 from the native detail stage; the torch-anchored aggregate correlation floor, which native shipping itself fails). This supersedes the 4,000-step blend as the promotion candidate; `Tools/run-candidate.sh` now launches it by default. The disk filled during this run (the Mac reached 232 MB free); the packet dumps of the earlier regressions were deleted after confirming their scores and decoded PNG receipts were archived.

Ladder r7 (1,000 and 3,000 steps) passes every gate on both development sets, like 2,000: raw +14.78 / +16.41 and +14.87 / +19.15 on the 48 pairs, +17.83 / +9.65 and +17.49 / +12.52 on the bank. Holdout scores for those and for the 0.6 / 0.9 blends of the 2,000-step checkpoint follow.

### Blend weight, 2,000-step checkpoint, 960-pair holdout (`paired-ladder/holdout8-blends*`)

| Blend toward raw | LPIPS | DISTS | Sources up | Gate |
|---:|---:|---:|---|---|
| 0.6 | +7.18% | +9.92% | 8 / 8 | pass |
| 0.8 | +9.41% | +13.04% | 8 / 8 | pass |
| 0.9 | +10.34% | +13.99% | 8 / 8 | pass |
| 1.0 (raw) | +10.83% | +14.17% | 8 / 8 | aggregate correlation 0.001 below the anchor |

At 2,000 steps the blend is nearly monotone in the raw weight and every setting improves every source; the blend is no longer rescuing grain the way it had to at 4,000 and 8,000 steps. 0.8 stays the measured native candidate; 0.9 is the obvious alternative if the live A/B wants a touch more detail. The 6,000-step blend regresses Sunflower DISTS 5.3%, consistent with the schedule finding.

### Ladder r7 on the 960-pair holdout (`paired-ladder/holdout8*` from r7)

| Steps, 80% blend | LPIPS | DISTS | Sources up |
|---:|---:|---:|---|
| 1,000 | +9.10% | +12.13% | 8 / 8 (Sunflower +16.3 / +11.0, RushHour +17.1 / +23.0) |
| **2,000** | **+9.41%** | **+13.04%** | 8 / 8 |
| 3,000 | +8.91% | +12.49% | 8 / 8 (Sunflower +8.7 / +5.9) |
| 4,000 | +9.48% | +12.65% | 8 / 8 (Sunflower +6.4 / +2.6) |

The 3,000-step raw checkpoint already regresses RushHour and Sunflower LPIPS (−4.9%, −9.2%), the 2,000-step raw does not: the drift onto grain starts between 2,000 and 3,000 steps. The 2,000-step blend remains the candidate; 1,000 steps is the safest on grain at a small aggregate cost.

## Results, r8 (receipts `paired-ladder/pre_l1_8k*`, `pre_then_critic_2k*`, `crop128_2k*`)

| Arm | 48 dev raw | 48 dev 80% | 96 bank raw | 96 bank 80% |
|---|---|---|---|---|
| `pre_l1_8k` critic-free reference fine-tune of the folded base | +0.06 / −4.08 | +0.02 / −2.73 | −1.04 / −2.47 | −0.77 / −1.60 |
| `pre_then_critic_2k` (2k critic on top of it) | +14.99 / +18.25 | +10.35 / +12.86 | pending | pending |
| `crop128_2k` (r6 recipe, crop 128) | +15.60 / +18.23 | +11.70 / +13.84 | +18.67 / +11.54 | +15.15 / +8.72 |
| `ref_w005_s2k` (r6, comparator) | +15.67 / +18.57 | +11.66 / +13.76 | +18.92 / +11.79 | +15.62 / +9.05 |

A critic-free L1/Sobel/FFT fine-tune to the reference makes the base *worse* (regression to the mean blurs it), and starting the critic from that base lands where the plain recipe already is. A larger training crop changes nothing measurable. The remaining levers are capacity (r9, ch48 direct 2x from random init) and data (a 756-sequence stream bank from the same 21 sources, three times the current bank, `stream-bank-big`).

## Results, r9: ch48 direct-2x from random init, short base (receipts `paired-ladder/ch48_*`)

Native cost of the wider graph on this Mac (CPU+GPU, random weights): 11.2 ms at 640×360 and 20.3 ms at 864×480, against 5.6 / 9.9 for the shipping ch32 graph; ch64 is 14.8 / 26.2. ch48 fits 30 fps at 480p, ch64 does not comfortably.

| Stage | 48 dev raw | 96 bank raw |
|---|---|---|
| 40,000-step base, batch 8, lr 5e-4, no critic | −6.26 / −3.92 (PSNR 28.54 vs shipping 28.99) | −6.31 / −1.95 |
| + 2,000-step paired critic | +7.52 / +10.32 | +9.54 / +6.33 |

The base is undertrained: 40,000 steps at batch 8 is a fraction of the schedule the SPAN base had, and it sits below the old shipping model on every metric. The critic still lifts it 14 LPIPS points, so the recipe transfers, but this arm says nothing about capacity yet. r11 trains the same architecture for 200,000 steps at batch 16 on the 756-sequence bank before the critic pass; that is the capacity answer. r10 (queued first) tests the larger bank alone with the ch32 recipe.
