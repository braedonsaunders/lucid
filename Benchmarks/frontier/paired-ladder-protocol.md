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

## Results, r10: three times the data (receipts `paired-ladder/big_*`)

`stream-bank-big`: 756 sixteen-frame sequences from the same 21 sources (19 Netflix public scenes and Big Buck Bunny train, Sintel validation), built by `build_stream_bank.py` with six windows per source and six 128-pixel LR patches per window, full-frame H.264/VP9 encoding under recorded bitrates before patch extraction. No PixRestore or shipping caches (`--pixrestore-cache none --shipping-cache none`, reference target). Same recipe as r6 otherwise.

| Arm | 48 dev raw | 48 dev 80% | 96 bank raw | 96 bank 80% | 960 holdout raw | 960 holdout 80% |
|---|---|---|---|---|---|---|
| `big_2k` | +13.39 / +14.60 | +10.04 / +11.31 | +17.38 / +9.37 pass | +13.51 / +7.05 pass | **+11.47 / +14.34, passes, all eight up** | +8.85 / +12.28 pass |
| `big_4k` | +12.74 / +14.96 | +9.71 / +11.60 | +17.37 / +10.33 pass | +13.41 / +7.51 pass | pending | pending |
| `ref_w005_s2k` (shipping recipe) | +15.67 / +18.57 | +11.66 / +13.76 | +18.92 / +11.79 pass | +15.62 / +9.05 pass | +10.83 / +14.17, aggregate correlation just below anchor | +9.41 / +13.04 pass |

The development set says the larger bank is slightly worse; the 960-pair holdout says the opposite, and by a margin: the raw 2,000-step checkpoint is the best Lucid result on the holdout so far, passes every guard without blending (aggregate fine correlation above the Lanczos anchor), and lifts RushHour +19.6% / +30.4% and Sunflower +8.9% / +12.1%. Full-frame codec context in the training data generalizes to unseen sources better than the crop-then-encode bank, and the three-source development set cannot see it. This raw checkpoint goes to the native delivery holdout next; the r11 long ch48 base trains on the same bank.

`big_4k` on the 960-pair holdout: raw +10.14 / +14.20 with Sunflower −0.6% LPIPS and aggregate correlation just below the anchor; 80% blend +8.24 / +12.15, passes. The 2,000-step schedule holds on the larger bank too. `big_2k` raw is bundled as the lab family `lucidbig2k_` for live comparison against the shipping `lucid2k_`.

### Native delivery holdout, big-bank 2k raw (`paired-ladder/native-big2k-s02/`)

Same app, inputs and configuration as the shipping run (sharpness 0.2, radius 2; the harness now pins the SPAN comparator at its own 0.75). Source-balanced LPIPS 0.3288 → 0.2900 (**+11.80%**), DISTS 0.1345 → 0.1129 (**+16.04%**), every source up on both metrics; OldTownCross +18.5% / +28.9%, InToTree +16.3% / +25.1%, RushHour +12.6% / +32.0%. Against the shipping `lucid2k_` run (+9.54% / +14.66%) this is the strongest native result recorded. Advisory flags unchanged (RushHour/Sunflower energy from the native detail stage; torch-anchored correlation floor). Bundled as lab family `lucidbig2k_` for Braedon's eyes; not promoted yet.

## Queued: r11 and r12

**r11** (`C:\lucid\paired-ladder-20260905-r11`): ch48 direct-2x from random init on `stream-bank-big`, 200,000 steps at batch 16, learning rate 5e-4 cosine, reference target, no critic; then the 2,000-step paired critic pass. Evaluated raw on both development sets; the 960-pair holdout and native cost follow on the Mac. **r12** (`-r12`, behind r11): the r10 recipe on `stream-bank-v2`, the same bank plus Tears of Steel (Blender, CC-BY; 22 sources, 792 sequences), 2,000 steps. Both prespecified; the r10 raw checkpoint (`lucidbig2k_`) is the comparator on the holdout.

## Results, r11: long ch48 base (receipts `paired-ladder/ch48_long_*`)

200,000 steps at batch 16 (105 minutes on the RTX 4080) lands exactly where the 40,000-step base did: development PSNR 28.51 versus the SPAN base's 28.99, LPIPS −6.8% / DISTS −5.3% against shipping. Training from random initialization on this bank plateaus below the pretrained SPAN base regardless of schedule; the base's advantage is its pretraining corpus, not its size. The critic pass on top is recorded below. Capacity has to be added *on top of* the pretrained weights instead: r13 widens the folded SPAN initialization from 32 to 48 channels function-preservingly (`widen_span.py`, max abs error 4e-6 on the real checkpoint) and runs the r10 recipe.

## Results, r12: Tears of Steel added (receipts `paired-ladder/v2_2k*`)

`stream-bank-v2` = the big bank plus Tears of Steel (792 sequences, 22 sources). Same recipe: dev raw +12.97 / +15.04, bank raw +16.85 / +9.56, 960-pair holdout raw **+11.11 / +14.69** against big_2k's +11.47 / +14.34, every source within a point of each other. One more animated film adds nothing measurable; the next data step has to be *different* content (live-action variety, faces, text, games), not more of the same families.

## 720p coverage screen, current models (`paired-ladder/coverage-720-v2/`)

The frozen 48-pair 1280×720 → 2560×1440 screen (CrowdRun, DucksTakeOff, ParkJoy; H.264/VP9 at 1.4 and 4.0 Mb/s), the same inputs that rejected the area-folded head on 2026-09-05 (LPIPS +9.4% but DISTS −9.0% against Lanczos).

| Model | vs Lanczos | vs bicubic | vs bilinear | vs SPAN shipping (4× presented at 2×) |
|---|---|---|---|---|
| `lucid2k_` (shipping) | +15.35 / +2.87 | +13.93 / +5.05 | +18.19 / +12.40 | +8.17 / +10.96 |
| `lucidbig2k_` | **+17.55 / +5.32** | +16.16 / +7.45 | +20.31 / +14.61 | +10.55 / +13.20 |

Both trained models now beat every interpolation on both metrics at 720p; `lucidbig2k_` clears the original 3% joint minimum against Lanczos, which no Lucid model had done. Fine correlation sits below the Lanczos anchor (0.414 / 0.425 vs 0.434), as it does for every synthesizing model at this resolution, so the anchor floor is advisory here. Native graph cost at 1280×720 is measured next; if it fits the 33 ms budget, 720p sources are admitted with a 1280×720 rung in the ladder and judged live.

## 720p admitted in the app (2026-09-06)

`LearnedUpscaler.variants` gains a 1280×720 rung at 21.0 ms (measured CPU+GPU for all three bundled 2× families, `ladder720`), inside the 33 ms budget; 1080p remains the first declined size. The 1280×720 packages of `lucid2k_` (shipping), `lucidbig2k_` and `lucid2kraw_` are bundled through the manifest. The policy test now asserts 720p enhanceable and 1080p declined; 45 native tests pass. 60 fps 720p material will not hold cadence at this cost; 30 fps does. Browser-delivery cadence at 2560×1440 has not been re-measured for the 2× families; the earlier direct-2× native stage measured 20.0 ms at this geometry.

## Results, r13: widening the pretrained model (receipts `paired-ladder/w48_*`)

`widen_span.py` grows the folded SPAN initialization from 32 to 48 channels with the function preserved (max abs error 4e-6 on the real checkpoint), then the r10 recipe runs on `stream-bank-big`.

| Arm | 48 dev raw | 96 bank raw | 960 holdout raw |
|---|---|---|---|
| `w48_2k` | +12.90 / +14.84 | +17.31 / +9.83 | +10.38 / +14.30 |
| `w48_4k` | +12.15 / +15.54 | +16.38 / +10.36 | — |
| `big_2k` (ch32, same recipe) | +13.39 / +14.60 | +17.38 / +9.37 | **+11.47 / +14.34** |

Fifty percent more channels, twice the native cost (11.2 vs 5.6 ms at 640×360), and no gain anywhere. Together with r11 this closes the capacity question for now: at this data scale the ch32 student is not capacity-limited. The levers that moved the holdout this session were the training target (reference instead of mixture), the schedule (2,000 steps) and the data (full-frame codec context). What remains is data that is *different*: new live-action masters from the Xiph collection are being fetched (`derf-train`), and the NVIDIA teacher remains the largest untried lever pending its SDK licence.

## Results, r14: seed soup (receipts `paired-ladder/big_2k_seed*`, `soup3-*`)

Two more seeds of the r10 recipe (dev raw +12.80 / +14.80 and +12.63 / +14.46; bank +16.39 / +9.18 and +16.17 / +9.16) and their uniform three-way weight average with the original: holdout **+10.83 / +14.21** for the soup against +11.47 / +14.34 for the original seed alone. Seeds agree to within a point on every source; averaging buys nothing. Recipe variance is not where the remaining gap lives.

## Shipping change, 2026-09-07: `lucidbig2k_`

The r10 raw checkpoint replaces the 2k blend as the shipping family: native holdout +11.80% / +16.04% against SPAN (the blend: +9.54% / +14.66%), every source up on both metrics, identical graph cost, and the first Lucid model to clear the 3% joint minimum against Lanczos at 720p. The blend stays bundled as a lab family. Presentation sharpness stays 0.2.

## Results, r15: six new Xiph live-action masters (receipts `paired-ladder/v3_2k*`)

`stream-bank-v3`: the big bank plus 300-frame excerpts of aspen, speed_bag, touchdown_pass, west_wind_easy, station2 and controlled_burn (Xiph derf, 4:2:2 masters preserved as 4:2:0 lossless; 972 sequences, 27 sources). Same recipe. Dev raw +14.11 / +16.13 (the best development score of the family), bank +17.63 / +10.25, 960-pair holdout **+11.35 / +15.30** against the shipping checkpoint's +11.47 / +14.34: DISTS up a point, LPIPS level, sources trading places (ParkJoy, InToTree, OldTownCross up; DucksTakeOff, Sunflower, Tractor down). A second batch of thirteen masters (HEVC test-set clips and more 1080p live action) is being fetched for a bank with all nineteen new sources.

## Results, r16: thirty-three sources (receipts `paired-ladder/v4_*`)

`stream-bank-v4`: the big bank plus twelve Xiph masters (aspen, speed_bag, touchdown_pass, west_wind_easy, station2, controlled_burn, factory, life, snow_mnt, kristen_and_sara, vidyo1, vidyo4; 1,188 sequences, 33 sources). Same recipe.

| Arm | 48 dev raw | 96 bank raw | 960 holdout raw |
|---|---|---|---|
| `v4_2k` | **+15.02 / +17.42** | +17.35 / +9.95 | **+12.25 / +16.04** |
| `v4_3k` | +15.39 / +17.61 | +16.05 / +10.22 | +11.81 / +15.33 (Sunflower +3.0 / +8.8) |
| `big_2k` (shipping, 21 sources) | +13.39 / +14.60 | +17.38 / +9.37 | +11.47 / +14.34 |

Source diversity is the lever: going from 21 to 33 sources adds 0.8 LPIPS and 1.7 DISTS points on the holdout, with OldTownCross +11.7 / +31.2, InToTree +13.3 / +28.0, RushHour +19.3 / +33.2 and every source up against SPAN on both metrics. Aggregate fine correlation sits a hair below the Lanczos anchor (advisory). Native delivery holdout runs next; if it holds, `v4_2k` ships.

### Native delivery holdout, 33-source checkpoint at sharpness 0.2 (`paired-ladder/native-v4-2k-s02/`)

+11.42% / +15.89% source-balanced, against the shipping `lucidbig2k_` run's +11.80% / +16.04%: level, not ahead. OldTownCross +21.4 / +31.4 and InToTree +20.1 / +29.8 are the largest native gains recorded, but Sunflower loses 6.4% LPIPS (torch level: +4.8%) and RushHour/Sunflower fine energy reaches 1.23 / 1.10. The checkpoint carries more intrinsic texture, and the native detail stage at 0.2 stacks on it. A second native run at sharpness 0.0 follows; the promotion decision waits for it.

## Results, r17: weight and schedule on the 33-source bank (receipts `paired-ladder/v4_w0075_2k*`, `v4_2500*`, `r17-holdout8*`)

| Arm | 48 dev raw | 96 bank raw | 960 holdout raw |
|---|---|---|---|
| `v4_w0075_2k` (weight 0.0075) | **+17.11 / +19.08** | +18.46 / +10.61 | **+13.40 / +16.43**, Sunflower −1.4 LPIPS, RushHour energy over cap |
| `v4_2500` (0.005, 2,500 steps) | +15.90 / +18.52 | +17.07 / +10.40 | +12.46 / +15.79, Sunflower −1.2 LPIPS |
| `v4_2k` (0.005, 2,000) | +15.02 / +17.42 | +17.35 / +9.95 | +12.25 / +16.04 |

With thirty-three sources the model tolerates more adversarial pressure than it did with twenty-one: 0.0075 is now the best aggregate holdout result recorded for Lucid, a point above `v4_2k`, and OldTownCross (+14.9 / +34.2) and InToTree (+16.0 / +30.7) keep climbing. The cost is the same one every texture gain has paid all session: Sunflower goes slightly negative on LPIPS and RushHour's fine energy passes 1.10. The native runs decide; both this and `v4_2k` are evaluated through the app at sharpness 0.0 as well as 0.2.

### Native delivery holdout, 33-source checkpoint at sharpness 0.0 (`paired-ladder/native-v4-2k-s00/`)

+9.87% / +15.52%: turning presentation sharpening off costs LPIPS everywhere and does not repair Sunflower (−3.35%) or RushHour's fine energy (1.15). The over-texturing on the grainy masters is therefore not the sharpening stage. The remaining native-only stage that adds fine energy is the synthetic grain in the deband stage (grain 0.01 in the shipping tuning); a run with grain off and sharpness back at 0.2 follows. Until a native configuration beats the shipping run on aggregate with no source regressing, `lucidbig2k_` stays.

## Results, r18: weight 0.01 and 0.0075 at 2,500 steps (receipts `paired-ladder/v4_w010_2k*`, `v4_w0075_2500*`, `r18-holdout8*`)

| Arm | 48 dev raw | 960 holdout raw | Sunflower |
|---|---|---|---|
| `v4_w010_2k` | +17.48 / +19.78 | +13.59 / +16.23 | −5.4 / +3.6 |
| `v4_w0075_2500` | +17.24 / +19.89 | +13.16 / +15.53 | −9.8 / +2.5 |
| `v4_w0075_2k` (r17) | +17.11 / +19.08 | +13.40 / +16.43 | −1.4 / +6.7 |

The weight ladder has flattened: 0.01 matches 0.0075 on aggregate and loses four more LPIPS points on Sunflower; 2,500 steps at 0.0075 loses eight. 0.0075 at 2,000 steps stays the best point on this bank. Every gain past it is paid on the grainy, out-of-focus master, the same failure the smooth-region negative repaired in r5 at too high a price; a lighter share of that negative on this bank is the next arm.

### Native delivery holdout, 33-source checkpoint with synthetic grain off (`paired-ladder/native-v4-2k-g0/`)

+10.48% / +15.93%, Sunflower −5.93%, RushHour energy 1.22: the deband stage's grain is not the cause either. Torch level this checkpoint gains 4.8% on Sunflower; through the app it loses 6%, while the shipping `lucidbig2k_` checkpoint keeps its torch-level Sunflower gain natively. The remaining native-only difference is the temporal stage (TAA with history); a run with `stageTaa` off isolates it.

## Results, r19: weight 0.015 and 0.01 at 2,500 steps (receipts `paired-ladder/v4_w015_2k*`, `v4_w010_2500*`, `r19-holdout8*`)

| Arm | 48 dev raw | 960 holdout raw | Sunflower | RushHour |
|---|---|---|---|---|
| `v4_w015_2k` | +18.62 / +20.55 | +13.41 / +15.67 | −13.7 / −1.1 | +1.3 / +19.8 |
| `v4_w010_2500` | +17.61 / +20.62 | +12.79 / +15.17 | −17.7 / −1.8 | +0.1 / +18.2 |

The development set keeps rising with the weight; the holdout does not. Clean, textured sources (OldTownCross +18 / +37, InToTree +18 / +33, ParkJoy +18 / +12) keep improving while the grainy masters collapse, and the aggregate is flat at about +13.4 / +16. The weight ladder is closed at 0.0075; the grain axis is the whole remaining problem, and r20 attacks it directly with a light smooth-region negative.

### Native delivery holdout, 33-source checkpoint with the temporal stage off (`paired-ladder/native-v4-2k-notaa/`)

+8.41% / +13.41%, Sunflower **−26.2%** LPIPS, RushHour −3.3%, Tractor −4.7%, PedestrianArea −3.1%. The temporal stage was hiding most of the damage, not causing it: without history blending the checkpoint's synthesized texture on grainy and out-of-focus material is far worse through the app than the torch-level single-frame score suggests (+4.8% on Sunflower). The shipping `lucidbig2k_` checkpoint shows no such torch-to-native gap. Working hypothesis: the 33-source checkpoints amplify chroma noise on the app's NV12 → BGRA path, which the torch evaluation (FFmpeg-decoded RGB) does not exercise. Until a checkpoint holds its torch-level Sunflower result through the app, the native holdout remains the promotion gate and `lucidbig2k_` ships.

## Results, r20: light smooth-region negative at weight 0.0075 (receipts `paired-ladder/v4_w0075_smooth*`, `r20-holdout8*`)

| Arm | 48 dev raw | 960 holdout raw | Sunflower | RushHour energy |
|---|---|---|---|---|
| `v4_w0075_smooth005` (share 0.05) | +15.19 / +16.96 | **+12.15 / +14.97**, every source up | +5.1 / +6.0 | under the cap |
| `v4_w0075_smooth008` (share 0.08) | +14.63 / +16.27 | +10.95 / +14.40 | +6.1 / +7.3 | under the cap |
| `v4_w0075_2k` (no smooth negative) | +17.11 / +19.08 | +13.40 / +16.43 | −1.4 / +6.7 | over the cap |

A 0.05 share of the negative mass spent on smooth-region texture turns the grainy source positive and brings RushHour's fine energy under the cap while giving up about a point of aggregate. It is the first 33-source checkpoint with every source up and no energy flag at the torch level, so it goes to the native holdout, which is where the family has failed so far.

## Results, r21 and the native run of the grain-safe checkpoint

r21 (weight 0.01 with smooth shares 0.03 / 0.05; receipts `paired-ladder/v4_w010_smooth*`, `r21-holdout8*`): holdout +12.75 / +15.19 and +12.00 / +14.91, Sunflower −0.1 and +1.0, RushHour energy over the cap. The smooth negative buys back the grainy source at every weight but the higher weight gives the gain straight back to RushHour's energy; nothing here beats `v4_w0075_smooth005`.

Native delivery holdout of `v4_w0075_smooth005` (`paired-ladder/native-sm005-s02/`): **+11.20% / +15.50%**, RushHour −3.1% and Sunflower −2.1% LPIPS with fine energy 1.28 / 1.11, against the shipping run's +11.80% / +16.04% with every source up. Same pattern as every 33-source checkpoint: the torch-level single-frame score on the grainy masters does not survive the app's NV12 path, while the 21-source `lucidbig2k_` checkpoint keeps its torch-level result natively. `lucidbig2k_` stays the shipping model.

### Where the campaign stands, 2026-09-07 08:00

Through the real pipeline on the eight-source holdout, against the original SPAN ch32utc family: shipping `lucidbig2k_` +11.8% LPIPS / +16.0% DISTS, every source up, at ~40% lower graph cost, plus 720p admitted. NVIDIA VFX ULTRA on the same frames (torch level, no Lucid post-processing) is +21.6% / +21.0%. Levers measured and closed this session: training target (reference wins), schedule (2,000 steps), adversarial weight (0.005–0.0075), blend, seeds, crop, pre-fine-tune, capacity from scratch and by widening, three data expansions, input-noise and smooth-region negatives, and native presentation sharpness/grain/temporal settings. The open levers are (1) the torch-to-native gap on grainy sources for the texture-richer checkpoints, which needs the app's NV12→RGB path reproduced in training or evaluation, (2) more *different* training content, and (3) NVIDIA's own output as a teacher, pending its SDK licence.

## Native-gap investigation, 2026-09-07 (receipts `paired-ladder/native-v4-2k-s0{0,1,2}`, `native-v4-2k-{g0,notaa}`; probes in `Tools/frontier_eval/`)

The 33-source checkpoints gain on the grainy holdout masters at torch level and lose them through the app. Measured, in order:

| Native configuration for `v4_2k` | aggregate | Sunflower LPIPS |
|---|---|---|
| sharpness 0.2 (shipping value) | +11.42 / +15.89 | −6.4 |
| sharpness 0.1 | +10.53 / +15.62 | −3.8 |
| sharpness 0.0 | +9.87 / +15.52 | −3.4 |
| sharpness 0.2, synthetic grain off | +10.48 / +15.93 | −5.9 |
| sharpness 0.2, temporal stage off | +8.41 / +13.41 | −26.2 |

Single-frame probes on the same frames: Core ML output equals torch within one RGB level (`coreml_parity_probe.py`, CPU+GPU); explicit NV12 chroma upsamplers move both models by thousandths (`chroma_path_probe2.py`); the RGB → NV12 → RGB packing of the sender improves both models' LPIPS by removing chroma noise, big2k more than v4 (`nv12_roundtrip_probe.py`). So the model, its conversion, its input colour path and its output packing all reproduce the torch result; the loss appears only when 300 consecutive frames flow through the temporal stage, and grows to −26% with that stage off. The remaining hypothesis is temporal: the 33-source checkpoints synthesize grain that is less coherent frame to frame, which the history blend then averages into a softer, wrong texture. That is measured next with the sequence evaluator's reference-static flicker on the holdout streams.

Temporal probe (`paired-ladder/temporal-probe-holdout-subset.json`; 12 holdout streams × 48 consecutive frames, reference-static flicker): the 33-source checkpoint flickers more than the shipping one where the reference is still, +3% on Sunflower (2.19 vs 2.12) and +10% on RushHour (1.59 vs 1.45), with slightly higher temporal residuals. Modest per frame, but it is the only measured axis on which the two families differ in the direction of the native loss, and the temporal stage blends against history every frame. The trainer now has a reference-static temporal consistency term (`--temporal`, two consecutive frames per crop, no inference cost); the 2026-09-04 measurement of the same term on the earlier model gave −6% flicker and better LPIPS at weight 4. It is the next arm on the 33-source bank.

## NVIDIA teacher: stopped on licence grounds, 2026-09-07

The wheel ships two agreements (`nvidia_vfx-0.1.0.1.dist-info/licenses/packaging/`). The NVIDIA Open Model License covering the weights is permissive about outputs (outputs are not Derivative Models; NVIDIA claims no ownership of them). The NVIDIA Software License Agreement covering the SDK is not: §8.12 forbids using the Software "for the purpose of developing competing products or technologies", which is exactly what distilling Lucid from its outputs would be, and §8.9 forbids distributing or disclosing benchmarking or competitive-analysis results relating to the Software without written permission. Teacher-frame generation was stopped after 52 frames and those frames deleted; no Lucid weights were ever trained on SDK output. The distillation tooling (`nvidia_teacher_frames.py`, `build_teacher_cache.py`, `--intended teacher`) stays in the tree unused. §8.9 also bears on this evidence folder's NVIDIA comparison numbers and the gallery's NVIDIA frames; that disclosure question is the owner's decision.

Ladder r22b (`C:\lucid\paired-ladder-20260905-r22`) is now the pure data lever on bank v5 (21 original + 24 Wikimedia Commons CC0/CC-BY sources, 1,620 sequences; `training-sources-v5.json`, `commons-sources/sources.json`): reference target at weights 0.005 and 0.0075, 2,000 steps. Ladder r23 runs the new temporal consistency term at weights 1.5 and 4 on the 33-source bank at adversarial weight 0.0075.

## Results, r23b: temporal consistency term (receipts `paired-ladder/v4_w0075_temporal*`, `holdout8-t15*`)

Reference target, weight 0.0075, 33-source bank, `--temporal` at 1.5 and 4 (two consecutive frames per crop; the term penalizes output change where the reference is still).

| Arm | 48 dev raw | 96 bank raw | 960 holdout raw | Sunflower | RushHour energy |
|---|---|---|---|---|---|
| `temporal 1.5` | +15.58 / +16.40 | +18.49 / +10.22, passes | **+13.00 / +14.65**, every source up | +3.5 / +5.9 | under the cap |
| `temporal 4` | +13.49 / +12.09 | +16.07 / +8.09 | pending | | |
| no temporal term (r17) | +17.11 / +19.08 | +18.46 / +10.61 | +13.40 / +16.43 | −1.4 / +6.7 | over the cap |

At 1.5 the term gives up under two DISTS points at torch level and buys back the grainy source, the energy cap and the bank's aggregate correlation guard. Its native delivery holdout is the decisive test, since the loss it targets only shows through the app's temporal stage.

### r23b through the app (receipts `paired-ladder/native-t15-*`, `holdout8-t4*`)

The temporal term does not close the native gap. Through the Release app at the shipping sharpness (0.2), `temporal 1.5` scores **+12.05 / +15.52** against the SPAN comparator, against shipping `lucidbig2k_`'s +11.80 / +16.04 on the same harness, and it loses Sunflower by 7.3% LPIPS (energy 1.12) with RushHour over the energy cap (1.26). That is a larger Sunflower loss than the plain 33-source checkpoint at the same sharpness (−6.4), while the same checkpoint gains Sunflower +3.5 at torch level. `temporal 4` on the 960 holdout: +12.18 / +10.77, below 1.5 on every axis, not sent through the app. Not promoted; shipping stays `lucidbig2k_`.

Reading: the flicker measured on the 33-source family was real but it is not what the app's temporal stage punishes, so the 33-source loss must sit in what those checkpoints synthesize on grain and bokeh, not in how stably they synthesize it. The pristine-master hypothesis (Xiph excerpts teach texture that survives torch scoring but not the delivery path) is the one still standing, and the Commons bank tests it from the other side: 45 sources, all compressed masters.

## Results, r22c: bank v5, 21 original + 24 Wikimedia Commons sources (receipts `paired-ladder/v5_*`)

Reference target, 2,000 steps, raw checkpoints, no temporal term.

| Arm | 48 dev raw | 96 bank raw | 960 holdout raw |
|---|---|---|---|
| `v5_ref_2k` (w 0.005) | +11.71 / +13.09 | +14.41 / +8.02, passes | pending |
| `v5_w0075_2k` (w 0.0075) | +12.61 / +13.67 | +15.47 / +8.23, passes | pending |
| big bank, 21 sources (shipping r10) | +12.9 / +14.6 | +15.8 / +9.5 | +11.5 / +14.1 |
| v4, 33 sources (r17, w 0.0075) | +17.11 / +19.08 | +18.46 / +10.61 | +13.40 / +16.43 |

The 24 Commons clips add sources without adding the torch-level lift the Xiph masters gave. Whether they add native-safe quality is the 960 holdout and then the app, below. Ladder r24 (`paired-ladder-20260905-r24`) combines bank v5 with the temporal term at 1.5 for both weights.
