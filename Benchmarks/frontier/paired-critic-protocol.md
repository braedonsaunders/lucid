# Paired critic quality experiment — 2026-09-05

The measured reference-detail student gains 9.57% LPIPS / 14.80% DISTS on 48 development pairs but loses fine correlation on all three sources. Attenuating its adversarial gradient removes most perceptual improvement. The pinned August 2026 PixRestore critic classifies output DINO tokens without the input image. This motivates testing whether **conditioning realism feedback on input content and explicitly rejecting misplaced detail** produces a better tradeoff. It is a hypothesis, not proof of the previous failure's cause.

The candidate concatenates output and aligned bicubic-LR DINO tokens at six layers. Each upstream critic head has 768 inputs and 96 hidden units (control: 384 and 96). Half the discriminator objective is true HR, one quarter generated output, and one quarter HR whose signed 3×3 highpass is displaced two pixels horizontally with replicated boundaries. Coarse content stays in place. The generator still seeks the real label with weight 0.005. Conditions and negative examples are detached. There are no additional inference operations or trainable inference parameters.

This is a combined conditioning/negative-example recipe, not an isolated causal attribution to either component, and not a reproduction of a diffusion paper. It adapts the released [August 2026 PixRestore critic](https://github.com/csslc/PixRestore). Contemporary [DNF-SR's CVPR 2026 supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Han_DNF-SR_Dual-Input_and_CVPR_2026_supplemental.pdf) instead uses reward-weighted positive/negative velocity directions; that algorithm is not implemented here.

Freeze the prior reference-detail training configuration: composed 29-source bank (26 training IDs), verified shipping and PixRestore caches, folded shipping initialization, full reconstruction graph, 8,000 steps, batch 4, crop 96, seed 20260914, AdamW 0.00002 with cosine decay, reference-target Sobel/FFT objectives. Preserve BF16 training and the prior numerical execution policy; do not claim bitwise deterministic CUDA training. Smoke the default path and compare its discriminator initialization and first batch with the recorded reference-detail control before reusing that completed control. Smoke the paired path before the full run. Check source/data/checkpoint hashes remotely and preserve other GPU jobs.

Evaluate only the final raw checkpoint and the previously fixed 80% candidate / 20% initialization weights on the frozen 48 full-frame and 96 bank-validation pairs. No coefficient, negative displacement, checkpoint, or critic-capacity search follows this result. Require at least 3% source-balanced LPIPS and DISTS gain against shipping, no more than 2% per-source perceptual regression, and no more than 0.01 fine-correlation loss per source. Compare changes with the completed reference-detail control as well. A pass is admission to native and fresh source-disjoint evaluation, not world-class proof; the competitive target remains the much larger gains of the measured commercial reference. The repeatedly inspected evaluation sets are development/regression evidence, not untouched release holdouts.

No shipping weights change, no flicker tuning, no remote publication. Training and evaluation use the authorized RTX 4080 over Tailscale.

The first launch stopped before candidate training because the launcher incorrectly required a first-output hash from the historical receipt, which does not contain that field. Both recorded historical hashes matched. The corrected launch retains the new control/candidate first-output comparison; this was a receipt-schema error, not evidence of a numerical mismatch. Preserve the failed launch status.

## Completed result (recovered after the interrupting session)

The r2 job finished all 8,000 steps in 11.87 minutes on the RTX 4080; unit, smoke, training, interpolation and both evaluations exited zero. Receipts: `paired-critic-experiment.json`, `paired-critic-development.json`, `paired-critic-bank-validation.json`, the frozen gate receipts `paired-critic-*-gate.json` and the perceptual gate receipts `paired-critic-*-perceptual-gate.json` (see `perceptual-gate-protocol.md`).

| Variant | Set | LPIPS | DISTS | Frozen gate | Perceptual gate |
|---|---|---:|---:|---|---|
| raw | 48 development | +10.05% | +17.98% | fails FourPeople, Johnny correlation | passes |
| 80% blend | 48 development | +7.85% | +11.81% | fails Johnny correlation (−0.0127) | passes |
| raw | 96 bank | +8.98% | +10.89% | fails Sintel LPIPS/correlation, REDS-154 correlation | fails Sintel |
| 80% blend | 96 bank | +11.10% | +7.60% | **passes** | **passes** |

### Eight-source holdout, torch level (`paired-critic-holdout/torch-holdout8.json`)

Scored on the frozen 960-pair quality holdout with the same scorer on the Mac (MPS). The 80% blend improves LPIPS and DISTS on every one of the eight sources (+6.10% / +11.92% source-balanced) but falls below the Lanczos correlation anchor on DucksTakeOff and RushHour. The raw checkpoint (+6.27% / +14.92%) regresses LPIPS on RushHour and Sunflower and exceeds the embellishment cap on RushHour (fine energy 1.104): on the grainy night scene it synthesizes speckle on flat dark surfaces, which DISTS rewards and pixel correlation punishes. Crops: `paired-critic-crops/rush_hour-*.png`, `ducks_take_off-*.png`.

### Native delivery holdout, 80% blend (`paired-critic-holdout/native-blend80-*.json`)

Converted with the shipping mixed-precision policy (`native-profile.json`: 5.63 ms at 640×360 and 9.89 ms at 864×480 on CPU+GPU, identical to the area-folded graph, conversion error under one RGB level). The Release app at `1d4c0aaf` ran all 32 conditions × 300 frames through preprocessing, reconstruction, detail and the real NV12 sender with the inherited candidate configuration (sharpness 0.4, radius 2). Source-balanced LPIPS 0.3288 → 0.3009 (**+8.47%**) and DISTS 0.1345 → 0.1175 (**+12.69%**), against the earlier raw-weight native candidate's +7.20% / +10.46%. Six sources improve both metrics by 6–24%; RushHour and Sunflower lose 1.9% / 2.6% LPIPS with fine energy 1.27 / 1.13, above the reference: the inherited 0.4 sharpening stacks on the model's synthesized grain. The frozen native gate fails on RushHour, Sunflower and Tractor correlation and Sunflower LPIPS; the perceptual gate fails on the same grainy sources. No promotion. A candidate-specific presentation sharpness (0 or 0.2) is the obvious next native test once the ladders choose the weights.
