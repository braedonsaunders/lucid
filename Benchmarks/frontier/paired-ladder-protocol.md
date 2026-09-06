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
