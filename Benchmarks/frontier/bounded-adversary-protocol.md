# Bound the adversarial contribution without changing inference

The reference-detail experiment's weighted GAN gradient at the output head grew from 0.149× the reconstruction gradient at initialization to 9.112× at step 200. It still lost detail on full-frame faces. This experiment tests whether keeping that contribution near its initial magnitude improves the fidelity/perception tradeoff.

At every step compute the norms of the reconstruction gradient and the already-weighted GAN gradient at the output convolution weights. Multiply the GAN term by `min(1, 0.15 * reconstruction_norm / max(weighted_GAN_norm, 1e-12))`, detached from autograd. This only attenuates the existing adversary; it cannot amplify it. The cap of 0.15 is fixed from the measured initial ratio before running this experiment. It is a head-gradient proxy, not a bound on every parameter or a guarantee of detail fidelity.

Keep the reference-detail run's data, caches, initialization, seed, architecture, losses, reference detail targets, optimizer and 8,000-step schedule. The pinned August 2026 PixRestore discriminator stays unchanged. Reuse its completed uncapped control after matching new two-step smoke runs on first-batch and discriminator hashes. New code records the maximum applied head ratio across all steps and minimum scale. Unit tests verify attenuation, detachment, zero-gradient behavior and the actual combined-head gradient.

Evaluate only the raw final checkpoint and the same predetermined 80% blend on the frozen 48 full-frame and 96 validation-patch pairs. Use the existing 3% LPIPS/DISTS minima and per-source perceptual/detail guards. No post-result cap or blend sweep is allowed. Any winner still needs native correctness, matched end-to-end timing and fresh release footage. Shipping weights, native processing, and flicker filters remain unchanged.

## Completed experiment: rejected

Both smoke runs matched discriminator initialization and first-batch hashes. The RTX 4080 completed 8,000 steps in **6.5013 minutes**. The maximum applied ratio across every training step was **0.1500000209**, within FP32 rounding of the cap. The minimum adversarial scale was 0.00200738. The cap worked computationally; it did not produce a perceptual winner.

| Evaluation / variant | LPIPS change versus shipping | DISTS improvement | Outcome |
|---|---:|---:|---|
| 48 full frames, raw | 4.60% worse | 3.41% | Fail: LPIPS and Johnny fine detail |
| 48 full frames, fixed 80% | 3.52% worse | 2.65% | Fine-detail guards pass; perceptual gates fail |
| 96 validation patches, raw | 2.38% worse | 1.04% | Perceptual gates fail |
| 96 validation patches, fixed 80% | 1.61% worse | 0.80% | Perceptual gates fail |

The 80% full-frame fine-correlation deltas were CrowdRun −0.00031, FourPeople −0.00060 and Johnny −0.00834. Attenuating the adversary recovered more faithful detail relative to the uncapped model but lost the desired perceptual improvement. This is evidence of a tradeoff, not evidence that another cap would solve it. No follow-up cap or blend sweep was performed and no weights were promoted.

The adjacent training logs, smoke receipts, full-frame/patch reports and gate outputs preserve the result. Evaluation verified matching frame-manifest and shipping-weight hashes against the previous controlled reports, with matching source/sequence/frame identities. Four objective tests and three isolated transport-boundary tests passed during this work. No flicker objective or native temporal filter changed.
