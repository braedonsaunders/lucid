# Input-conditioned activation controller: matched experiment

Frozen September 5 before any trained result. The [training-source probe](activation-control-protocol.md) found a sensitive sparse control surface, but fixed attenuation reduced detail. This experiment tests learned input-dependent modulation against a static-input control. It is informed by [SPARK, September 3, 2026](https://arxiv.org/html/2609.03813v1), with a different convolutional backbone, conditioning representation, channel ranking and training objective. It is not a SPARK reproduction or a novelty claim.

The 2× area-folded shipping graph and all its weights stay frozen. At six block outputs, modulate the four channels selected by the completed training-only probe. A 7,408-parameter controller receives channel-wise mean and mean absolute first-layer features, normalizes the resulting 64-vector, and predicts 48 values through a 64-wide SiLU MLP. Zero-initialized final weights produce the exact original output. Tanh bounds each selected gain to [0.5, 1.5] and shift to ±0.2 of that channel's measured training RMS. Unselected channels and auxiliary block outputs are unchanged. The reconstruction still changes nonlinearly: frozen weights do not guarantee image fidelity.

The **dynamic** arm uses real pooled features. The **static** arm has the same parameters, initialization and computation, but supplies zeros to the controller. It can learn constant adjustments. This isolates access to input information; it does not assert equal effective function capacity. Both arms use fresh training, seed 20260914, 8,000 steps, batch four, crop 96, AdamW learning rate 0.0002 with cosine decay, no weight decay, 50/50 cached shipping/PixRestore target, reference Sobel/FFT terms, and released DINO discriminator weight 0.005. The controller's learning rate differs from earlier full-network fine-tunes, so comparisons with those experiments do not isolate architecture alone. Conditioning is learned on crops; larger-frame generalization is an explicit risk.

Require identical first input/output, controller/discriminator initialization, bank/cache and shipping identities in two-step CUDA smokes before full training. Verify immutable reconstruction/masks at saved checkpoints. The bank and shipping SHA256 values remain those in the activation probe. Native preflight uses a fixed nonzero random controller, requiring the existing maximum/mean RGB error limits of 3/0.6 levels. Interleave the unchanged folded graph to measure controller cost; no untrained native timing is a quality or playback claim.

Evaluate final unblended checkpoints on the frozen 48 full-frame development pairs and 96 source-disjoint bank-validation patches against shipping, with exactly the existing 3% LPIPS/DISTS improvement minimum, 2% per-source perceptual-regression maximum and 0.01 fine-correlation-loss maximum. Match shipping/Lanczos controls and downloaded checkpoint hashes. No validation-guided checkpoint, channel-count, range or blend selection. A passing development result still requires a fresh independent quality evaluation and native delivered quality/cadence. Do not admit 720p merely because an untrained graph runs quickly. Flicker work remains deferred.

## Nonzero native preflight

Three controller tests and four checkpoint/subspace regression tests pass. The random nonzero controller passes native RGB conversion checks at both sizes. In an interleaved 20-sample comparison after ten warmups, controller versus folded-baseline inference means are 6.640 versus 6.477 ms at 360p and 21.178 versus 21.013 ms at 720p. The respective p95 pairs are 8.612/8.018 and 21.389/21.352 ms. The observed mean overhead is about 0.16 ms in this short probe, not a sustained playback measurement or evidence of trained quality.

## CUDA identity preflight correction

The first static smoke stopped before optimization: adding a zero FP32 shift promoted BF16 block outputs and changed downstream residual-sum rounding. The correction casts gain and shift to each block's activation dtype before application. A new BF16 identity regression passes alongside the three previous controller tests. The failed smoke is preserved separately; no training or quality result was selected from it.

A fresh nonzero native preflight after the correction again passes RGB limits. Controller/folded means are 5.877/5.567 ms at 360p and 21.501/21.088 ms at 720p; p95 values are 6.156/5.904 and 22.888/21.842 ms. The variation across short probes is retained rather than selecting the best timing. Fresh CUDA runs use `activation-controller-20260905-r2`, with the same experimental recipe.

## Full training verification

Both arms completed 8,000 steps on the RTX 4080: static 7.114 minutes and dynamic 7.095 minutes, excluding initialization/cache loading. Their ten initialization/data/probe fields match each other and their verified smokes. Independent CPU hashing of downloaded final tensors confirms that reconstruction weights and masks retain their initial digest; the learned controllers differ from initialization and match their completion receipts. `activation-controller-verify.py` reproduces these checks. The two final checkpoint SHA256 values are `3ec428fe291d56bf03819736908a6c6f885d90caca9ab20db93d0faf074c6dff` (static) and `ba014dd0ffd442556a671a4fab61a1ed6bfb2c52745dc97b6df7490c52e30bba` (dynamic). Both frozen evaluations completed successfully; their outcome follows.

## Quality outcome: reject both final controllers

| Arm | Development LPIPS / DISTS improvement | Bank-validation LPIPS / DISTS improvement | Detail guards |
|---|---:|---:|---|
| Static | 4.64% / 5.48% | 6.06% / 3.24% | Fail all three full-frame sources and both REDS validation sources |
| Dynamic | 5.70% / 6.24% | 8.86% / 4.42% | Same failed source guards |

The input-conditioned controller adds perceptual gains over the matched static control, but increases detail loss on the two face sources. Both remain rejected; no final-model native export or 720p promotion is justified. Frozen weights alone have again not guaranteed final image fidelity. Original folding results suggest smaller initial detail losses, but their shipping controls differ from the current scorer beyond the predeclared 1e-5 tolerance. A fresh zero-controller evaluation is required to isolate that initial loss; the old report is not silently reused as a matched baseline.

The fresh diagnostic now confirms the cause: shipping controls reproduce exactly, and the untouched folded reconstruction loses only 0.00428/0.00879/0.00742 fine correlation on CrowdRun/FourPeople/Johnny. All pass the original 0.01 guard. Controller training adds the disqualifying losses. The diagnostic uses the verified frozen reconstruction with the controller's final projection zeroed, not a substituted checkpoint or altered evaluator. `activation-controller-fidelity-diagnosis.json` records the matched comparison.
