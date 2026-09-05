# Stronger detail at lower native output cost

Shipping still beats the small causal prototypes on LPIPS and fine correlation. Preserve its learned reconstruction, reduce unnecessary output work, and test whether the August 2026 PixRestore teacher supplies complementary restoration. Neither the head folding nor output blending is claimed as a new research invention.

## Fixed target combinations

`teacher-complementarity.json` contains all 48 frames / three development sources with input and teacher hashes. Every coefficient is fixed across the entire set; no reference pixel guides output construction. Shipping is its genuine 4× RGB8 output resized to 2× with PIL bicubic.

| Variant | LPIPS ↓ | DISTS ↓ | Fine correlation ↑ |
|---|---:|---:|---:|
| Shipping | 0.302483 | 0.128347 | 0.564311 |
| PixRestore tiled | 0.276422 | 0.113650 | 0.525790 |
| 25% teacher / 75% shipping | 0.300454 | 0.124150 | 0.568305 |
| 50% teacher / 50% shipping | 0.293713 | 0.119831 | 0.561982 |
| 75% teacher / 25% shipping | 0.285284 | 0.116202 | 0.547023 |
| Teacher low band + shipping high band | 0.296644 | 0.118236 | 0.564126 |
| Direct 2× area head | 0.296884 | 0.128084 | 0.557484 |

The 50/50 blend improves LPIPS 2.90% and DISTS 6.64%, with per-source fine-correlation changes of -0.00631, +0.00716 and -0.00784. It comes close to the development perceptual gate while preserving substantially more real detail than the teacher alone. The frequency split uses PIL Gaussian radius 1.2 and improves DISTS 7.88% with almost unchanged aggregate fine correlation, but LPIPS improves only 1.93%. These are potential offline training targets, not deployable two-model inference paths. They are repeatedly inspected development results, not untouched validation.

## Direct 2× without retraining

`fold_shipping_head.py` averages the correct 2×2 phase groups in the 8-phase output convolution. Its 192 output channels become 48 and PixelShuffle(8) becomes PixelShuffle(4); the unshuffled reconstruction trunk and learned input weights stay intact. The output equals 2×2 area pooling of the original **unclipped floating-point** 4× result, including borders. It is not mathematically identical to the usual clipped RGB8 bicubic presentation. Full-image quality is therefore measured separately above.

Three focused tests verify phase order and borders, reject wrong head scales, verify trunk/head gradients, and load the resulting 2× checkpoint without changing its output. Shipping assets are not overwritten.

`shipping-direct2x-native.json` records alternating 40-sample measurements after ten warmups on M4 Pro, CPU+GPU Core ML, using four deterministic RGB noise frames. Conversion error stays below one RGB level against each graph's own FP32 reference.

| Input | Shipping 4× mean / p95 ms | Direct 2× mean / p95 ms | Mean reduction |
|---|---:|---:|---:|
| 640×360 | 9.693 / 11.161 | 5.856 / 6.830 | 39.6% |
| 1280×720 | 35.893 / 36.511 | 21.205 / 21.704 | 40.9% |

Output dimensions differ deliberately: 4× versus the requested 2×. Shipping's subsequent downsampling is excluded. These are synchronous Python/Core ML graph timings, excluding capture, transport and presentation, not sustained browser cadence. An ALL-compute probe failed the strict maximum RGB error check (shipping maximum 4.97 levels, mean 0.418 on random noise); it is not used for performance conclusions. CPU+GPU passes.

## Controlled transfer experiment

The new 2× student starts from the folded shipping weights, rather than the much weaker causal student. Two 8,000-step RTX 4080 arms share initialization, batch/crop, seed 20260914 and cosine learning rate 0.00002→0.000002. Both cache the complete 256-pixel LR / 512-pixel presented shipping training targets before random crops. Control targets are shipping alone; candidate targets are the fixed 50/50 shipping/PixRestore mixture. Training uses L1 + 0.2 signed Sobel + 0.05 FFT to the target, plus 0.1 L1 to the real reference. Eight output border pixels are excluded. There is no new temporal/flicker loss.

The bank still has only 21 training/validation source identities and correlated crops. This test determines whether the measured complementary target can transfer into the stronger native route. It cannot establish broad generalization. Final checkpoints require matched development evaluation and then untouched holdout evidence before any promotion.

## Native Swift delivery confirmation

The Release app's offline `--presented-native-ms` probe alternates 60 measured predictions after ten warmups, then uses the real `EnhancedFrameSender` to produce identically sized 2560×1440 NV12 packets. Each full 5,529,799-byte packet is consumed with SHA256 outside timing. The two graphs have matching 1280×720 BGRA inputs and verified 4×/2× output sizes.

| Native Swift path | Graph mean ms | Packet mean ms | Total mean / p95 ms |
|---|---:|---:|---:|
| Shipping 4× → native downscale to 1440p | 31.768 | 13.923 | 45.691 / 46.855 |
| Direct 2× → 1440p NV12 | 19.587 | 0.412 | 19.999 / 20.498 |

Total measured time falls **56.2% (2.28× throughput)** at this fixed delivery size. Avoiding the large BGRA downscale saves substantial time beyond the graph reduction. This is an actual native delivery-stage saving, not a claim of 2.28× whole-browser throughput. It excludes capture/decode/input conversion, detail postprocessing, network and rendering. The native downscaler is not numerically equivalent to PIL bicubic in the quality screen. `shipping-direct2x-swift720.json` contains raw timings; `shipping-direct2x-swift-provenance.json` pins executable, source and model bytes. The local unsigned Release build succeeds.

At 960×540 input with 1920×1080 delivered output, the same Swift diagnostic measures **25.242→11.634 ms** mean and **25.935→12.271 ms** p95, a 53.9% mean reduction. Each packet is 3,110,599 bytes. Native graph / packet means are 18.170 / 7.072 ms for shipping and 11.319 / 0.315 for direct 2×. `shipping-direct2x-swift540.json` preserves all samples. Independent Python/Core ML conversion checks at this additional size pass; `shipping-direct2x-native540.json` records 20.036→12.088 ms graph means. These stage timings leave useful room in a 16.7 ms frame budget, but do not establish sustained 60 fps playback.

## Pixel-target transfer result: not promoted

Both 8,000-step RTX 4080 arms completed in 2.01 minutes each, exit zero. `presented-detail-training-results.json` records exact checkpoints and objectives. Each completed evaluation uses the same 12 source/codec conditions and frame indices.

| Variant | LPIPS ↓ | DISTS ↓ | Fine correlation ↑ |
|---|---:|---:|---:|
| Direct 2× initialization | 0.296885 | 0.128084 | 0.557484 |
| Matched shipping-only target control | 0.307230 | 0.128811 | 0.564920 |
| 50/50 teacher target candidate | 0.320489 | 0.124426 | 0.558477 |

The candidate improves DISTS 3.40% over its control but worsens LPIPS 4.32%. Fine correlation also falls 0.00644. Plain pixel/edge/FFT regression has not transferred the blended target's joint advantage. The initial folded weights retain the best LPIPS of these deployable graphs. Keep them as the experimental performance baseline; no new training checkpoint is promoted.

One plausible limitation is regression toward the average of generative teacher textures. This is a hypothesis, not a diagnosis proved by these metrics. PixRestore's official August 2026 implementation includes adversarial feedback through frozen multi-layer DINO features. The next controlled arm adds that released discriminator against known training HQ references, retaining the same reconstruction target/objective, initialization, data seed and schedule. The completed 50/50 candidate is the no-adversary control. `dino_adversary.py` verifies the exact upstream GAN file hash and the existing pinned DINO code/weights. Its discriminator and feature encoder are discarded at inference.

The adapted discriminator uses the released six spectrally normalized token heads and real label 0.8. Its optimizer is AdamW at 1e-4; the loss is half the sum of real/fake BCE and gradients are clipped at one. Generator adversarial weight is 0.005. Training logs compare weighted adversarial and reconstruction output-head gradient norms at steps 1, 200 and 1000. A two-step CUDA smoke run precedes the full arm; a focused gradient test verifies that generator gradients traverse the frozen encoder without updating its weights, while discriminator updates cannot alter the generator or reference. This is a tested adaptation, not a claim of reproducing the entire PixRestore training pipeline or inventing an adversarial method.


## Adversarial transfer and frozen candidate (September 5)

The matched 8,000-step adversarial arm completed on RTX 4080 in 5.89 minutes, exit zero. Its development LPIPS/DISTS improve to 0.277817/0.113572, but fine correlation falls to 0.548027 and violates the per-source fidelity guard. This raw checkpoint is not suitable for promotion. Training adds no inference-time discriminator or DINO encoder.

A fixed parameter interpolation with the shipping-target control recovers fidelity without adding inference branches. The coarse development sweep tested 25%, 50% and 75% adversarial parameters. Their LPIPS/DISTS/fine-correlation triples are 0.304551/0.126492/0.565626, 0.300958/0.123757/0.563796 and 0.294156/0.120224/0.558797. The 75% point narrowly misses the 3% LPIPS gate, so one refinement at 80% was evaluated and selected. This is development tuning, not an independent validation claim. Network interpolation is an established calibration technique; these nonlinear networks are not equivalent to blending their outputs.

The final paired CUDA development test consumes SHA-verified PNG pixels exported by the Mac. Shipping scores 0.302424 LPIPS, 0.128330 DISTS and 0.564319 fine correlation; the 80% candidate scores **0.291860 / 0.119177 / 0.557348**. That is **3.49% LPIPS and 7.13% DISTS improvement**. Per-source fine-correlation changes are -0.004499 (crowd), -0.008805 (four people), and -0.007611 (Johnny), all within the frozen 0.01 limit. The candidate SHA is `98886f65c0877f3813f46c6c9895597a8969a6b9f027446ebad5b1d803959852`.

`quality-holdout-candidate.json` freezes this selection before inspecting or scoring the eight holdout sources. No coefficient will be retuned against that holdout. The 21-source training bank remains small, and upstream teacher/legacy shipping training overlap cannot be fully excluded. Native delivery quality, broader content and actual playback remain separate requirements.

### Decoder reproducibility

The Windows FFmpeg build is from May 2023. Decoding the identical development MKVs there produced slightly different Lanczos/control scores from the Mac, so that preliminary Windows-decoded report is retained locally as a diagnostic and excluded from paired conclusions. `score_checkpoint_frames.py` verifies every frozen PNG's SHA and requires complete one-to-one frame/source pairing. The final development comparison and holdout use that route. CUDA versus MPS metric implementations still differ slightly; gains use the same device/backend within each paired report. Sequence evaluation now records decoder version and decoded RGB hashes.

### Trained candidate performance

The trained calibrated model preserves the 2× route's native saving. Python/Core ML conversion checks pass against each graph's own FP32 reference. In the Release Swift graph-plus-real-NV12-packet probe, with 60 alternating measured frames and ten warmups:

| Input → delivery | Shipping mean / p95 ms | Calibrated candidate mean / p95 ms |
|---|---:|---:|
| 540p → 1080p | 24.931 / 25.796 | 11.505 / 12.136 |
| 720p → 1440p | 45.830 / 47.063 | 20.026 / 20.599 |

`presented-calibrated-native-provenance.json` pins weights, native executable and packages. `presented-calibrated-profiler-snapshot.py` is the exact profiling source; the current utility subsequently gained stricter 4× input validation. As before, these measurements exclude preprocessing, detail, capture, network and presentation, and use synthetic input.

A separate real-video native enhancement-stage diagnostic includes preprocessing, learned RGB/NV12 conversions and detail. At 640×360 input it measures **12.01→7.24 ms** mean, with 60 frames after eight warmups. Shipping reconstructs 2560×1440; candidate reconstructs the requested 1280×720. This uses repeated development H.264 footage and sequential runs, with decode, input preparation and packet delivery excluded; it is not sustained browser cadence. The diagnostic now rejects incomplete sample counts and accepts a 2× external model only through `--pipeline-ms`. Production selection and bundled weights are unchanged. Geometry tests, 24 experiment tests and 15 evaluation tests pass; unsigned Release builds succeed.


The first actual sender-packet quality screen covers six H.264 development conditions, three sources and four distinct frames per condition. Native AVFoundation decode, current preprocessing, Core ML conversions, detail and sender downsampling are all included. Explicit Rec.709 NV12 decoding yields shipping **0.332802 LPIPS / 0.163547 DISTS / 0.412982 fine correlation**, versus candidate **0.300926 / 0.150519 / 0.408385**. The candidate improves distances 9.58%/7.97%, with all three fine-correlation drops below 0.01. These absolute scores are materially worse than weight-only scoring; stage exports are being added to distinguish input/decode alignment from processing losses before any production promotion. No new flicker tuning is involved. `presented-calibrated-native-delivered.json` pins all packet headers and scores. This diagnostic is development-only, uses repeated short footage and does not include browser rendering.
