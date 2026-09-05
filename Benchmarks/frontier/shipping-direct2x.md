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
