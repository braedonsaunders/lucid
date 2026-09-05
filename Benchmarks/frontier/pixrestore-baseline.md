# Current pretrained quality baseline — PixRestore-S

The official [PixRestore](https://arxiv.org/abs/2608.16793) paper was submitted August 17, 2026. Its released small model is a VAE-free pixel diffusion transformer with DINO conditioning and adversarial fine-tuning. This experiment evaluates the published pretrained model, beyond the feature-loss adaptation used in Lucid's small causal candidate.

Weights came from the authors' linked [Hugging Face repository](https://huggingface.co/VCLab-PolyU/PixRestore), revision `a5fe719a0c517422cff9f7a4895ab303e2320d51`. SHA-256: `13c10ca30fa865977de393457e731db9f2469159dbd69c85fb3c79b3edf06895`. The reviewed official code revision is `909ca06e614cf72e218d52b4fa41077c603d872b`. Strict checkpoint loading passed. Configuration, code hashes, DINO weights, exported frame hashes and settings are in `pixrestore-baseline-provenance.json`.

The RTX 4080 processed all 48 spatial screening frames using official `inference.py`, one step, CFG 1, seed 0, BF16 and bicubic 2× preprocessing. `--test-mode off` retains complete 1280×720 frames; there is no center crop or reference-dependent processing. The upstream Windows run initially failed because Triton was unavailable. `TORCHDYNAMO_DISABLE=1` bypasses only compilation; CUDA/BF16 inference remains active. This is not a reproduction of the paper's runtime measurement. Dependencies live in an isolated target directory; the training environment was not upgraded.

| Variant | LPIPS ↓ | DISTS ↓ | Fine correlation ↑ | Detail energy/reference |
|---|---:|---:|---:|---:|
| Lanczos | 0.345604 | 0.128056 | 0.509324 | 0.507199 |
| Bicubic | 0.335413 | 0.132425 | 0.490064 | 0.477270 |
| PixRestore-S | 0.290007 | 0.131534 | 0.445912 | 0.621701 |

PixRestore-S improves LPIPS **16.09%** versus Lanczos, while DISTS worsens **2.72%** and fine-detail correlation falls **12.45%**. It produces more high-frequency energy, but that energy is less aligned with the reference. Visual inspection of the low-bitrate Four People output showed visible vertical texture artifacts and distorted fine text/facial structure. It is therefore a useful perceptual target, not a replacement or an unquestioned source of training truth.

This result uses full-frame dynamic-size inference; the author's README also recommends a 512-square test crop. The subsequent controlled tile experiment below materially improves the result.

## Training-size tiles: stronger quality target

The same checkpoint, seed and one-step CUDA/BF16 inference now process overlapping 512×512 patches after identical bicubic 2× enlargement. Each complete 1280×720 frame uses six tiles, blended with normalized separable sin² windows. Processing never reads the reference. No evaluation pixels are cropped away. Identity round-trip tests verify complete pixel-exact coverage, including borders; missing coverage is rejected.

| Variant | LPIPS ↓ | DISTS ↓ | Fine correlation ↑ |
|---|---:|---:|---:|
| Lanczos | 0.345604 | 0.128056 | 0.509324 |
| PixRestore-S, full frame | 0.290007 | 0.131534 | 0.445912 |
| PixRestore-S, 512 tiles | **0.276422** | **0.113650** | **0.525790** |

The tiled model improves LPIPS **20.0%**, DISTS **11.3%** and fine correlation **3.2%** relative to Lanczos. Both perceptual distances improve on all three source identities. Fine correlation falls on Crowd Run (0.358420→0.346813) while improving on the two talking-head sources, so the aggregate improvement does not establish uniformly faithful reconstruction. Visual review shows the full-frame vertical texture streaks substantially reduced; severely compressed facial detail and text remain unresolved. This is a promising teacher for a controlled, reference-checked distillation experiment, not a shipping promotion.

The RTX 4080 completed all 288 tiles successfully. This test includes image file I/O and Windows compilation is disabled; it establishes no native latency or live-cadence result. `pixrestore-tiles-spatial-evaluation.json` contains all rows and output hashes, `pixrestore-tiles-provenance.json` links exact processing/model identities, and `pixrestore-tiles-command.ps1` records the remote invocation. Split/merge manifests and tiles remain under `.build/pixrestore-tiles`. Ten frontier evaluation tests pass.

`export_spatial_frames.py` exports frames 0, 4, 8 and 12 from the same 12 codec conditions as the causal development screen. `score_spatial_frames.py` verifies every source-frame hash and output size, records output hashes and averages equally across the three source identities. Its Lanczos scores reproduce the sequence evaluator. Full rows are in `pixrestore-spatial-evaluation.json`. Upstream training overlap is unverified; these 48 frames are development evidence, not an untouched generalization benchmark. No temporal metric is used in this quality/performance experiment.

Another current video baseline, [SwiftVR (June 2026)](https://arxiv.org/html/2606.09516v1), reports 26 fps at 1080p on an RTX 5090 and uses dense gathered-window attention. Its reported 1440p throughput processes 24-frame chunks in 0.766 seconds on an H100. That is useful throughput evidence, but chunk buffering/compute is incompatible with assuming sub-50-ms live latency. We have not run SwiftVR locally; these are the authors' measurements, not Lucid comparisons.
