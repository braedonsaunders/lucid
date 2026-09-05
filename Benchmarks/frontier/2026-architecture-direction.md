# Architecture direction after the September 4 review

The owner explicitly wants architectural progress using current research. EfRLFN remains a recorded baseline only; it is not the development target. Recent publication dates do not make inherited components novel. The current shipping SPAN graph and four-thousand-step loss ablations cannot by themselves satisfy the requested breakthrough.

## Research checked against primary sources

- **August 17, 2026 — [PixRestore](https://arxiv.org/html/2608.16793v1):** pixel-space flow matching, adaptive hierarchical DINO guidance and one-step adversarial refinement. Its reported 44 ms latency is for a 512×512 image on A800, not real-time Mac video. It motivates retaining pixel evidence and learning from stronger visual features; our prototype is neither its architecture nor its trained model.
- **August 3, 2026 — [MoCRA](https://arxiv.org/html/2608.01829v1):** shares rank-1 conditioning between coarse processing and a shallow native-scale refiner, with temporal regularization and degradation-evidence-weighted supervision. Its reported native-4K runtime is roughly half a second; the task is restoration, not our streaming SR benchmark. Frequency-specific computation and compact conditioning are useful hypotheses to adapt and test.
- **July 24, 2026 — [TRaM-VSR](https://arxiv.org/html/2607.22231v1):** motion/semantic token importance and local/global routing reduce one-step diffusion work. It offers a direction for allocating capacity to difficult detail. We have not reproduced its gains or established an M4 deployment route.
- **July 11, 2026 — [NanoVSR](https://arxiv.org/html/2607.10495v1):** compact recurrent propagation and inference reparameterization, with progressive sequence training. The published implementation is bidirectional and evaluates 15-frame chunks. Its throughput is not live capture-to-display latency; using it unchanged would introduce future-frame dependency.
- **WWDC 2026 — [Metal neural rendering](https://developer.apple.com/videos/play/wwdc2026/359/):** custom ML in Metal command buffers and small shader networks. The GPU neural accelerator described by Apple targets M5/A19 Pro. Lucid's measured machine is M4 Pro; no M5 hardware gain may be attributed to it.

These papers combine established and newer ideas. Selecting a new title is not the same as demonstrating a useful advance.

## Implemented experiment

`Tools/architectures/causal_detail.py` is a small, explicitly causal reconstruction prototype, separate from shipping. It accepts the current frame and caller-owned feature history. It packs every source sample into channels on a grid with 16× less area, updates bounded history, applies image-conditioned low-rank channel mixing and local gated detail blocks, then reconstructs a 2× or 4× residual. A bilinear path preserves the initial image; the residual starts at zero.

This is an original engineering combination for experimentation, not a reproduction of the papers and not a verified novelty claim. Both the 8,000-step screen and completed 20,000-step causal model fail quality versus Lanczos; see `causal-training-v1.md`. No trained weights have been promoted. Tests cover reset isolation, bounded recurrence, both output scales and an effective gradient path through temporal state. Core ML probes now check both nonzero random graphs and trained weights across independently accumulated state and mid-sequence resets, including explicit state transfers.

## Mac feasibility results

Both 2× prototypes passed Core ML conversion checks. Worst observed image difference from PyTorch was below one 8-bit level; state differences were below 0.004. The probe uses ten timed predictions after five warmups, not a sustained playback test.

| Prototype | Route | CPU + GPU mean / p95 | All compute units mean / p95 |
|---|---|---:|---:|
| 16 channels | 1080p → 4K | 17.10 / 18.36 ms | 29.15 / 29.86 ms |
| 32 channels | 1080p → 4K | 19.36 / 22.49 ms | 33.57 / 34.79 ms |

The wider graph retains plausible room for 30 fps inference, while the all-compute-units configuration is slower. Explicit float32 history is 8.29 MB / 16.59 MB respectively. See `causal-detail-ch16-profile.json` and `causal-detail-ch32-profile.json` for the complete 360p/720p/1080p measurements and limitations. These are untrained graph costs, not image quality, end-to-end latency or delivered browser cadence. Larger temporal capacity, motion reliability and training may change the design and cost.

## Early experiment sequence (historical)

1. Completed architecture probes at 360p, 720p and 1080p. The trained v1 1080p→4K graph also passes recurrent conversion checks, with CPU+GPU means of 18.8–20.4 ms in two short probes. This is not playback cadence.
2. Built 168 true consecutive codec-degraded 2× sequences with source-family receipts and shared HR/LR geometry. Training and validation are separated by source family; references still inherit source compression.
3. Completed causal v1 training on the RTX 4080; its matched no-history control follows in the same protected experiment. The completed causal model is rejected. Forcing resets scarcely changes spatial quality; changing only its interpolation floor improves detail but increases flicker.
4. Implemented v2 with a stronger fixed interpolation floor and direct packed-pixel bypass beside temporal features. Its untrained nonzero graph passes conversion and measures approximately 21 ms mean at 1080p→4K. Train it with the same bank and schedule after the existing GPU job finishes, then score it against the floor, v1 and controls. No v2 quality result is implied by timing.
5. The official PixRestore code is now reviewed and its reverse-reliability hierarchical feature supervision is adapted as a training-only loss. A real-weight gradient probe and CUDA smoke pass; a matched fine-tune/control experiment is running. See `dino-supervision-experiment.md`. This does not add the full PixRestore architecture or teacher to native inference.
6. Require independent perceptual/detail gains and temporal stability before integrating explicit feature state with cut/seek/reconnect resets. V2's final checkpoint still fails the spatial/temporal gates, and its training-validation improvement did not generalize. A separate full-frame codec bank is verified for a later data comparison. Measure sustained native/browser cadence and latency, including all copying and presentation.

The regular Codex goal remains active. Quality, long-duration temporal behavior, higher-resolution coverage, end-to-end cadence and final delivery are unfinished.

## Quality/performance follow-up

The later stateful native graphs reduce explicit-history cost, but the causal students still trail shipping quality. Increasing width, hierarchical feature loss and pixel-target distillation did not establish a joint quality win. `shipping-direct2x.md` records the stronger route: preserve shipping reconstruction and fold its output head to the requested 2× size. On M4 Pro, native prediction plus identically sized NV12 packets falls from 45.7 to 20.0 ms at 720p→1440p, and 25.2 to 11.6 ms at 540p→1080p. These omit capture and browser delivery. The folded initialization remains the experimental quality/performance baseline; none of the new training checkpoints is promoted.

PixRestore's tiled teacher improves perceptual quality but loses real detail relative to shipping. A fixed 50/50 teacher/shipping target preserves more detail while improving perceptual distances. Pixel regression fails to transfer that joint benefit. The current RTX 4080 arm uses the **released PixRestore multi-layer DINO discriminator**, checked by source hash, against real training references. Its two-step CUDA smoke passes; the initial added output-head gradient is 10.9% of the reconstruction gradient. Full-run results and independent source evaluation remain required. The feature encoder/discriminator are training-only. Flicker work remains deferred at the owner's direction.

## Additional 2026 primary-source checks

[DUO-VSR, CVPR 2026](https://cszy98.github.io/DUO-VSR/) combines progressive initialization, distribution matching, feature-space adversarial feedback and preference refinement. The authors identify shifted/artifact-contaminated teacher guidance as a limitation. This supports testing richer supervision; it does not prove the cause of Lucid's failed pixel-transfer experiments. The project page still labels its code release pending; no implementation or performance result is claimed here.

[CDA-VSR, CVPR 2026](https://github.com/sspBIT/CDA-VSR) uses bitstream motion vectors, residual maps and adaptive reconstruction for online VSR. Its released source was inspected at commit `5630821e5df4e110878d0caae39bcc53f658e9a9`. This is a concrete quality/performance research direction, but Lucid's current browser bridge supplies decoded pixels rather than those coded-domain inputs. The implementation also requires MMCV deformable convolution. Neither missing input priors nor that operator may be replaced silently while claiming a reproduction.

The checked source selects its heavy reconstruction branch only for the initial frame without supplied history; its public test YAML also passes `spynet_path`, absent from the model constructor. These integration issues need resolution before a faithful benchmark. No CDA weights were downloaded, no model was executed, and no author-reported speed is attributed to the Mac.
