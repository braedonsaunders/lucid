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

## Current experiment sequence

1. Completed architecture probes at 360p, 720p and 1080p. The trained v1 1080p→4K graph also passes recurrent conversion checks, with CPU+GPU means of 18.8–20.4 ms in two short probes. This is not playback cadence.
2. Built 168 true consecutive codec-degraded 2× sequences with source-family receipts and shared HR/LR geometry. Training and validation are separated by source family; references still inherit source compression.
3. Completed causal v1 training on the RTX 4080; its matched no-history control follows in the same protected experiment. The completed causal model is rejected. Forcing resets scarcely changes spatial quality; changing only its interpolation floor improves detail but increases flicker.
4. Implemented v2 with a stronger fixed interpolation floor and direct packed-pixel bypass beside temporal features. Its untrained nonzero graph passes conversion and measures approximately 21 ms mean at 1080p→4K. Train it with the same bank and schedule after the existing GPU job finishes, then score it against the floor, v1 and controls. No v2 quality result is implied by timing.
5. Require independent perceptual/detail gains and temporal stability before integrating explicit feature state with cut/seek/reconnect resets. A stronger training teacher remains an option to test, not an implemented feature. Measure sustained native/browser cadence and latency, including all copying and presentation.

The regular Codex goal remains active. Quality, long-duration temporal behavior, higher-resolution coverage, end-to-end cadence and final delivery are unfinished.
