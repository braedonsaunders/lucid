# Causal reconstruction follow-ups — September 4, 2026

Three 4,000-step RTX 4080 fine-tunes are complete. None passes promotion. Shipping weights remain unchanged.

The feature-supervision experiment adapts the layer-reliability weighting in [PixRestore, submitted August 17, 2026](https://arxiv.org/abs/2608.16793), using its [official implementation](https://github.com/csslc/PixRestore) as the reviewed reference. This is an adaptation of its training supervision, not a reproduction of the pixel diffusion model, conditioning router or adversarial training. The older DINOv2 encoder is frozen and training-only; the Mac inference graph remains the 57,512-parameter causal v2 network.

All runs start from v2 step 20,000, use seed 20260908, AdamW learning rate 0.00002, batch 4, crop 96, BF16 CUDA and the same 3/7/12-frame curriculum. The candidate adds feature weight 0.03. Its control uses zero feature weight. A third run changes only the data-bank argument relative to the control, testing patches extracted after full-frame compression and inter-frame warmup. That changes the sampling/degradation distribution as a whole; it does not isolate one codec parameter.

`causal-followup-training.json` retains checkpoint hashes, full arguments, code hashes, teacher provenance and development validation. Shared initialization, architecture, code and optimization settings were checked directly from all three checkpoints. Remote paths are `C:/lucid/causal-frontier-20260904-dino/{dino,control,stream-data}`. Training took 3.17, 2.09 and 2.10 minutes respectively. After completion, no Python jobs remained and the RTX 4080 reported 0% utilization. No preexisting job was displaced.

## Independent development screen

Three source identities, four codec/bitrate conditions each, 16 consecutive frames, 360p→720p. Source identities are separate from the documented training bank. These repeatedly inspected excerpts are development data, not an untouched final set. Scores below are source-balanced; lower LPIPS, DISTS and residuals are better, higher correlation is better.

| Model | LPIPS | DISTS | Fine correlation | Static flicker | Flow residual |
|---|---:|---:|---:|---:|---:|
| Lanczos | 0.345604 | 0.128056 | 0.509324 | 1.003134 | 3.505586 |
| Matched control | 0.344263 | 0.128078 | 0.515652 | 1.006320 | 3.496671 |
| Feature supervision | 0.341207 | 0.128013 | 0.514907 | 1.009030 | 3.501556 |
| Full-frame codec data | 0.343127 | 0.127926 | 0.515917 | 1.007736 | 3.494243 |

Feature supervision improves LPIPS 0.89% over its control but slightly worsens correlation, static flicker and flow residual. Relative to Lanczos, DISTS improves only 0.034%, and face-clip flicker increases 7.67% / 5.58%. The data experiment improves both spatial metrics slightly, but face-clip flicker increases 8.56% / 7.25% versus Lanczos. These gains do not meet the ≥3% spatial improvement and per-source temporal gates.

Full rows and reference-flow coverage are in `causal-dino-evaluation.json` and `causal-stream-data-evaluation.json`. The flow mask is computed from reference frames only; it is an approximate consistency check, not ground-truth optical flow. The old 4× shipping model is not directly scored in this 2× screen, so these results establish neither superiority to shipping nor RTX parity.

The next training hypothesis should target the demonstrated stationary-face flicker while preserving detail, rather than spending more steps on the same reconstruction objective. The full-frame bank is retained for that controlled experiment. Longer independent clips and a broader final corpus remain required before any promotion.
