# Causal 2× training experiment

This is an experiment toward useful 720p/1080p coverage, not a promoted model. The initial Core ML timing result measures the architecture's cost; it does not prove reconstruction quality.

## Data and controls

`causal-bank-manifest.json` records 168 sequences of 16 consecutive frames. Each contains a 256×256 RGB reference and an independently codec-degraded 128×128 input. Every LR sequence is encoded from the corresponding single decoded HR sequence, preserving frame order; there are no separate LR/HR seeks. H.264 and VP9 use four CRF severities, with crop scale and position varied deterministically. Source compression remains part of the reference.

160 sequences come from the Netflix public scenes and Big Buck Bunny; eight come from Sintel. All Netflix scenes are conservatively grouped in one source family. Sintel is excluded from this new model's training, and both duplicate source hashes and family overlap across splits are rejected. This validation family is development data, not an untouched final test set. One validation crop is black; aggregate validation PSNR is computed from aggregate MSE so it cannot receive disproportionate weight from an arbitrarily high per-image PSNR.

The two runs use the same initialization seed, samples, augmentation, optimizer, learning-rate schedule and 20,000-step budget. The control disables temporal history. The curriculum advances from 3 to 7 to 12 frames. BF16 accelerates CUDA convolutions; reconstruction, spectral and temporal residual objectives are computed in float32. Their weights are 1× Charbonnier, 0.05× signed Sobel, 0.01× FFT and 0.02× change in reconstruction error. This is initial reconstruction training; a generative or feature-guided fine-tune is not yet tested for this architecture.

Remote directory: `C:/lucid/causal-frontier-20260904-v1`. Actual command: `causal-training-command.ps1`. The launcher checks existing Python jobs before each run, refuses existing output directories and holds an experiment lock. It does not interrupt other jobs. Each checkpoint records code hashes, bank hash, arguments, RNG state and optimizer state. Shipping weights are unchanged.

## Initial independent screen: reject checkpoint 8,000

`causal-8k-evaluation.json` scores three sources excluded from this bank, two codecs and two bitrates, at 640×360 → 1280×720. Source-balanced results:

| Metric | Lanczos | Causal 8k |
|---|---:|---:|
| LPIPS ↓ | 0.34560 | 0.36183 |
| DISTS ↓ | 0.12806 | 0.13406 |
| PSNR Y ↑ | 28.6318 | 28.1147 |
| Fine-detail energy, reference=1 | 0.50720 | 0.43977 |
| Fine-detail correlation ↑ | 0.50932 | 0.49839 |
| Static flicker ↓ | 1.00313 | 0.96408 |

The early model trades fidelity/detail for stability and fails promotion. Finish the controlled experiment before attributing this to recurrence. A promising runtime is not sufficient: future changes must preserve a strong spatial reconstruction floor, demonstrate that history adds useful detail, and improve perceptual quality without adding shimmer. Native color/preprocessing parity also requires verification before any trained graph enters playback.

## Completed causal run: checkpoint 20,000 also rejected

The causal run completed 20,000 steps in approximately ten minutes on the RTX 4080. Its matched history-free control is a separate run; no control conclusion is inferred from an inference-only reset. `causal-final-evaluation.json` contains the complete 12-condition independent development screen. Compared with Lanczos, final LPIPS is 0.36100 versus 0.34560, DISTS 0.13199 versus 0.12806, PSNR Y 28.2468 versus 28.6318, and fine-detail correlation 0.50374 versus 0.50932. Static flicker is slightly lower, 0.97522 versus 1.00313. The model still fails the quality gate.

`causal-final-diagnostic.json` holds a controlled inference diagnostic:

| Variant | LPIPS ↓ | DISTS ↓ | Detail correlation ↑ | Static flicker ↓ |
|---|---:|---:|---:|---:|
| Bilinear | 0.36157 | 0.14355 | 0.48341 | 0.86581 |
| Lanczos | 0.34560 | 0.12806 | 0.50932 | 1.00313 |
| Trained causal | 0.36100 | 0.13199 | 0.50374 | 0.97522 |
| Same weights, reset every frame | 0.36087 | 0.13246 | 0.50169 | 0.95583 |
| Lanczos plus the unchanged learned residual | 0.34322 | 0.12335 | 0.51331 | 1.12175 |

The weak spatial result persists with history disabled. Simply substituting a sharper base recovers perceptual/detail scores but raises flicker approximately 11.8% versus Lanczos and reduces PSNR. That post-hoc variant was not trained and is not a promotion candidate. It motivates training a stronger spatial path, rather than assuming temporal recurrence is the primary cause.

The trained checkpoint also passes six-frame Core ML conversion checks with independently accumulated native/Torch history and a mid-sequence reset. Both tested compute-unit configurations remain below one 8-bit image level of maximum discrepancy. CPU+GPU 1080p→4K prediction measured 20.44/24.45 ms mean/p95 in the first run and 18.77/20.59 ms in the repeat that records the final profiler hash. Both reports are retained, rather than selecting only the faster measurement. These short graph timings exclude capture, video decoding and display; they do not establish sustained cadence or quality. Explicit float32 history consumes 16.59 MB.

## Completed matched control and v2 checkpoint screen

Both v1 runs finished 20,000 steps. `causal-v1-matched-training.json` verifies their identical code/bank hashes and all training arguments except history and output directory. `causal-matched-flow-evaluation.json` scores the completed pair and a v2 checkpoint at 12,000 steps on all 12 conditions with reference-defined flow and occlusion masks.

| Variant | LPIPS ↓ | DISTS ↓ | Detail correlation ↑ | Static flicker ↓ | Flow residual ↓ |
|---|---:|---:|---:|---:|---:|
| Lanczos | 0.34560 | 0.12806 | 0.50932 | 1.00313 | 3.50559 |
| Causal v1, 20k | 0.36100 | 0.13199 | 0.50374 | 0.97522 | 3.46090 |
| No-history v1, 20k | 0.36093 | 0.13243 | 0.50192 | 0.95724 | 3.45420 |
| Causal v2, 12k | 0.34192 | 0.12747 | 0.51535 | 1.01377 | 3.50192 |

History gives v1 a small detail/DISTS benefit, but no convincing overall gain and slightly worse flicker than its matched control. V2 recovers spatial detail: source-balanced LPIPS improves 1.07% and DISTS 0.46% versus Lanczos. Both miss the 3% gate. Its aggregate flicker increase of 1.06% also hides increases of 8.70% on Four People and 8.76% on Johnny. It is not eligible for promotion. Reference-flow coverage averages 86.8%; reference warp error averages 2.90 gray levels, so these estimated-motion scores remain imperfect evidence. The unwarped metrics are retained alongside them.

This comparison changes the next work: finish v2, test feature supervision with a matched control, and address source-specific static-region instability. The separately built full-frame codec bank is a later data experiment; combining it into the feature-loss comparison would hide which change helped.

```sh
.venv-convert/bin/python Tools/frontier_eval/diagnose_causal_detail.py \
  --manifest .build/frontier-sequences-2x/sequences.json \
  --checkpoint .build/causal-detail-ch32/step020000.pth \
  --report .build/causal-detail-ch32/diagnostic-final.json
.venv-convert/bin/python Tools/experiments/profile_causal_detail.py \
  --checkpoint .build/causal-detail-ch32/step020000.pth \
  --sizes 640x360 1920x1080 --out .build/causal-trained-profile-verified
```

## Spatial-path experiment v2

`Tools/architectures/causal_detail_v2.py` retains the v1 causal trunk and adds two deliberately explicit changes: a fixed half-pixel-centered Lanczos-3 floor and a direct packed-pixel bypass into the reconstruction head. This preserves the source samples beside the compressed temporal features. The correction still starts at zero. The fixed filter has no learned parameters and does not mix colors; replicated borders differ from PIL's truncated boundary kernels. Overshoot is clipped only after adding the learned correction.

This is an engineering experiment informed by the measured failure and the native-detail principle in the reviewed August 2026 MoCRA paper, not a reproduction or a claimed new interpolation method. V1 code and checkpoints remain intact. The same source bank, sample seed, curriculum and training budget are used to evaluate this combined architecture change; a positive result alone would not isolate which of its two changes helped.

Nine experiment tests pass, including constant-color preservation, channel isolation, a continuous-sinusoid reference that tests interpolation detail fidelity, reset independence, and gradients through the new pixel bypass. A CPU training smoke completes forward/backward, validation and checkpoint serialization. The nonzero random-head v2 graph passes recurrent Core ML conversion and costs approximately 21.4 ms mean at 1080p→4K on CPU+GPU in its initial probe. It has not yet established trained quality or playback performance. `causal-v2-training-command.ps1` refuses existing GPU jobs and writes to a separate experiment directory, reusing the immutable v1 bank.

## Reproduce the bank and training

```sh
.venv-convert/bin/python Tools/experiments/build_causal_bank.py \
  --sources .build/causal-training-sources.json --out .build/causal-bank-v1
.venv-convert/bin/python Tools/experiments/train_causal_detail.py \
  --bank .build/causal-bank-v1 --out .build/causal-trained \
  --steps 20000 --channels 32 --blocks 4 --batch 4 --crop 96 --device cuda --precision bf16
# Repeat into a separate directory with --no-history for the matched control.
```

The source-list fields are `id`, `path`, `family`, `split`; the full normalized list is retained in the bank manifest. Existing original sources are required. Six experiment tests cover temporal state, reset behavior, source leakage rejection and shared spatial/temporal augmentation. Four evaluator/loss tests cover the scoring guardrails.

## Browser packet change

`browser-packet-copy.json` verifies copying NV12 video directly into a payload view inside the final packet, eliminating the second full-frame input memcpy. Actual Chrome pixels and plane layout match byte-for-byte. The native app parses the padded JSON header and presents the result. After Off settles, the sent count remains 91 across two seconds, no frames remain in flight, and the canvas clears.

The short fixture measured 29.9 versus 30.4 presented fps and reported p95 latency 35.6 versus 37.0 ms. This does not establish a cadence improvement. The fixture uses MessageChannels rather than an installed extension, and native delivery is resized to its 1280×720 display surface; it is not a full-1440p transport measurement. The change saves one 345,600-byte copy per input frame in this test. Canonical Safari resources match. Isolated app tests must set `LUCID_EPHEMERAL=1` and separate bridge/token ports to preserve the user's instance.
