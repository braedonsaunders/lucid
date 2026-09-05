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
