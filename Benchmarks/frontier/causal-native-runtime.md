# More reconstruction capacity within the native frame budget

The user directed work toward quality/performance on September 4, deferring flicker work. The stationary-loss experiment was stopped; its partial checkpoint and local patch are retained outside shipping code. Current training uses the existing reconstruction objective and reference-feature supervision.

## Measured execution improvement

The causal 1080p→4K prototype previously returned a 16,588,800-byte FP32 feature tensor after each prediction and supplied it to the next one. Core ML's persistent `MLState` removes that explicit round trip. This uses an established [Apple runtime capability](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html), not a claim of inventing stateful inference. The learned architecture and weights are unchanged.

`causal-native-paired.json` interleaves four graphs, rotating execution order over 60 measured rounds after ten warmup rounds. All use the same trained v2 checkpoint, moving random image input, CPU+GPU placement, and source/output sizes 1920×1080 / 3840×2160 on the M4 Pro. Other host workloads were left running. Image preparation and diagnostic output comparisons are outside timing; synchronous Core ML prediction and explicit history handling, when present, are inside.

| Fixed filter | History handling | Mean ms | p95 ms |
|---|---|---:|---:|
| Dense Lanczos | Explicit input/output | 22.179 | 22.662 |
| Separable Lanczos | Explicit input/output | 22.165 | 22.596 |
| Separable Lanczos | Internal MLState | 13.886 | 14.882 |
| Dense Lanczos | Internal MLState | 13.831 | 14.266 |

Internal state reduces mean graph time **37.6%**, or **1.60× throughput** in this comparison. A prior independent alternating run measured 23.526→14.975 ms, retained in `causal-native-paired-initial.json`. Separate non-interleaved measurements were noisy and are not used to establish the speedup.

The separable filter reduces fixed-filter arithmetic from 588 to 126 MACs per source RGB pixel but delivers no meaningful native speed benefit. It remains an optional experiment; dense filtering remains the default. This distinction matters: lower operation count alone did not predict runtime.

`causal-native-stateful.json` verifies six consecutive Core ML predictions against independently advanced Torch history, including a reset. Maximum image error is 0.712 RGB levels and state error 0.000669. The alternating native comparison differs by at most one RGB level across execution variants. State is initialized separately for each model; timing does not read it back to Python.

## Spending the saved time on quality

A nonzero, untrained 64-channel graph has **152,712 parameters**, versus 57,512 at 32 channels. With internal state, its 1080p→4K mean/p95 are **19.559/20.087 ms** (`causal-capacity64-native.json`). This is a capacity/latency feasibility result, not a quality result or a sustained playback claim.

`Tools/experiments/widen_causal.py` expands the best existing feature-supervised 32-channel candidate to 64 channels. Duplicated channels use complementary outgoing weights whose sum preserves the original function, while allowing different gradients. A trained four-frame check, including reset, changes output by at most 1.1921e-7 in normalized RGB. Tests also verify that the extra channels can receive different gradients. `causal-capacity-initialization.json` records the source, seed and transformation hash.

An RTX 4080 experiment now compares this expanded model with the original 32-channel initialization: 8,000 steps each, same seed 20260912, full-frame codec bank, crop/batch/schedule and DINO feature weight 0.03. See `causal-capacity-training-command.ps1`. Its quality outcome is pending; no candidate is enabled in Lucid.

## Reproduction

```sh
.venv-convert/bin/python Tools/experiments/profile_causal_detail.py \
  --checkpoint .build/causal-detail-ch32/v2-step020000.pth \
  --sizes 1920x1080 --compute-units CPU_AND_GPU --samples 60 \
  --stateful --out .build/causal-dense-stateful
```

Omit `--stateful` for explicit history; add `--separable-floor` for the optional filter variant. `compare_causal_native.py --model LABEL PACKAGE ... --report PATH` requires matching checkpoint identities and interleaves the supplied graphs. The profile checks fixed-shape conversion and independently evolved state before timing.

These are experimental Core ML graphs. Browser capture, native pixel conversion, transport, presentation, power and sustained memory remain outside these numbers. The shipping app has not gained 1.60× end-to-end performance from this experiment, and reconstruction quality still needs to improve before integration.
