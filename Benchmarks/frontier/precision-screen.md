# Core ML precision screen: no useful speedup

The converter historically kept multiplication in FP32 to avoid near-black output on older conversion paths. On this macOS 26.5.1 / Core ML Tools 9.0 host, keeping only pixel shuffle in FP32 or converting everything to FP16 both passed the initial seeded-noise check. Separate timing screens were heavily confounded by unrelated workstation load and are retained only as exploratory evidence.

The subsequent comparison loaded all three package variants together. It checked eight input classes against each model's FP32 reference at both 360p and 720p: black, near-black, white, ramp, one-pixel checkerboard, saturated colors, noise and a decoded video frame. Timing order was shuffled deterministically within each iteration, with ten warmups and 60 recorded predictions per accepted variant. The input class changed each iteration. System load averages were recorded; this was not an idle machine.

| Input | Shipping current FP32-mul mean/p95 | Shipping FP32-shuffle mean/p95 | Shipping FP16 mean/p95 |
|---|---:|---:|---:|
| 640×360 | 9.75 / 10.80 ms | 9.95 / 10.68 ms | 9.70 / 10.27 ms |
| 1280×720 | 35.97 / 36.68 ms | 36.40 / 37.48 ms | 35.92 / 36.51 ms |

The sub-percent mean difference between the current shipping policy and FP16 does not establish a useful performance gain. Production precision policy is unchanged.

Every shipping case passed the maximum 3 RGB-level / mean 0.6 RGB-level correctness limits. The unshipped trained 2× candidate failed the checkerboard under **all three policies**, including the existing conversion policy: maximum error about 4.4 levels and mean 1.3–1.4 levels. Its timing was excluded from the interleaved comparison. This exposes a conversion limitation missed by random-noise-only checking; no candidate conversion is promoted. The synthetic failure does not establish how often it occurs in natural footage, and the cause is not isolated yet.

`precision-interleaved.json` contains every per-input correctness result, timing sample and system-load observation. `precision-*-screen.json` preserves the preceding screens. These are native graph measurements and do not include playback, transport or output downsampling. The graph still uses shipping or previously evaluated candidate checkpoints; there is no new training in this precision experiment.
