# Locate the detail loss before another reconstruction experiment

The completed RTX 4080 diagnostic compares shipping, the fixed 80% reference-detail candidate and its fixed 80% bounded-adversary counterpart on the same 48 full-frame development pairs. All input file hashes, the frozen manifest and checkpoint hashes are retained. This is post-result diagnosis, not an independent model-selection set or a replacement promotion metric.

To separate strong edges from lower-gradient areas, masks use only reference luminance after float64 sigma-1 Gaussian blur. Thresholds are <1, 1–4 and ≥4 luminance levels/pixel. The labels describe gradient magnitude, not semantic faces, texture, or noise. The float blur avoids the legacy metric's uint8 blur rounding but is a distinct kernel implementation. Existing promotion gates remain unchanged.

| Reference edge region | Shipping fine correlation | Uncapped 80% | Bounded 80% |
|---|---:|---:|---:|
| CrowdRun | 0.39485 | 0.38589 | 0.39465 |
| FourPeople | 0.69540 | 0.68228 | 0.69557 |
| Johnny | 0.72095 | 0.69987 | 0.71056 |

The uncapped candidate increases edge-region fine energy while reducing correlation on all three sources. Its fine-band error also increases on all three. Thus the observed fidelity loss persists with floating-point filtering and in strong reference-edge regions; it cannot be dismissed as only low-gradient quantization noise. The bounded candidate recovers most edge fidelity on CrowdRun and FourPeople but still loses correlation on Johnny. Neither model has passed the existing joint perceptual/detail gates.

These measurements identify a fidelity/perception conflict but do not identify a successful remedy. They do not prove texture hallucination semantically, and they cannot validate a different cap or blend chosen after seeing the results. No further cap or blend sweep was run. More high-frequency energy alone is not faithful reconstruction.

`detail-regions.json` stores all 144 frame/model rows. `detail-regions-summary.json` averages the 16 conditions/frames per source. `detail-regions-command.ps1`, the log and exit receipt preserve execution. The helper tests verify float filtering and reference-only masks with synthetic constant/edge examples. Existing training jobs were checked before using the GPU; the diagnostic finished successfully and left no Python training process. Shipping weights and temporal filters remain unchanged.
