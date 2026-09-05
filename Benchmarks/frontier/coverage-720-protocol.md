# 720p reconstruction coverage screen

Frozen before inspecting 720p reconstruction scores, September 5, 2026. This tests a new resolution route, not a replacement for the supported 144–480p models. No new training, output blending, temporal filtering or checkpoint selection.

Candidate: the deterministic area-folded shipping head, checkpoint `8dd69c03424003aa98585edcbbc6c659d882a94fa680efba7d784a28deba241a`. Shipping control: `fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65`. Keep both immutable. The already measured direct-2× native graph plus NV12 packet cost is 20.00 ms mean at 1280×720→2560×1440; it excludes capture, detail processing and presentation.

Use the first 16 frames of the 2160p50 CrowdRun, DucksTakeOff and ParkJoy sources from the [Xiph/SVT collection](https://media.xiph.org/video/derf/), reserved for evaluation. Preserve their linked SVT terms and lossless excerpt receipts. All three identities are excluded from documented Lucid training; legacy shipping/teacher provenance is incomplete. Lower-resolution versions of these scenes have already been evaluated, so this is a development coverage screen, not an untouched release holdout. Outdoor content dominates and cannot establish faces, animation or text quality.

Build full-frame 2560×1440 references and 1280×720 degraded inputs with the existing consecutive-sequence builder, explicit Rec.709 range contract, H.264 and VP9 at 1.4 and 4.0 Mb/s. These rates preserve the earlier 360p screen's nominal bits per input pixel. Export frames 0/4/8/12 to pinned PNGs: 48 pairs. Short excerpts do not establish steady encoder rate control or temporal performance.

Score the fixed candidate and genuine shipping 4× output with its declared RGB8 bicubic 2× presentation adapter. Include PIL Lanczos, bicubic and bilinear at exactly 2560×1440. No tiled scoring, resized metric inputs, per-frame coefficients or reference-guided output construction. The scorer's existing LPIPS/DISTS implementations and fine-correlation metric remain unchanged. PIL comparators do not reproduce browser scaling, color management or native pixel conversion.

To advance to native delivery validation, the candidate must improve source-balanced LPIPS and DISTS by at least 3% against **each** interpolation comparator, with at most 2% per-source perceptual regression and at most 0.01 per-source fine-correlation loss. Also report the unchanged replacement gates against shipping, without requiring a new-resolution route to improve a currently unsupported shipping route by 3%. This distinction is fixed in advance; existing replacement gates remain intact.

A successful spatial screen only authorizes the next experiment: exact native input/output geometry, native delivered quality versus interpolation, RGB correctness, and 720p30 installed-browser delivery of 2560×1440 at ≥28.5 fresh source-PTS frames/s and ≤50 ms p95 over ≥600 seconds. Any production admission needs those results and broader independent source coverage. No production resolution guard is changed by this screen.

## Completed result: reject 720p admission

The protected RTX 4080 evaluation completed successfully on all 48 frozen full-frame pairs. The fixed folded head fails every interpolation comparison under the predeclared joint gates.

| Comparator | LPIPS improvement | DISTS improvement | Joint gate |
|---|---:|---:|---|
| lanczos | 9.38% | -9.04% | Fail |
| bicubic | 7.86% | -6.59% | Fail |
| bilinear | 12.42% | 1.67% | Fail |
| shipping | 1.69% | 0.04% | Fail |

Negative improvement means worse. Against Lanczos, DISTS regresses on all three sources and DucksTakeOff exceeds the fine-correlation loss limit. Against shipping, the folded output remains close, but neither joint perceptual minimum is met. Model cost is therefore not the sole missing gate.

No native/browser admission run follows this failed spatial screen, and the production 144–480p support table is unchanged. The 20 ms native-stage timing remains a performance fact, not evidence that 720p reconstruction is better than interpolation. Full scores, per-source deltas, exact source/PNG manifests, evaluator hashes and the successful remote exit are committed alongside this protocol.
