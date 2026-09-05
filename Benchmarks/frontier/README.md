# Lucid development evidence — September 4, 2026

The active objective uses the regular Codex goal. These are measured development results, not a declaration of RTX parity or release readiness.

## Current decisions

- **Current priority: quality/performance; flicker work deferred at the user's direction.** The [direct extension-surface capture route](surface-capture-transport.md) removes a frame serialization hop. Installed Chrome with shipping weights delivered 59.47 draw acknowledgments/s over 621 seconds at 40.60 ms p95. A same-session route comparison improved pooled draw cadence 29.6%. A second 620-second test confirms 58.38 changing-source-timestamp frames/s at 40.90 ms p95. This does not establish physical scanout or Safari performance.
- **Quality replacements still fail promotion.** The [native eight-source result](native-quality-holdout.md) improves aggregate perception but loses fine detail on three sources. Broader data, reference-detail supervision and [bounded adversarial training](bounded-adversary-protocol.md) did not establish a joint winner. The latest RTX 4080 run completed 8,000 steps in 6.50 minutes; the fixed 80% checkpoint preserves the full-frame detail guards but worsens LPIPS 3.52%. Shipping weights remain unchanged. The [regional diagnostic](detail-regions.md) checks where reconstruction errors occur without changing any gate.
- The [folded shipping head](shipping-direct2x.md) offers a measured native 2× cost reduction, but trained replacement correctness and independent quality still gate higher-resolution deployment. [Persistent Core ML state](causal-native-runtime.md) reduces experimental recurrence costs; the causal models themselves remain rejected. Verified 2026 sources and measured limitations are summarized in [architecture direction](2026-architecture-direction.md).
- **Latest external baseline:** [July 2026 NanoVSR](nanovsr-baseline.md), pinned released weights with strictly loaded parameters. Its past-only execution improves development LPIPS 5.55% and DISTS 3.57% but fails all three source fine-detail guards; it is not promoted. Official bidirectional execution also fails.
- Latest causal experiments: [August-2026 feature-supervision and full-frame codec-data ablations](causal-followups.md) are complete and rejected for promotion; spatial gains remain small and face flicker regresses. [Native/browser color correction](browser-color.md) fixes darkened sRGB midtones, with a reproducible Chrome chart and 34 passing native tests.
- Shipping ch32utc weights remain unchanged. Both 4,000-step RTX 4080 fine-tunes failed the independent sequence screen. Feature/edge supervision helped relative to its matched control but still lost to shipping.
- `sequence-evaluation.json` contains all 48 condition/model results: 3 source identities × 2 codecs × 2 bitrates × 4 variants. Every sequence has 16 consecutive frames at 320×180 → 1280×720. Spatial scores use frames 0, 4, 8, 12; temporal scores use all consecutive frames. A source-balanced average prevents repeated conditions from overweighting a scene.
- Shipping: LPIPS 0.45957, DISTS 0.20856, static flicker 0.90169. Feature/edge candidate: 0.46883, 0.20967, 0.92492. Control: 0.47498, 0.20872, 0.93144. Lower is better for these three metrics. Candidate detail energy also fell from 0.31256 to 0.29517 relative to the reference's 1.0.
- The source clips are excluded from the documented Lucid training inputs. The legacy remote patch bank lacks complete source provenance; patch-held-out training PSNR is not an independent result. These excerpts are now development data and must not be reused as a final untouched test set.
- The old Park Joy file is truncated (`moov atom not found`). It was excluded, not silently accepted or scored. The new corpus builder preflights every input before generating a corpus.
- Native stationary-history fix: 31 native tests pass, including a test that failed with the previous shader. On the two legacy clips, motion-on flicker falls 6.27% / 6.11% versus shipping. LPIPS changes +0.00051 / +0.00083 and DISTS +0.000058 / +0.00234; this is a stability/detail tradeoff, not a universal quality win. Crowd Run still has 14.15% more flicker than motion-off. See `native-motion-evaluation.json`.
- The current WebCodecs color contract is covered by native tests and browser policy tests (`color-contract.json`). PQ/HLG and P3 tags survive supported native metadata operations. The SDR browser enhancement path rejects HDR and explicit unknown metadata before copying pixels.

EfRLFN 4× comparison (`efrlfn-evaluation.json`): source-balanced LPIPS 0.53910 and DISTS 0.22861 versus shipping 0.45957 / 0.20856 on this short 180p→720p screen. EfRLFN has less static flicker (0.87535 versus 0.90169), but substantially less accurate fine detail (correlation 0.14523 versus 0.24498). This does not establish overall competitiveness or independence from EfRLFN training. Upstream commit: `1f7f3678f1bd7ba04ca8ccb04726eef71bf8520a`; loader strictly checks weights and records source hashes.

The subsequent causal 2× experiments are documented in `causal-training-v1.md` and `2026-architecture-direction.md`. Their 360p→720p inputs differ from the shipping 180p→720p screen, so their metric values cannot be used as a direct model ranking against shipping. The completed v1 causal model fails against Lanczos; v2 changes its spatial floor and pixel bypass while retaining the original training bank for the architecture comparison.

## Stream-context training bank

`causal-stream-bank-manifest.json` is a separate, verified bank: 168 aligned 16-frame patch sequences from 42 full-frame encoded windows, four crops per window. Its 160 training and eight validation patches keep the same source-family separation as the original bank. These crops are correlated within each window; they are not 168 independent scenes.

The builder first decodes a full reference canvas at 1280 or 1920 pixels wide, preserving aspect ratio to a multiple of eight pixels. It encodes the complete 2× downsampled sequence as H.264 or VP9 under a recorded bitrate constraint, decodes it once, then extracts aligned HR/LR crops after 8, 16 or 24 warmup frames. Codec decisions therefore see the surrounding image and prior frames. References remain compressed source masters. This bank changes several data-generation factors and is reserved for a subsequent controlled data experiment; it does not silently replace the frozen bank used by v1 and v2.

```sh
.venv-convert/bin/python Tools/experiments/build_stream_bank.py \
  --sources .build/causal-training-sources.json --out .build/causal-stream-bank-v1
```

`causal-stream-bank-validation.json` records the verified counts, bitrates, dimensions and bank hash. Every NPZ hash passes the trainer's provenance loader, and none of the new patches has zero reference variance.

## Motion-compensated scoring

The sequence evaluator accepts `--flow-width 640` to supplement static flicker and unwarped temporal residuals. Install the pinned optional dependency from `Tools/frontier_eval/requirements-flow.txt`. Reference-only [OpenCV DIS flow](https://docs.opencv.org/4.x/de/d4f/classcv_1_1DISOpticalFlow.html) supplies both directions; forward/backward disagreement, image bounds and reference photometric disagreement define visibility. The same maps and masks score every model. Output frames cannot influence correspondence or exclude their own errors.

`flow_residual_l1` measures the temporal change in reconstruction error after warping. `flow_motion_residual_l1` restricts it to motion of at least one output pixel. Coverage, valid-pair fraction and reference-warp error accompany the scores; an empty valid region returns null, not a perfect zero. This remains estimated motion at the recorded flow resolution, not ground-truth flow, and no temporal score replaces perceptual/detail checks. The short development clips still do not meet the long-duration release gate.

## Reproduce

```sh
python3 Tools/frontier_eval/fetch_sources.py --frames 16 --names four_people johnny
python3 Tools/frontier_eval/build_sequences.py \
  --source crowd_run .build/eval-sources/eval-crowd_run.mp4 \
  --source four_people .build/frontier-eval-sources/four_people.mkv \
  --source johnny .build/frontier-eval-sources/johnny.mkv \
  --out .build/frontier-sequences
.venv-convert/bin/python Tools/frontier_eval/evaluate_sequences.py \
  --manifest .build/frontier-sequences/sequences.json \
  --checkpoint shipping Model/weights/span_ch32utc.pth \
  --checkpoint perceptual .build/frontier-candidates/perceptual/span_ch32utc.pth \
  --checkpoint control .build/frontier-candidates/control-matched/span_ch32utc.pth \
  --report .build/frontier-candidates/sequence-evaluation.json
.venv-convert/bin/python -m unittest discover -s Tools/frontier_eval -p 'test_*.py'
```

To repeat the external comparison, clone the official EfRLFN repository at the recorded commit, download its linked 4× state dictionary, and add `--efrlfn efrlfn REPOSITORY WEIGHTS` to the evaluation command. Only reviewed upstream code should be loaded. The external weights load with `weights_only=True`; the original MIT-licensed source remains in the separate checkout.

The existing crowd source is a previously downloaded compressed SVT excerpt, not raw ground truth. `source-receipts.json` records the new raw-source URLs, licensing descriptions, exact excerpt hashes and lossless conversion commands. Media remain outside git. Re-encoded container byte hashes may depend on the FFmpeg version, so retain generated assets and verify their recorded hashes for exact reruns.

`perceptual-experiment.json` records checkpoint hashes, exact arguments, training code hashes and environment. Both runs started from the same shipping checkpoint and seed, used the same patch bank, optimizer schedule and 4,000-step budget; the candidate added VGG19 weight 0.025 and signed Sobel weight 0.1. Checkpoints and logs remain available locally and under `C:/lucid/frontier-perceptual-20260904` remotely. No existing GPU job was displaced. GPU utilization returned to 0% after the two runs.

## Remaining release requirements

These gates define work still required; none should be inferred to pass from the short screen above.

| Requirement | Evidence required | Current state |
|---|---|---|
| Independent quality | Freeze a source-family-disjoint final set before selecting models; include live action, animation, text, faces, high motion, gradients, several codec/bitrate conditions; ≥8 sources and ≥300 consecutive frames per condition. A replacement must improve both source-balanced LPIPS and DISTS by ≥3%, with no source worsening either >2%, while retaining detail correlation within 0.01. Confirm with blinded visual review. | Eight-source native evaluation improves perception but fails fine-detail guards; subsequent development candidates also fail. Fresh long release footage and blinded review remain. |
| Temporal stability | Reference-defined static flicker and flow-compensated residual with occlusion handling; no source >2% worse than shipping, alongside the spatial gates. Include cuts, seeking, subtitles and low-contrast motion. | Static/noise, translation and seek/cut native regressions exist; legacy native ablation shows 6% less flicker than shipping, but high-motion still loses to motion-off. |
| Native and browser performance | On the target M4 Pro, measured installed-extension capture-to-draw p95 ≤50 ms and ≥95% of source cadence over 10 minutes for 360p/480p 30 fps; 360p60 must deliver ≥57 fps. Include queue depth, drops, warmup and steady memory. | Direct surface route passes 360p60 installed-Chrome draw gate: 59.47 fps / 40.60 ms p95 over 621 s. A second 620 s source-PTS gate passes at 58.38 fps / 40.90 ms. Broader resolutions, Safari and full browser memory checks remain. |
| Resolution coverage | Efficient, useful 720p→1440p and 1080p→2160p routes at ≥95% of 30 fps cadence, with quality gains over native interpolation. Measure the actual intended routes before admitting them. | Current shipping guard covers 144–480p SDR. |
| Color | Native buffer tests plus end-to-end Rec.709/sRGB/P3 chart comparisons; HDR either validated end-to-end or explicitly declined. Preserve range, transfer and matrix consistently. | Metadata and rejection tests pass; image color charts remain. |
| Native experience | Verify capture admission, comparison, Off, reconnect, unsupported sources, permissions and keyboard/accessibility flows in the actual app and both companions. | Improved baseline preserved; final interactive audit remains. |
| Delivery | Clean release build and regression suite, correct model checksums and extension parity, install/launch smoke and verified package receipts. Signing/notarization status must be stated accurately. | Prior local ad-hoc package exists; updated release remains. |
| Competitive value | Run licensed public baselines on identical sources; report platform/runtime constraints and training overlap limitations. Native RTX comparison needs an available SDK and a reproducible harness. | Published NanoVSR (2026) scored with official and causal contexts; both fail detail gates. PiperSR native timing and earlier EfRLFN quality are recorded. RTX and broader matched native comparisons remain. |

Verified research links: [EfRLFN official implementation](https://github.com/EvgeneyBogatyrev/EfRLFN), [EfRLFN paper](https://arxiv.org/html/2602.11339v1), [SPANV2](https://arxiv.org/html/2604.03198v1), [August 2026 WebCodecs draft](https://www.w3.org/TR/2026/WD-webcodecs-20260827/), [Metal 4 WWDC26](https://developer.apple.com/videos/play/wwdc2026/359/). Publication results are not Lucid results.
