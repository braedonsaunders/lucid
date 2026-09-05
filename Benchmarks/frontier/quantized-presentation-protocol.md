# Preserve quantized presentation while avoiding a 4× image output

The area-folded output head saves native work but differs from shipping's clipped RGB8 bicubic presentation. This experiment retains the exact shipping reconstruction weights and inserts quantization plus separable bicubic 2:1 downsampling inside the graph. It exports a 2× image directly, avoiding a large 4× image boundary and later resampling. It does not reduce the reconstruction convolution's arithmetic, claim a new resampling algorithm, or alter any training checkpoint.

The fixed Catmull-Rom antialias coefficients are `[-3,-9,29,111,111,29,-9,-3]/256` in each direction. Normalize boundary weights over valid pixels. Clamp and nearest-round the 4× reconstruction to RGB8, apply horizontal filtering with RGB8 rounding/clipping, then vertical filtering with the same rounding/clipping. Small random and constant-color tests include every border: maximum PIL difference is at most one level and mean difference below 0.01. No input/reference-dependent output selection is used.

Native conversion keeps shipping's mixed precision but preserves quantization, normalization and the small separable filters in FP32. The first 360p random-input check passes the existing maximum/mean error limits (2.0/0.06296 RGB levels versus this graph's FP32 reference). Its interleaved short timing run was noisy under a Mac load average above 26: shipping 4×, area-folded 2×, and quantized bicubic 2× means were 19.32/13.23/11.51 ms. These are not accepted as sustained throughput or a stable ranking. The original raw profile and executed profiler snapshot are retained; later profiler edits add explicit variant metadata and correct the weight label when combined with an optional trained checkpoint.

Before measuring source-image quality, fix a **performance-only noninferiority screen**: both native models use CPU+GPU on the same 48 full-frame development inputs. Compare shipping's 4× RGB8 image bicubic-resized to 2× with the new direct 2× RGB8 image. Require ≤0.5% source-balanced and per-source LPIPS/DISTS regression and ≤0.002 per-source fine-correlation loss. Preserve matched interpolation controls. This checks a presentation transform of unchanged weights; it does not replace or satisfy the separate ≥3% quality-improvement gate for a trained replacement. A useful outcome still requires stable native delivery timings and native postprocessing equivalence before product integration. No 720p coverage or end-to-end cadence is implied by graph export.

## Native quality screen

All 48 native full-frame pairs pass the predeclared presentation noninferiority screen. Source-balanced LPIPS and DISTS distances increase by **0.0386% and 0.0417%**, well within the 0.5% limits; every per-source perceptual and fine-correlation guard passes. The same MPS scorer evaluates both native outputs and the shared interpolation control in one run. `quantized-presentation-native-quality.json` records all model package hashes, output PNG hashes and per-frame scores. This preserves shipping quality on this limited screen; it is not a trained-quality improvement or release admission.

## Native Swift stage measurement

The rebuilt Release application's existing offline timing path completes 60 paired measurements after ten warmups, consuming full NV12 packets at the same 1280×720 delivered size. Shipping's graph + native downscale/packet path takes **19.381 ms mean / 25.781 ms p95**; the quantized direct 2× graph + packet path takes **9.954 / 12.734 ms**. Mean stage time falls **48.6%** in this run. Graph means are 15.793/9.484 ms and packet means 3.587/0.470 ms. Both packets contain 1,382,598 bytes.

These are synthetic BGRA input measurements on a busy host, excluding capture, decode, detail postprocessing, network and rendering. The shipping sender's native downscaler differs from the PIL bicubic adapter used for the quality screen; the two results therefore do not prove native delivered quality parity. `quantized-presentation-native-provenance.json` pins the actual rebuilt binary and relevant Swift source. All raw samples remain in `quantized-presentation-native-swift360.json`. Product integration, native postprocessing/packet quality and stable long-duration browser cadence remain required.

## Native integration screen, frozen before packet results

Run all 16 consecutive frames of each of the same 12 development codec conditions through the real native preprocess/model/detail pipeline. Export frames 0/4/8/12 through the actual sender in both arms. Keep Standard settings and sharpness 0.75 in both; the existing pipeline derives radius from actual model scale (4 versus 2). Use fixed grain phase zero as in prior diagnostics. No tuning sweep, temporal/filter change, or new source selection is permitted in this screen. Apply the same 0.5% perceptual / 0.002 fine-correlation noninferiority limits to decoded NV12 packets. A failure keeps the route experimental despite the raw-native quality and stage-time results.

The initial packet export had correct effective settings but included unused historical selection metadata copied from an older configuration. A fresh export with an explicit minimal configuration provides the canonical receipt. All effective tuning values and weights remain identical; no quality scores were computed or selected between these exports. The first export and its executed configuration remain under `.build/quantized-presentation-native-packets`.

## Full native pipeline result: current integration rejected

Standard preprocessing/detail and actual NV12 delivery fail the same noninferiority screen despite the raw-model pass. Aggregate LPIPS improves 3.76% and DISTS 0.55%, but FourPeople LPIPS regresses 6.58%, Johnny DISTS regresses 3.35%, and fine correlation falls by 0.02997/0.04010/0.03743 on CrowdRun/FourPeople/Johnny. The route remains experimental. No weight change or production switch follows the faster stage timing.

The source exposes a concrete spatial gain difference to isolate next: `gainNormalisation(radius:reference:)` multiplies the default gain by 1.5 at radius two versus 1.0 at radius four. The pipeline correctly derives geometric radius from model output scale, but retains reference radius four; nominal sharpness 0.75 therefore becomes 1.125 for this 2× model. This is a measured-code observation, not yet a proved explanation of the entire delivered-quality loss.

## Separate nominal-gain ablation

Before its outputs are inspected, fix one diagnostic intervention: candidate geometric radius remains two, while its gain reference radius is also two, making gain normalization exactly one. Keep nominal sharpness 0.75 and every tuning value, checkpoint, input and gate unchanged. Shipping remains radius/reference four and gain normalization one. This tests removal of the extra 1.5× gain, not a gain sweep or an inferred full correction. The change is opt-in to the offline diagnostic and does not modify production behavior. Verify the effective reference radius in each native process log.

For this fixed ablation, fine/mid/micro gains are zero and lobeScale is 0.3. The effective lobe sampling distance rounds to one pixel with either reference radius, so changing reference radius affects the active sharpening gain without changing this kernel's sampling offsets.

## Nominal-gain result: limited native screen passes

The fixed nominal-gain correction passes all 48 native delivered pairs under the unchanged presentation thresholds. Source-balanced LPIPS improves 4.38% and DISTS 2.88%; fine correlation improves by 0.01036/0.01582/0.02502 on CrowdRun/FourPeople/Johnny. Every per-source perceptual score improves. Shipping control scores reproduce the preceding native run exactly (maximum difference zero). This supports the gain mismatch as a cause of the preceding regression on these sources. It does not prove equivalence for every setting or scene.

The earlier 48.6% stage-time reduction excludes the detail pipeline, so it cannot yet be combined with this quality result into a measured end-to-end gain. The route remains an offline prototype pending broader native regression, sustained full-pipeline/browser timing and resolution/color checks. No production defaults or shipping weights change.
