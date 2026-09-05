# Preserve 4× processing while changing the Core ML output boundary

Frozen before the native probe, September 5, 2026. The quantized 2× route fails the wider detail screen even with nominal gain fixed. This experiment retains full 4× reconstruction geometry and leaves the subsequent NV12 conversion, detail filtering and final presentation order available unchanged. It tests whether Core ML image-output materialization can be replaced efficiently. No model training or shipping asset changes.

Export the same frozen shipping reconstruction and mixed precision three ways: existing RGB image output, FP32 tensor output and FP16 tensor output, all in 0–255 range. The sole graph intervention is output storage type. A Metal kernel reads tensor strides and packs RGB8 into a full-resolution BGRA IOSurface using fixed clamp/nearest rounding. It borrows storage through `MLMultiArray.withUnsafeMutableBytes`; GPU work completes before that closure returns. Page-aligned, page-sized storage on unified memory is wrapped directly; otherwise an explicit copy is made and reported. This does not prove that Core ML itself performed no internal copy. Apple's [documented borrowing API](https://developer.apple.com/documentation/coreml/mlmultiarray/withunsafemutablebytes(_:)) and installed macOS 26.2 SDK signatures were checked.

First require maximum RGB error ≤3 levels and mean ≤0.6 against the matching native image output for each of four deterministic synthetic inputs. Failures receive no timing admission. Then measure 60 interleaved predictions after ten warmups at 640×360, including borrowed-buffer access, any explicit copy, Metal packing and GPU completion. Compare the whole boundary time, not tensor prediction alone. Both candidates must finish with the same 2560×1440 BGRA geometry. Keep all samples and actual dtype/stride/storage-mode receipts.

A candidate needs at least 10% lower mean boundary time to justify native integration work. Freeze the fastest passing storage type before wider image/content checks; do not select per-frame outputs or tune rounding. This short probe does not establish full native delivered quality, browser speed, or a new reconstruction method. A successful boundary probe still requires identical downstream order, broad source/color regression and actual browser delivery before promotion.

## Initial boundary result

Both tensor formats pass four synthetic RGB comparisons: maximum one level, mean about 0.045. All borrowed buffers meet page-alignment requirements and use the shared-storage Metal wrapper; FP32 storage is 44,236,800 bytes and FP16 is 22,118,400 bytes. The full 2560×1440 output is retained.

Mean image-output time is 18.03 ms. FP32 tensor plus Metal packing averages 13.97 ms (22.5% lower); FP16 averages 14.25 ms. Corresponding p95 values are 28.37/25.25/29.22 ms. The high variance prevents claiming stable tail-latency improvement. Freeze FP32, the predeclared fastest passing mean, for a longer two-arm confirmation. The first three-arm order alternated forward/reverse; the confirmation uses only the frozen candidate and image control with balanced alternating order.

Before using this packing path for quality evaluation, inspect whether the remaining single-level differences arise from RGB8 half-way rounding. This is a numerical-equivalence diagnostic against the native image output, not a source-quality or sharpening-parameter sweep. No broader quality or deployment claim follows from the short timing run.

The full raw-tensor diagnostic checks 44,236,800 channel values across the four inputs. All 1,994,878 image/tensor packing differences occur away from exact half-way values; half-up and nearest-even rounding have identical mismatch counts. Therefore tie-breaking does not explain the discrepancy, and the quantizer is unchanged. Small graph/output-boundary numerical differences remain possible. The frozen FP32 route proceeds only to the declared longer boundary confirmation; source-quality admission is still required.

## Fixed FP32 confirmation

180 balanced interleaved samples after ten warmups pass the same RGB thresholds: max one level, mean 0.045. Image output averages 8.188 ms (p95 8.926); FP32 tensor plus Metal packing averages 6.018 ms (p95 6.301), a 26.5% lower mean. These timings supersede the noisy short probe for this boundary only. Proceed to measurement-only native integration with full 4× geometry, existing color conversion, Standard gain/radius4 and existing 2× sender. No shipping model change.

## Native 48-pair admission

The fixed FP32 route passes the unchanged noninferiority gates through actual native NV12 conversion, Standard radius4 detail and 2× sender. Source-balanced LPIPS is 0.093% worse and DISTS 0.254% worse; all per-source perceptual and fine-correlation limits pass. This is a small accepted numerical difference, not a quality improvement. All 48 shipping packet payloads/headers and decoded PNG hashes reproduce previous controls exactly. The 12 short paired conditions average 12.893 ms shipping and 11.378 ms tensor (11.8% lower), excluding sender/browser and too short for sustained latency conclusions. Proceed with the unchanged configuration to the frozen eight-source 960-pair regression. No quality-based tuning or selection.

The first GPU scoring launch lacked copied Python dependencies; a second exposed the missing architecture package. Both failed before scoring. The third runs the identical scorer with the full dependency tree and separately redirected output/error logs; it completed 96 scores on the RTX 4080. No model training occurred.

## 480p boundary-only check

The same frozen FP32 output at 864×480→3456×1920 passes all four RGB comparisons (maximum one level, mean about 0.045). In 180 balanced interleaved samples, image output averages 19.267 ms and tensor plus packing 14.610 ms (24.2% lower), p95 25.152/19.408 ms. Full native processing, source-quality, memory and browser gates at this resolution remain. The third exported storage type is not evaluated or selected; the original FP32 choice stays fixed.

## Wider result: frozen FP32 implementation rejected

The eight-source, 960-pair native screen completes all 1,920 CUDA scores. Aggregate LPIPS improves 0.121% and DISTS 0.036%, but OldTownCross exceeds the fixed 0.5% per-source DISTS regression limit. All fine-correlation guards pass. Therefore this implementation does not advance to browser testing or production. Its 32-condition native enhancement-stage mean is 13.994 ms shipping versus 12.552 ms candidate (10.3% lower), but that speed is insufficient for admission. All 960 shipping payload/header and decoded PNG controls match previous artifacts exactly.

The downstream RGB differences average 0.23–0.36 levels by source; the 99th percentile is three levels, with isolated maxima up to fourteen after the native color/detail pipeline. These are descriptive numerical differences, not a replacement for perceptual gates. The original same-geometry reasoning alone did not guarantee sufficiently close delivered quality. Any further work must identify a concrete numerical/ABI mechanism on controlled synthetic inputs; the failed endpoint and thresholds remain fixed. No quality-driven coefficient, checkpoint, gain or rounding sweep is authorized by these results.
