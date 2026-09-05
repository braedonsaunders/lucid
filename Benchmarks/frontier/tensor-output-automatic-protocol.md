# Automatic tensor-output admission, 2026-09-05

This integrates the previously measured full-4x output-boundary optimization into
normal app operation. It does not retrain reconstruction or claim a perceptual
quality improvement. The shipping checkpoint remains
`fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65`.

## Admission and recovery

Automatic selection requires M4 Pro, macOS 26.5.1, CPU+GPU placement and no model
override. Other backends keep the original image model. The explicit measurement
override remains separate, and `LUCID_DISABLE_TENSOR_OUTPUT=1` disables automatic
selection. No 720p tier is added: its quality gate still fails.

The app packages all six original image models plus FP32 tensor alternatives at
the same geometries. Packaging verifies every source package hash and reference
geometry. Runtime admission checks input/output names, formats, geometry, tensor
type and frozen checkpoint metadata. It predicts three deterministic full-size
inputs (random RGB, color ramp, checkerboard) through both actual bundled models.
Admission requires exact BGRA bytes, matching attachments and the same non-IOSurface
backing class needed by the unchanged VideoToolbox conversion.

The image model stays loaded as a fallback. If tensor prediction or packing throws,
the same input frame is retried through the image model and that upscaler instance
stays on image output thereafter. Tests inject this failure and verify one tensor
attempt, two image predictions and identical delivered buffers. A compatibility
failure declines the optimization before playback. The three probes are a runtime
compatibility screen, not exhaustive proof over all possible images; the wider
numerical/native evidence remains in `tensor-output-protocol.md`.

## Verification contract

On the measured host, all six bundled alternatives must pass their image
comparisons. The native test suite must pass, including color, packing lifetime,
geometry, OS/device admission and recovery; packaging tests must reject a modified
package, mismatched reference geometry and a duplicate target name. Release must
build with twelve compiled packages while preserving the original six hashes.

Browser comparison uses the installed unpacked companion, headed Chrome and the
same isolated Release app for both arms. Shipping explicitly selects the image
model; candidate removes the override and uses normal runtime admission, with
Standard gain 0.75 in both. The native log must confirm admission. Short ABBA runs
use 30 samples per arm at 360p60 and 480p30. Their thresholds are duration >=30s,
distinct consecutive source-timestamp cadence >=57/28.5fps, p95 capture-to-draw
acknowledgment <=50ms, and correct Off behavior.

The subsequent 480p candidate endurance run requires >=600 measured seconds at
the same 28.5fps/50ms limits. Retaining the image fallback changes memory use;
report measured RSS rather than importing the earlier candidate-only memory
numbers. The exploratory readiness field in the first 480p ABBA report is not a startup
measurement: the stream was already enabled. It is not interpreted and was
removed from subsequent runs. Startup latency remains unmeasured.
These are local-fixture measurements, not third-party-site coverage, physical
scanout, power consumption, new quality evaluation or evidence for other Macs.

## Native and short browser results

The measured host is M4 Pro on macOS 26.5.1, build 25F80. Xcode's structured result
bundle records **46 tests passed, zero failed or skipped**; all six runtime model
comparisons ran on this eligible host. Three packaging tests and six trace-analyzer
regressions also pass. The original six model manifest entries are unchanged, and
all six tensor packages reproduce the previously recorded export bytes. Release
builds with all twelve compiled packages.

| Source | Image controls: distinct-PTS fps / p95 ms | Automatic candidates: distinct-PTS fps / p95 ms |
|---|---|---|
| 360p60 | 59.098 / 37.0; 59.816 / 34.4 | 59.874 / 26.7; 59.978 / 26.0 |
| 480p30 | 30.005 / 48.6; 29.999 / 38.4 | 30.001 / 30.5; 29.999 / 30.2 |

All eight runs pass their fixed short-run gates. Both arms use the same Release
binary in each comparison, and native logs confirm automatic admission in each
candidate without recovery to image output. Native RSS samples are 172–173 MiB
for the 360p candidate versus 195–197 MiB for controls, and 220–223 MiB at 480p
versus 266 MiB for controls. These are sparse native-process samples, not complete
browser or system memory accounting.

The initial 360p ABBA is also preserved under `confounded-abba360`: a Release
rebuild overlapped the first shipping control, which delivered 47.154 fps. It is
excluded from paired performance interpretation. The complete comparison was
repeated without builds or native benchmarks. Neither this exclusion nor the
later successful repeat changes any gate or erases the earlier result.

## Sustained automatic 480p result

The automatic candidate completes **622.341 seconds**, with
**29.99159 distinct consecutive source-PTS
frames/s** and **33.300 ms capture-to-draw p95**. All duration, cadence,
latency and Off gates pass. The trace contains 19,276 draws, of which
611 repeat the previous source timestamp;
they are excluded from the distinct-PTS rate. Native logs confirm admission and no
fallback. Twenty native RSS samples have first/peak/last values of
220.688/224.281/145.297 MiB.
This ten-minute run is candidate-only; the paired comparisons are the short ABBA
runs above. It is not a fresh ten-minute control or proof of all-device behavior.

The integration is admitted on the measured backend, with original weights,
supported resolutions and color pipeline preserved. Full world-class release
requirements, independent quality gains, broader hardware and Safari validation
remain open.
