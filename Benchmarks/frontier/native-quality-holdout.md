# Frozen native delivery holdout

The raw-weight candidate improved LPIPS/DISTS on the eight-source holdout but failed two detail guards. This separate native test evaluates the already frozen whole pipeline: the same candidate checkpoint, radius two and sharpening 0.4 versus shipping weights with Standard sharpening 0.75 and radius four. Native configuration was fixed before the raw-weight holdout scores were inspected. No settings are selected from this native holdout.

AVFoundation cannot decode the VP9-in-MP4 fixture on this host. The new `--pipeline-ms INPUT.json COUNT` diagnostic accepts SHA-checked, tightly packed NV12 with explicit geometry, frame count, rational cadence, Rec.709 video range and left chroma siting. It copies rows into ordinary padded Core Video planes, then runs the existing preprocessing, reconstruction, detail and sender implementations. Geometry, plane bytes, timestamps, incorrect hashes and truncated input are covered by native tests. Model selection in the shipping app is unchanged.

All 300 consecutive frames of each of the 32 source/codec/bitrate conditions traverse the pipeline. Frame indices 0, 10, …, 290 are exported, including startup frames. Both arms deliver 1280×720 NV12, producing 960 packets each. Source SHA and all packet hashes are recorded. Missing source transfer/primaries tags use the frozen SDR corpus contract; absent chroma tags use Lucid's existing left-chroma policy. Explicit contradictory metadata is rejected, and each default is recorded. This does not validate other color spaces.

`decode_native_holdout.py` converts each batch with explicit Rec.709/video-range interpretation on the same Mac/FFmpeg version as the development packet screen. A checked batch output matches the previous per-packet decoder pixel-for-pixel. Output/reference PNG SHA checks avoid cross-host decoder differences during RTX scoring. The original 3% aggregate LPIPS/DISTS, 2% per-source perceptual regression and 0.01 fine-correlation limits are reused unchanged.

This is native sender-output quality, not Chrome rendering, monitor color management or sustained browser cadence. Grain phase remains fixed at zero as declared in the frozen diagnostic configuration. The earlier raw-weight fidelity failure remains recorded regardless of this test's outcome.


## Completed result: reject candidate promotion

The RTX 4080 evaluation finished with exit zero on September 5 at 04:42:05 UTC. All **960 sender outputs per arm** were scored against their frozen references. Source-balanced LPIPS decreases from **0.328772 to 0.305116 (7.20%)** and DISTS from **0.134518 to 0.120450 (10.46%)**; both improve on every source. Aggregate fine correlation changes from 0.489505 to 0.487865.

The joint gate fails: fine-correlation changes are **−0.024534 Rush Hour, −0.021680 Sunflower and −0.012385 Tractor**, exceeding the predeclared 0.01 per-source loss limit. Five other source detail guards pass. `native-quality-holdout-evaluation.json`, `native-quality-holdout-gate.json` and `native-quality-holdout-result.json` retain the complete scores, hashes and exit receipt. Shipping weights and profile defaults remain unchanged. Better perceptual metrics alone do not justify promotion.

The capture ran with exporter source committed in `b1e07c1`. The later exporter additionally checks Core ML image input/output geometry (640×360 to 2560×1440 for shipping, 1280×720 for candidate) before capture. The completed packages were independently checked against those dimensions; the later guard is not retroactively attributed to the capture command's source hash.
