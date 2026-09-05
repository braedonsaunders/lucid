# Native deployment and competitor follow-up, 5 September 2026

[Apple's WWDC26 Core AI authoring session](https://developer.apple.com/videos/play/wwdc2026/325/) documents PyTorch export, hardware specialization, intermediate-tensor comparison and embedded custom Metal kernels. These provide a concrete future route for diagnosing precision divergence and fusing inference operations. This checkout's host has macOS 26.5.1, SDKs through 26.2 and no `/System/Library/Frameworks/CoreAI.framework`; no Core AI execution was measured here. Existing supported Core ML execution remains the runtime under test. No OS or SDK upgrade was performed.

[PiperSR's published repository](https://github.com/ModelPiper/PiperSR) advertises a compact 2× ANE-targeted model. Its GitHub commit `f164924de22e52a91177b05897bc6cc9b05a0c07` contains a package manifest and weights but omits the referenced `model.mlmodel`, so that checkout cannot independently instantiate the advertised package. The author's [Hugging Face release](https://huggingface.co/ModelPiper/PiperSR-2x), pinned to `8daecfccbbe023de6580e7eecbff3d44a51d0b13`, includes complete packages. The generic package is actually a fixed 128×128 image model. The separate video package enumerates 360p, 480p and 720p tensor shapes; it is the appropriate runtime baseline. Do not substitute a small image benchmark for video coverage.

The model weights are attributed to **PiperSR by Ben Racicot / ModelPiper** and released under [CC BY 4.0](https://github.com/ModelPiper/PiperSR/blob/master/MODEL_LICENSE). The research checkout does not import its separately licensed application source. The benchmark uses Lucid's independent Core ML harness. Requested `CPU_AND_NE` does not by itself prove zero CPU fallback. Published performance and quality claims remain distinct from our local measurements.

## Independent M4 Pro result

The complete video package was tested through Core ML with ten warmups and 40 measurements per configuration. Four deterministic source tensors rotate across predictions; compute-unit order is shuffled within each iteration. Output geometry and finiteness are checked every time. Compilation, preprocessing and display are excluded. This measures native inference and output-array materialization, not end-to-end FPS or image quality.

| Source → output | CPU+ANE mean/p95 | CPU+GPU mean/p95 | ALL mean/p95 |
|---|---:|---:|---:|
| 640×360 → 1280×720 | 19.15 / 22.67 ms | 34.12 / 45.97 ms | 38.24 / 71.91 ms |
| 854×480 → 1708×960 | 31.19 / 41.02 ms | 60.62 / 71.55 ms | 72.33 / 112.95 ms |
| 1280×720 → 2560×1440 | 67.33 / 77.00 ms | 134.38 / 141.30 ms | 139.69 / 216.07 ms |

CPU+ANE was the best requested configuration for this package. Lucid's separately measured shipping graph takes about 9.75 ms at 360p and 35.97 ms at 720p on CPU+GPU, but produces 4× output through a different image interface. These separate runs are not a matched quality, power or delivered-playback comparison. PiperSR's architecture is not a demonstrated throughput upgrade for this host. The subsequent matched quality result is recorded below.

`pipersr-native-profile.json` and `pipersr-download-receipt.json` contain the package hashes, all measurements, requested compute units and observed system load. The benchmark did not modify or redistribute the model package.

## Matched native quality outcome

The complete video package has now processed the same frozen 48 full-frame development pairs at 360p→720p using CPU+ANE. Its direct RGB8 outputs score **4.98% worse LPIPS and 0.81% worse DISTS** than shipping's declared 4× RGB8-to-2× presentation, and lose too much fine correlation on all three sources. It fails the unchanged joint spatial gate; the package remains an external baseline. These limited codec-degraded scenes do not establish a general model ranking.

The initial MPS scoring differed from cached CUDA interpolation controls by up to 0.000115 LPIPS and 0.0000283 DISTS, beyond the predeclared 1e-5 comparison tolerance. Therefore shipping was recomputed on MPS, where all interpolation controls reproduce exactly. No threshold was relaxed. `pipersr-native-quality-gate.json`, both raw reports and the hashed native-output manifest provide the evidence; `pipersr-quality-protocol.md` declares preprocessing and limitations.
