# Native M4 Pro browser video enhancer proof of concept

## Current result

The corrected proof of concept runs entirely on this MacBook Pro and no longer
captures or scales the whole browser window. A small Chrome/Safari companion
identifies the largest visible `<video>`, publishes its exact window-local crop
and intrinsic decoded dimensions, and the native app captures only that rectangle.

Every decoded source pixel is retained. Frames that exceed Apple's 960×960
per-session limit are divided into the smallest supported grid, processed by
multiple VideoToolbox super-resolution sessions, and reconstructed at 2× with
32-pixel context overlap. Only each tile's center is retained, which removes the
model-edge seam without blending or shrinking source pixels.

Measured locally on September 1, 2026:

| Source → output | Native tiles | Warm processing time | Sustained output |
| --- | ---: | ---: | ---: |
| 1280×720 → 2560×1440 | 2 | about 6.2 ms/frame | about 57 fps |
| 1920×1080 → 3840×2160 | 6 | commonly 14–18 ms/frame | commonly 57–59 fps |

Test hardware is an Apple M4 Pro with 12 CPU cores, 16 GPU cores, and 24 GB of
memory. The 720p run is visible in
[`output/playwright/tiled-720p-overlap-live.png`](output/playwright/tiled-720p-overlap-live.png)
and the native 1080p-to-4K run in
[`output/playwright/tiled-1080p-to-4k-live.png`](output/playwright/tiled-1080p-to-4k-live.png).

## Open-model evaluation

Apple's low-latency model is exceptionally fast and stable, but its restoration
is visually conservative. Several open models were therefore tested natively on
the same M4 Pro rather than selected from model-card claims alone.

| Model | Route | 1280×720 latency | Finding |
| --- | --- | ---: | --- |
| Apple low-latency SR | VideoToolbox, 2 overlap tiles | ~6.2 ms | Real-time baseline; modest detail recovery |
| PiperSR 2× | Core ML, Neural Engine | 64.5 ms median | 15.5 fps, but the published checkpoint introduced color/detail errors in the controlled test |
| Real-ESRGAN `general` | Core ML, GPU, 6 tiles | 0.94 s | Too slow live, but the best tested restoration quality |
| Real-ESRGAN `x4plus` | Core ML, GPU | slower still | Not a live-browser candidate |

A controlled 720p→1440p test was generated from a known 1440p reference. Plain
bicubic scored 41.9706 dB PSNR / 0.9760 SSIM. The small Real-ESRGAN general model
scored 42.3188 dB / 0.9849, so it is a defensible quality teacher, but at roughly
one frame per second it cannot be the runtime. The tested PiperSR checkpoint was
faster but worse than bicubic on this scene, so it is not bundled merely to make
the app say "AI".

Running PiperSR jobs on the Neural Engine and GPU concurrently did not add their
throughput. Shared memory pressure reduced aggregate throughput to 11.8 fps,
versus 15.5 fps on the Neural Engine alone.

The useful next engine is therefore a distilled temporal hybrid:

1. Keep Apple's stable full-rate 2× reconstruction as the base image.
2. Distill the restoration behavior of Real-ESRGAN into an ANE-native student
   trained on codec, blur, ringing, and noise degradations rather than clean
   bicubic images.
3. Enhance selected keyframes at roughly 15 fps.
4. Use Apple hardware optical flow/frame interpolation to synthesize the missing
   enhanced frames with a small bounded playback delay.
5. Apply learned detail primarily as a motion-gated luminance residual so the
   student cannot shift hue, exposure, or broad gradients.

This follows the same high-level insight as NEMO-style mobile video SR: expensive
neural work belongs on selected frames and its benefit should be propagated.

## Browser integration

`BrowserExtension/` is one Manifest V3 source shared by Chromium and Safari. It
reports geometry only; video pixels remain in the native ScreenCaptureKit/Core
Video path.

- Chrome/Edge: load `BrowserExtension/` as an unpacked developer extension.
- Safari: the generated macOS Safari Web Extension project is
  `SafariCompanion/Lucid Companion/Lucid Companion.xcodeproj`.
  It builds successfully. Safari still requires the user to install/enable the
  extension and grant website access.

The native app now accepts tagged Chrome, Safari, or Edge windows and refreshes
the intrinsic source dimensions each time capture starts. It refuses to fall
back to whole-window capture when metadata is missing.

## Run the local test

Requirements are macOS 26, Xcode 26, Screen Recording permission, and Chrome.

```sh
cd /Users/braedonsaunders/Documents/Code/lucid
./Tools/run-poc.sh
```

The default page is the 1280×720 test. Open
`http://127.0.0.1:8765/?source=1080` for the lossless 1920×1080→3840×2160 test.

## DLSS 5 / Wine verdict

`DLSS5oneclick` does not contain DLSS source or a portable model. It orchestrates
a leaked closed `nvngx_dlssnr.dll`, ReShade, Direct3D hooks, and NVIDIA NGX driver
interfaces. Wine, D3DMetal, or a Metal compatibility layer can translate graphics
APIs, but cannot supply NVIDIA's signed NGX runtime, RTX tensor implementation, or
closed weights on Apple silicon. Reverse-engineering the installer would reveal
integration glue, not a model that can be legally or technically ported.

The transferable idea is temporal reconstruction, not binary emulation.

## Remaining product work

- Train/distill and validate the temporal student across natural video, faces,
  animation, text, low bitrate streams, and scene cuts.
- Integrate the student and bounded-delay temporal scheduler in Swift/Core ML.
- Replace the separate proof window with a polished click-through presentation
  mode while avoiding WindowServer's observed capture throttling/feedback loop.
- Handle moving/resizing/full-screen videos continuously rather than only at
  capture start.
- Package, sign, and notarize the native app plus Safari extension.
- DRM-protected video may intentionally be black to ScreenCaptureKit and is not
  expected to be bypassed.
