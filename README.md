<p align="center">
  <img src=".github/assets/lucid-logo-dark.svg" alt="Lucid — sharper browser video on Apple silicon" width="320">
</p>

<p align="center">
  <a href="#requirements"><img alt="macOS 26+" src="https://img.shields.io/badge/macOS-26%2B-0b0e14?style=flat-square"></a>
  <a href="#requirements"><img alt="Apple silicon" src="https://img.shields.io/badge/Apple%20silicon-required-5b8cff?style=flat-square"></a>
  <a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/license-MIT-8b5cf6?style=flat-square"></a>
</p>

Lucid makes low-bitrate video in your browser look better, and it does it where the
video already is. There is no separate window, no player to switch to and nothing
pasted over the top of the page: the enhanced picture is drawn into the page's own
video box, so it scrolls, clips and stacks exactly like the video did.

It is built for the ordinary case that makes streaming look bad — a 240p to 480p
stream stretched across a large Retina window — and it runs on Apple's hardware
video scaler plus a small Metal pipeline, at source frame rate.

---

## What it actually does

A frame makes this trip, all of it on-device:

1. **The page hands over its decoded frames.** A companion extension pulls real
   decoded `VideoFrame`s out of the page and sends them to the app. This is the
   video's own pixels, not a screenshot of them.
2. **Clean-up at source resolution.** Deblocking, dering, debanding and temporal
   filtering all run *before* upscaling. Every one of these algorithms is
   calibrated on a one-pixel artifact; after a 4× upscale a block edge is a
   twelve-pixel ramp and none of the tests fire. It is also sixteen times fewer
   pixels.
3. **Apple's scaler.** `VTLowLatencySuperResolutionScaler`, 2× per pass, chained
   for 4×, tiled when the frame is larger than one session accepts.
4. **Grade and detail.** Contrast-adaptive sharpening at the right radius, a tone
   grade, and perceptual saturation in Oklab.
5. **Back into the page**, at exactly the size the video box occupies on screen.

The last step is the awkward one. Most large sites — YouTube among them — set a
Content-Security-Policy that forbids connecting to `ws://127.0.0.1`, and that
policy binds the page's content scripts too, so a canvas in the page cannot
receive anything. Lucid draws from an iframe at the extension's own origin
instead, which is governed by the extension's policy rather than the page's. The
iframe is still an ordinary element in the page's DOM, so it scrolls, clips and
stacks like the video does. One path, every site.

### Stages you can turn on and off

| Stage | Default | What it is for |
|---|---|---|
| Chroma siting | **on** | H.264/VP9 4:2:0 is left-sited. Untagged buffers get read as centre-sited, which shifts colour half a luma pixel — two whole pixels after a 4× upscale. |
| Temporal (neighbourhood clip) | **on** | Karis/Playdead-style history clamping. Steadies compression noise. Fails to softening rather than ghosting. |
| Oklab saturation | **on** | Y′CbCr is non-constant-luminance, so a naive chroma multiply makes saturated red measurably *brighter*. Doing it in Oklab does not. |
| Blind H.264 loop filter | off | The normative deblocking filter, run without a bitstream. |
| CDEF dering | off | AV1's directional deringer. Picks its own strength from decoded-pixel variance. |
| Debanding + grain | off | Stochastic debander for flat-gradient banding, plus dither to cover the residual. |

## Requirements

- **Apple silicon.** `VTLowLatencySuperResolutionScaler` is not available otherwise.
- **macOS 26 or later.**
- Chrome, Edge or Safari, with the companion extension in `BrowserExtension/`.

## Install

Download the latest `Lucid-x.y.z.dmg` from
[Releases](https://github.com/braedonsaunders/lucid/releases), drag Lucid to
Applications, and open it. It asks for Screen Recording the first time — it needs
that to place the enhanced picture over the video, and it records nothing.

Then load the browser companion: open `chrome://extensions`, turn on Developer
mode, choose **Load unpacked**, and pick the `BrowserExtension` folder from a
checkout of this repository. (It is not on the Chrome Web Store yet.)

### Building it yourself

```bash
git clone https://github.com/braedonsaunders/lucid.git
cd lucid
Tools/run-poc.sh        # builds, signs and launches for development
Tools/release.sh 0.3.0  # builds a signed .dmg
```

`run-poc.sh` signs with a stable identity on purpose. macOS ties the Screen
Recording grant to the code signature, so an ad-hoc signature — which changes on
every build — makes the system ask again each time you rebuild. If it still nags,
`tccutil reset ScreenCapture com.braedonsaunders.lucid` and grant it once more.

## Using it

Click the aperture in the menu bar. The panel stays open while you adjust it,
and everything takes effect on the next frame:

- **On/off**, and a line saying what it is doing.
- **Quality** — Off, Subtle, Standard, Strong. This is the control that matters.
- **Adjustments** and **Stages**, collapsed, for when you want them.

Right-click the menu bar item for a plain menu instead.

Lucid has no Dock icon on purpose. A Dock icon would make it a regular app, and
activating a regular app while you are watching something full screen throws you
out of that Space — the menu bar hides again and the panel never appears.

## Status, honestly

Verified working end to end on **YouTube** and **Vimeo**. Both were driven through
a real browser with the extension loaded and confirmed from the app's own logs.

What does not work, and why:

- **DRM video** (Netflix, Disney+, and anything else using Widevine) is impossible.
  The frames are never available to the page, so there is nothing to read.
- **One video at a time**, top frame only. No iframes, no picture-in-picture, no
  CSS-transformed players.
- **The loopback bridge is not authenticated.** Any local process can connect to it.
  That is acceptable for a development tool and is not acceptable for distribution.
- **Performance has only been measured on one machine.** On an M4 Pro, 640×360 →
  2560×1440 in 3 tiles with the shipping defaults runs at source rate, about 10 ms
  a frame. `Lucid/Resources/DeviceCapabilities.json` records that single measurement
  and marks every other chip unmeasured. Nothing in the code adapts by chip yet.

## Development

```bash
# Offline engine comparison against ground truth
Lucid.app/Contents/MacOS/Lucid --bench input.mp4 reference.mp4 outdir 3 6
python3 Tools/score.py outdir

# Parameter sweeps
python3 Tools/band_sweep.py

# Unit tests
xcodebuild -project Lucid.xcodeproj -scheme Lucid -configuration Debug \
  -derivedDataPath .build/DerivedData test -only-testing:LucidTests \
  CODE_SIGNING_ALLOWED=NO
```

The test clips are not in the repo — they are re-encodable from public sources
(Big Buck Bunny and the Sintel trailer, both Blender Foundation, CC-BY) and the
originals are large. `TestSite/lab.html` expects them in `TestSite/`.

A note on measuring quality: fidelity metrics will mislead you here. Measured
against ground truth, plain bicubic correlates *better* with the truth at every
spatial band than either Apple's scaler or this pipeline. Perceptual super-resolution
synthesises plausible detail; it does not recover real detail. Judge changes by eye
on real clips, and use the bench to catch regressions rather than to rank looks.

## Licence

**MIT.** See [LICENSE](LICENSE).

## Credit where it is due

The interesting parts of the pipeline are the community's work, not mine, and
they are all implemented here from their published descriptions:

- **CDEF**, the directional deringing filter, from AV1 — Midtskogen & Valin.
- **Temporal accumulation with neighbourhood clipping** — Karis's TAA work and
  Playdead's shipped implementation.
- **Stochastic debanding** — the approach used by libplacebo and mpv.
- **Contrast-adaptive sharpening** — the lobe from AMD's FidelityFX CAS.
- **Oklab** — Björn Ottosson.
- **The H.264 deblocking filter** — ITU-T H.264, run blind.

Apple's `VTLowLatencySuperResolutionScaler` does the upscaling itself.
