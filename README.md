<p align="center">
  <img src=".github/assets/lucid-logo-dark.svg" alt="Lucid — sharper browser video on Apple silicon" width="320">
</p>

<p align="center">
  <a href="#requirements"><img alt="macOS 26+" src="https://img.shields.io/badge/macOS-26%2B-0b0e14?style=flat-square"></a>
  <a href="#requirements"><img alt="Apple silicon" src="https://img.shields.io/badge/Apple%20silicon-required-5b8cff?style=flat-square"></a>
  <a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/license-MIT-8b5cf6?style=flat-square"></a>
</p>

**Video super-resolution for Apple silicon — the thing RTX Video Super Resolution
does for NVIDIA cards, for Macs.**

Lucid makes low-bitrate video in your browser look better, and it does it where the
video already is. There is no separate window, no player to switch to and nothing
pasted over the top of the page: the enhanced picture is drawn into the page's own
video box, so it scrolls, clips and stacks exactly like the video did.

It is built for the ordinary case that makes streaming look bad — a 144p to 480p
stream stretched across a large Retina window — and it runs a trained
super-resolution network plus a small Metal pipeline, at source frame rate.

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
3. **SPAN, 4×, in Core ML.** A ~1M-parameter super-resolution network
   in Core ML, trained on real codec degradation — x264 and VP9 at streaming
   bitrates — rather than on the bicubic downsampling every stock checkpoint
   assumes. Its trunk runs at quarter area, which is what makes 4× fit inside a
   frame. Where no model covers a source, Lucid leaves the frame alone; see
   [When it runs](#when-it-runs).
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
| Temporal (neighbourhood clip) | **on** | Karis/Playdead-style history clamping. Removes about a twelfth of the frame-to-frame shimmer on moving content, and fails to softening rather than ghosting. |
| Oklab saturation | **on** | Y′CbCr is non-constant-luminance, so a naive chroma multiply makes saturated red measurably *brighter*. Doing it in Oklab does not. |
| Blind H.264 loop filter | off | The normative deblocking filter, run without a bitstream. |
| CDEF dering | off | AV1's directional deringer. Picks its own strength from decoded-pixel variance. |
| Debanding + grain | **on** | Stochastic debander for flat-gradient banding, plus synthetic grain to cover the residual. The grain is scaled by how much fine detail the frame already carries — a fixed amount is proportionally enormous on a smooth scene and marginal on a textured one. |

## How this compares to RTX Video Super Resolution

The goal is the same and the approach has to be different.

|  | RTX VSR | Lucid |
|---|---|---|
| Where it runs | Inside the GPU driver and the browser's video compositor | A separate app, plus a companion extension |
| What does the upscaling | An NVIDIA-trained model | SPAN in Core ML, then a Metal pipeline |
| Controls | Quality 1–4 | Quality: Off, Subtle, Standard, Strong |
| Declines when | The video already matches or exceeds what is displayed | Same — 720p source and above, or under 1.15× stretch |

The important difference is the first row. NVIDIA works with Chrome and Edge, so
VSR sits inside the browser's own compositing path and has nothing to reach
around. macOS offers no equivalent hook, so Lucid takes the page's decoded frames
through an extension, enhances them, and draws the result back into the page from
an iframe at the extension's origin. Getting that to behave like part of the page
— scrolling, clipping and stacking correctly, on sites whose security policy
blocks the obvious route — is most of what this project is.

Lucid is not affiliated with or endorsed by NVIDIA; RTX Video Super Resolution is
their trademark, named here only to say plainly what this is.

## When it runs

| Source | | Per frame |
|---|---|---|
| Below 128×72 | left alone — too little to reconstruct from | |
| **144p** (256×144) | **enhanced** | 2.9 ms |
| **180p** (320×180) | **enhanced** | 3.7 ms |
| **240p** (426×240) | **enhanced** | 5.8 ms |
| **270p** (480×270) | **enhanced** | 6.8 ms |
| **360p** (640×360) | **enhanced** | 10.7 ms |
| **480p** (854×480) | **enhanced** | 17.8 ms |
| 720p and above | left alone | 37 ms — past the frame |

Those are whole-pipeline figures — source clean-up, the model, and the detail
pass — measured end to end on an M4 Pro, not model time in isolation. A 30fps
frame is 33.3 ms, so the largest source Lucid takes uses about half of one.

The ceiling is measured rather than chosen. At 1280×720 the pipeline costs 37 ms,
and the model alone accounts for 33.4 of it, so there is no combination of stages
that brings 720p inside a frame on this graph.

The ceiling is a frame budget, not a resolution: one frame at 30fps, less what
the rest of the pipeline costs. Above it Lucid does nothing at all rather than
reaching for a weaker upscaler, because the weaker upscaler measured worse than
leaving the frame alone.

**The target is the window Microsoft Edge uses: enabled below 720p, and Lucid
now reaches it.** Edge got there independently, from inside a browser
compositor, and NVIDIA's RTX VSR declines on the same principle — below 720p is
where a source is genuinely short of what the display is showing, and above it
there is progressively less to recover.

Reaching it took a rearranged network rather than a smaller one. Folding a 2×2
block into the channel dimension before the trunk runs the 18 trunk convolutions
at quarter area and lets the head do ×8 instead of ×4, which is worth more than
any channel count: the trunk is activation-bandwidth bound rather than MAC
bound, so the lever is pixels, not channels. See `LearnedUpscaler.swift` for the
arithmetic.

Every input width in that table is a multiple of 16, and that is the most
important thing about the ladder. Both compute engines tile along width, and an
unaligned width costs an entire extra pass:

| | native width | padded to a multiple of 16 |
|---|---|---|
| 240p | 426×240 — **13.4 ms** | 432×240 — **4.4 ms** |
| 480p | 854×480 — **50.8 ms** | 864×480 — **14.7 ms** |

Three times the cost for the same picture, and at 480p it is the difference
between fitting a frame and not fitting one at all. Height alignment buys
nothing — 480×272 measured *slower* than 480×270, by exactly the two extra rows.

So the two real streaming sizes that are not aligned — 426 and 854 wide — run
models converted at the next multiple of 16, and the frame is stretched that
last 1.4% on the way in. The stretch itself is free: measured against a model
converted at the exact width, it costs 0.09 ms. The page draws the result into
the video box, which has the true aspect, so nothing is left distorted.

It also stays out of the way unless the video is actually being stretched: the
player has to be at least 1.15× the decoded width in real pixels before Lucid
starts, and it keeps going down to 1.0× once running so it does not flicker on
and off as a window is resized.

## Requirements

- **Apple silicon.** The pipeline is Core ML and Metal throughout, and the frame
  budget assumes an Apple GPU and unified memory.
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

## Scope

Lucid works on **YouTube**, **Vimeo**, and any site that serves ordinary
`<video>`. It enhances one video at a time — the largest playing one on the
page.

It does not touch **DRM-protected video**. Netflix, Disney+ and anything else
using Widevine decode inside a protected path that the page itself never sees,
so there are no frames to read. This is a property of DRM, not a limitation to
be worked around, and it applies equally to any tool that is not the graphics
driver.

Picture-in-picture windows, videos inside cross-origin iframes, and players
under CSS transforms are outside the current scope and are left alone.

## Development

```bash
# Score a model on footage it has never seen, against a Lanczos anchor
.venv-convert/bin/python Tools/eval_checkpoint.py \
  --corpus .build/eval-corpus Model/weights/span_ch32u.pth

# Render frames from a clip for side-by-side comparison
Lucid.app/Contents/MacOS/Lucid --bench input.mp4 reference.mp4 outdir 3 6

# Check the model ladder is identical everywhere it is named
Tools/check-ladder.sh

# Unit tests
xcodebuild -project Lucid.xcodeproj -scheme Lucid -configuration Debug \
  -derivedDataPath .build/DerivedData test -only-testing:LucidTests \
  CODE_SIGNING_ALLOWED=NO
```

The test clips are not in the repo — they are re-encodable from public sources
(Big Buck Bunny and the Sintel trailer, both Blender Foundation, CC-BY) and the
originals are large. `TestSite/lab.html` expects them in `TestSite/`.

`Tools/eval_checkpoint.py` is the scoreboard. It judges perceptually — LPIPS,
DISTS and no-reference IQA — on footage the model has never seen, against a
plain Lanczos anchor, and reports per resolution tier rather than averaging
across them.

Distortion metrics are guards, not objectives. Correlation with a reference
answers whether the output is still the same scene; band energy answers whether
enough detail is there. Neither can rank reconstructions, because a plain
bicubic upscale beats every perceptual super-resolution on both. `Tools/score.py`
and `Tools/band_sweep.py` compute distortion figures and are kept for that
narrow purpose.

`Tools/check-ladder.sh` verifies that the model ladder is identical in all four
places that name it — `LearnedUpscaler.variants`, `release.sh`, `run-poc.sh` and
`.gitignore`. Run it after any change to the model set.

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

The upscaler is **SPAN** — Wan et al., *Swift Parameter-free Attention Network
for Efficient Super-Resolution* — rearranged to run its trunk at quarter area,
and trained here from scratch on real codec degradation rather than on bicubic
downsampling. The architecture is Apache-2.0; see [NOTICE](NOTICE).

Apple's `VTLowLatencySuperResolutionScaler` is not used. It synthesises a
mottled texture in foliage that is not present in the source, and on a 1080p
reference it scores below a plain Lanczos upscale. Lucid declines to enhance a
source it has no model for, rather than falling back to it.
