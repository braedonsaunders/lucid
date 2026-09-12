<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/assets/lucid-logo-dark.svg">
    <img src=".github/assets/lucid-logo.svg" alt="Lucid — sharper browser video on Apple silicon" width="320">
  </picture>
</p>

<p align="center">
  <a href="#build-and-browser-companion"><img alt="macOS 26+" src="https://img.shields.io/badge/macOS-26%2B-0b0e14?style=flat-square"></a>
  <a href="#build-and-browser-companion"><img alt="Apple silicon" src="https://img.shields.io/badge/Apple%20silicon-required-5b8cff?style=flat-square"></a>
  <a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/license-MIT-8b5cf6?style=flat-square"></a>
</p>

**Video super-resolution for Apple silicon — what RTX Video Super Resolution
does for NVIDIA cards, for Macs.**

Lucid makes low-bitrate video in your browser look better, and it does it where the
video already is. There is no separate window, no player to switch to and nothing
pasted over the top of the page: the enhanced picture is drawn into the page's own
video box, so it scrolls, clips and stacks exactly like the video did.

It is built for the ordinary case that makes streaming look bad — a 144p to 720p
stream stretched across a large Retina window — and it runs a trained
super-resolution network plus a small Metal pipeline, at source frame rate, entirely
on your Mac.

---

## What ships

- **2× learned reconstruction, one model.** Lucid 1.0 ships a domain-trained
  Nano-derived network with 48 channels and six blocks, trained by this project
  on codec-degraded video. Seven fixed input shapes cover 144p through 720p.
  The same model runs everywhere; device calibration selects the shapes your
  Mac can sustain. All packages are verified against `Lucid/Resources/Models.json`.
- **Grain-aware post-stages.** Debanding runs only on true quantisation
  plateaus (any neighbour more than about 1.3 levels away marks grain or
  texture and is left alone), followed by contrast-adaptive sharpening that
  cannot leave the range of the pixels it was computed from, and a tone grade.
- **Motion-aligned temporal filtering.** A bounded Metal block matcher reprojects
  previous-frame luma. Photometric confidence rejects unreliable matches;
  neighborhood clipping limits trails. Seeks, repeated timestamps, long gaps,
  and incompatible frame geometry reset history.
- **An explicit color contract.** Browser color metadata travels with the frame.
  BGRA, RGBA, I420, and NV12 are validated and normalized to video-range Rec.709
  NV12 before enhancement. PQ and HLG inputs are declined.
- **Bounded streaming.** The browser captures only for the admitted session,
  with expiring status leases and two capture credits. A single pending native
  frame bounds conversion work. Output favors fresh frames over a backlog.
- **Modern browser transport.** Chrome 148+ uses typed-array structured cloning;
  older Chrome versions use the JSON-compatible path. The authenticated drawing
  iframe receives NV12 directly over loopback WebSocket. Structured cloning
  still copies data; this is not a zero-copy browser pipeline.
- **Native controls.** Clean, Subtle, Standard, and Strong picture presets,
  adjustments, companion setup, hold-to-compare, and Launch at login.
  Click Lucid’s aperture icon in the Mac menu bar for its dropdown controls.
  Reopening Lucid reveals the same dropdown. Dismissing it keeps enhancement running.
  Turning Lucid off stops frame capture and clears the enhanced surface.
- **Presentation feedback.** Sequence numbers correlate input with browser draw
  acknowledgments. The panel reports presented frames/s and capture-to-canvas
  p95, separately from native processing time. This measures draw submission,
  not physical screen scanout.

Decoded playback drawn in the browser needs **no Screen Recording permission**
and no native window match. Screen-capture fallback is a separate path.

## Supported video

Lucid targets unprotected SDR video from 144p through 720p that is being
enlarged by at least 1.15×. It keeps an existing session down to 1.0× to avoid
toggling at the boundary. A source must fit a bundled model shape; 1080p and
above are declined.

HDR enhancement, protected playback, and universal website compatibility are
not supported promises. Lucid is an independent application; it does not
contain NVIDIA RTX software or integrate with the browser compositor at driver
level.

## How it is measured

The 1.0 model was selected after live browser review. On the fixed browser
holdout (eight clips, three frames each), it improves LPIPS by 9.28% and DISTS
by 12.02% against the previous Nano control. On a separate eight-frame lab
holdout, those distances worsen by 2.37% and 4.46%. Results vary by content;
this release does not claim that every source improves.

The production checkpoint is
`5514d2739d662c6dc92258618dbe807ac2ce95ca402a32f76839b0084f5968ba`.
All seven converted shapes are checked for output dimensions, image range and
numerical agreement with the trained PyTorch network. Native tests cover frame
integrity and the shipped model, followed by live browser playback checks.

## Build and browser companion

Requires Apple silicon, macOS 26+, and Xcode with the macOS 26 SDK.

```sh
xcodebuild -project Lucid.xcodeproj -scheme Lucid -configuration Release \
  -derivedDataPath .build/release build CODE_SIGNING_ALLOWED=NO
```

Every Xcode build verifies the source model hashes in
`Lucid/Resources/Models.json`, compiles the seven shipping models into the app
and removes anything else from the bundle. Changed models invalidate their
compilation receipt; missing or corrupt source models fail the build.

Open the app at `.build/release/Build/Products/Release/Lucid.app`.
For Chrome or Edge, open `chrome://extensions` or `edge://extensions`, enable
Developer mode, choose **Load unpacked**, and select `BrowserExtension`.
Reload your video tab. The companion is not published in a browser store.

For Safari, build the project under `SafariCompanion/Lucid Companion`, open
Lucid Companion, and enable its extension in Safari Settings. Its resources
are generated from the canonical browser implementation:

```sh
python3 Tools/sync_safari.py
python3 Tools/sync_safari.py --check
```

`Tools/release.sh 1.0.0` builds a Developer ID signed, notarized disk image with
Lucid and the Chrome/Edge companion. It requires a Developer ID Application
certificate and the `LUCID_NOTARY` keychain profile. The command verifies the
app, notarization tickets, disk image and model checksums before finishing.
`Tools/release.sh 1.0.0 --local` explicitly creates a development-signed package
with a `-local` filename; that package is not notarized for public distribution.

Copy the companion folder to a permanent location before loading it in Chrome
or Edge; do not load it from a disk image that you will eject.

## Verification

```sh
node --test Tools/stream-policy.test.js
python3 -m unittest discover -s Tools -p 'test_*.py'
zsh Tools/check-ladder.sh
xcodebuild -project Lucid.xcodeproj -scheme Lucid -configuration Debug \
  -derivedDataPath .build/tests test -only-testing:LucidTests CODE_SIGNING_ALLOWED=NO
```

The native suite exercises actual Metal kernels, moving grain, motion and scene
cuts, color conversion, malformed frame layouts, model predictions, settings
parity, and presentation metrics. GPU tests require a physical Apple silicon
Mac; hosted CI excludes that suite explicitly.

`Tools/browser_fixture.py` creates the local worker/iframe fixture. Launch a
separate test app with `LUCID_BRIDGE_PORT=47911 LUCID_TOKEN_PORT=47912
LUCID_EPHEMERAL=1` and serve the fixture over loopback HTTP. Ephemeral mode keeps
the test token and enable switch out of the normal app's persisted state.

## Training and evaluation

The minimal SPAN architecture and its Apache-2.0 license are included under
`Tools/architectures`. Training requires PyTorch, NumPy, Pillow and a
codec-degradation stream bank (`Tools/experiments/build_stream_bank.py`); the
shipping model was fine-tuned with `Tools/experiments/train_presented_detail.py`
on an RTX 4080 and converted on a Mac with PyTorch 2.14 and coremltools 9.
Older notes on how the model got here live in `Docs/history/`.

```sh
python Tools/train_span.py --init baseline.pth --bank-dir path/to/bank \
  --channels 32 --input-frames 1 --motion-temporal --temporal 0.15 \
  --fft 0.05 --steps 2000 --batch 16 --lr 0.00002 --seed 20260904 --device cuda
python Tools/convert_trained.py --weights checkpoint.pth --sizes 640x360
python Tools/eval_checkpoint.py checkpoint.pth --corpus path/to/paired/corpus \
  --report evaluation.json
```

Checkpoints preserve optimizer, scheduler, RNG states, arguments, and framework
version. This supports resuming the same environment; it does not guarantee
bitwise reproducibility across different GPU kernels or PyTorch versions.
Patch-bank validation is an internal training diagnostic, not an independent
benchmark. Checkpoints and training corpora are separate artifacts.

Tuning tools require an explicitly matched reference or a registered clean pair;
they cannot silently score an arbitrary clip against the compressed BBB video.
The experiment called `version=2` is a learned-sigmoid SPAN adaptation, **not**
the published SPANV2 architecture.

## Research basis

The implementation favors measurable real-time reconstruction over an untested
claim of generative detail. The September 2026 review considered
[NTIRE 2026 efficient super resolution](https://arxiv.org/html/2604.03198v1),
[streaming-specific training and evaluation](https://arxiv.org/html/2602.11339v1),
[StreamDiffVSR's August 2026 revision](https://arxiv.org/abs/2512.23709v3), and
[Apple's WWDC26 Metal developments](https://developer.apple.com/videos/play/wwdc2026/359/).
The shipping network remains Core ML SPAN; it is not diffusion or Metal 4 inline
ML. Browser transport follows
[Chrome's structured-clone messaging support](https://developer.chrome.com/blog/structured-clone-messaging).
