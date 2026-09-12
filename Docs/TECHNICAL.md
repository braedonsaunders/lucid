# How Lucid works

Lucid 1.0 uses a domain-trained Nano-derived network with 48 channels and six blocks for direct 2× reconstruction. Seven fixed input shapes cover supported 144p–720p sources. The model runs through Core ML, followed by Metal enhancement stages. Device calibration chooses shapes the Mac can sustain.

## Model and image pipeline

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
  Controls open only when the menu bar icon is clicked. Launching or reopening Lucid leaves them closed.
  Turning Lucid off stops frame capture and clears the enhanced surface.
- **Presentation feedback.** Sequence numbers correlate input with browser draw
  acknowledgments. The panel reports presented frames/s and capture-to-canvas
  p95, separately from native processing time. This measures draw submission,
  not physical screen scanout.

Decoded playback drawn in the browser needs **no Screen Recording permission**
and no native window match. Screen-capture fallback is a separate path.

## Admission and compatibility

Lucid starts enhancement for supported unprotected SDR video enlarged by at least 1.15×. It keeps an existing session down to 1.0× to avoid toggling at the boundary. Sources must fit a bundled model shape and the measured frame budget. HDR, protected playback, and 1080p-or-higher sources are declined.

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

## Reproducing the model packages

`Tools/export_nano.py` validates the checkpoint hash and architecture, exports the seven Core ML shapes, compares them against PyTorch, and writes model metadata. Run it with `--help` for the required checkpoint and output arguments. The package checksums and model identity live in [`Models.json`](../Lucid/Resources/Models.json).

The packaged model is a single-frame Nano-derived network. SPAN architectures and older experiment tools remain in the repository for research; they do not define the shipping model.

Training checkpoints and corpora are separate artifacts. Patch-bank validation is a training diagnostic, not an independent benchmark. Results can vary across content, devices, and framework versions.
