# Transfer decoded frames through the extension surface

The content script now transfers each decoded-frame buffer through a private MessageChannel to its extension-origin iframe. That iframe already has an authenticated native WebSocket for presenting results, and sends the input packet on the same connection. This removes the extra typed-array allocation and service-worker serialization hop from normal capture. The runtime-port path remains available while the iframe or its connection is unavailable.

The relay accepts decoded-frame packets only, validates the packet's session and sequence, and obeys authenticated native enablement and socket backpressure. It cannot forward tokens, attach messages or arbitrary controls from the page. Lost connections and replaced iframes release the direct route's pending capture credits. A failed transfer drops that frame and allows subsequent frames to use fallback; it cannot send an already-released frame outside the credit budget. Chrome and Safari resource copies are byte-identical.

Six regression tests exercise valid delivery, Off, wrong parent/session, malformed data, backpressure, frozen pages, iframe replacement, stale callbacks, failed handshakes and failed-transfer cleanup. Tests execute the production surface and transport functions. Installed Chrome provides the separate integration evidence below; Safari delivery has not yet been measured.

## Installed Chrome ABBA

Four fresh browser/native sessions used the same shipping weights, video, tuning, viewport and executed harness hash. Only the extension's input route changed. Each run measured 60 one-second samples with actual per-draw acknowledgments, not an average of displayed FPS counters.

| Route order | Actual FPS | Capture-to-draw p95 |
|---|---:|---:|
| Service worker | 25.33 | 55.5 ms |
| Direct surface | 57.75 | 42.1 ms |
| Direct surface | 59.93 | 33.7 ms |
| Service worker | 13.58 | 58.4 ms |

Pooled direct delivery was **58.84 fps**, versus **19.46 fps** for the service worker: 3.02× in this short comparison. Pooled p95 improved from 57.4 to 41.1 ms. Workstation load and startup variability limit generalization. The raw reports, compressed traces, exact executed builder/harness and input hashes are in `surface-capture-abba/`.

## One continuous session

To reduce startup confounding, a benchmark-only prototype switched routes inside one live browser/native session. Four phases each recorded 60 samples after a five-second transition wait. The trace was validated globally and then counted inside each phase's recorded bounds. No model or browser restart occurred between phases.

| Phase | Route | Actual FPS | p95 |
|---|---|---:|---:|
| 1 | Service worker | 38.15 | 48.3 ms |
| 2 | Direct surface | 58.73 | 40.9 ms |
| 3 | Direct surface | 58.59 | 41.2 ms |
| 4 | Service worker | 52.39 | 43.5 ms |

Pooled delivery improved **29.6%**, from 45.27 to 58.66 fps, with p95 falling from 46.4 to 41.1 ms. The warm final control phase also remained slower. This supports an input-transport improvement, not a universal 3× claim. `surface-capture-switch/` preserves the phase bounds, raw trace, full status data and exact code. The DOM route selector exists only in the benchmark prototype, not in the integrated extension.

These measurements cover a 640×360 60 fps local video, shipping 4× reconstruction and 1280×720 delivered output. Canvas draw submission is measured, not physical scanout. No weights, native image processing or flicker filters changed. The sustained result follows below; third-party CSP and Safari coverage remain open. These traces do not identify repeated source frames, so draw cadence alone is not a unique-video-frame guarantee.


## Integrated extension: sustained result

The integrated route recorded **36,944 draws over 621.2254 seconds: 59.4696 fps**, with **40.6001 ms p95** and 42.7000 ms p99 capture-to-draw latency. All duration, draw-cadence, latency and Off gates passed. Every complete minute exceeded 58 fps. Native RSS was 196.14 MiB initially, 186.06 MiB finally and 198.41 MiB peak over 20 samples; this is not a full browser-process memory gate.

`surface-capture-10min/` preserves the report, trace, exact executed harness/analyzer and complete installed extension with verified hashes. It includes lifecycle cleanup but predates the telemetry-only correction that resets the socket label on runtime fallback. The correction changes neither packet delivery nor measured draw acknowledgments. A tested null-header rejection was also added after that run; valid packets follow the same path. The live extension now reports fallback accurately. A subsequent test-only probe links acknowledgments to native output timestamps to measure repeated source frames separately.


## Reproducing historical route comparisons

The prototype builder applies checked replacements to the pre-integration source. Extract `BrowserExtension/` from commit `e4b8634` into an isolated directory and pass that directory with `--source`; it is not intended to patch the already-integrated current source. Exact executed builder snapshots and receipts are preserved with both comparisons. The regular installed-browser harness accepts the integrated extension directly, after assigning the isolated test ports.
