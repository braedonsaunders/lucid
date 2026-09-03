# Plan: enhancement rendered invisibly inside the browser's video rectangle

Status: architecture A (native shadow layer) is now implemented (2026-09-02);
see README for the current build and install steps. The research below is the
basis for it. Measurements come from two throwaway probes in `Tools/` and one test page,
all run on this M4 Pro (macOS 26, Chrome 151, Safari 26) on 2026-09-02. Raw numbers
are in `output/research/overlay-occlusion-2026-09-02.md`.

## 1. The invariant

The enhanced pixels occupy the page's own `<video>` box and nothing else. No second
window the user can see, no native controls, no side-by-side, no visible handoff. The
output follows the video through scroll, resize, window drag, fullscreen, captions and
the site's control bar, in Chrome and Safari, with zero user knobs.

## 2. What RTX Video Super Resolution actually is, and why there is no literal port

RTX VSR is not an extension and not a shader in the page. Chrome and Edge call
`ID3D11VideoContext::VideoProcessorSetStreamExtension` with an NVIDIA GUID on the
compositor's video processor, and the NVIDIA driver intercepts the next
`VideoProcessorBlt` and runs its model into the overlay swap chain. It sits between
the decoder and the compositor, which is why it works on DRM content and why nothing
on the page can tell it is there.

macOS has the matching engine (`VTLowLatencySuperResolutionScaler`, the temporal
`VTSuperResolutionScaler`, `VTTemporalNoiseFilter`) but no browser calls it, and Apple
exposes none of it to web content. Injecting into Chrome's or Safari's GPU process is
blocked by hardened runtime and library validation. A Chromium fork would cover Chrome
only and is not a product. So the literal RTX architecture is closed to us. Two
architectures can deliver the same user experience.

## 3. The two viable architectures

### A. Native shadow layer (recommended as the primary path)

The native app keeps doing what it does now (ScreenCaptureKit window capture of the
video rectangle, VideoToolbox super resolution, later the Neural Engine student), but
the output window becomes a borderless, click-through, shadowless NSWindow that is
pinned exactly over the video box and z-ordered directly above the browser window.
The user sees enhanced pixels in the video box and nothing else.

Why it was previously believed impossible, and what the measurements show:

- The old overlay covered the whole browser window. Both Chrome and Safari suspend
  rendering when their window is fully occluded (AppKit occlusion state), so capture
  starved. That was the "WindowServer feedback loop". There is no feedback loop:
  window capture never includes other windows.
- An opaque click-through window over only the video box does not reduce capture
  frame rate at all in either browser (30.8 to 32.0 fps against a 30 fps source).
- Fullscreen is solved: a window with `alphaValue = 0.99` covering the entire browser
  window is not treated as occluding by either browser, and Chromium's own occlusion
  code skips windows with alpha below 1.0 or `ignoresMouseEvents`. The 1% bleed of the
  original video under the enhanced frame is not visible.
- Cross-process pinning works: `order(.above, relativeTo: browserWindowNumber)`
  places the overlay right above the browser at normal level, so other apps' windows
  stack correctly above it. The pin is not sticky across app activation, so the app
  has to watch the on-screen window list (sub-millisecond) and re-pin.
- `sharingType = .none` hides the overlay from screenshots and screen sharing with no
  effect on delivery.

Strengths: reuses the entire pipeline already built, uses Apple's 6 ms super
resolution today and the Neural Engine student later, works on any readable video
including cross-origin `<video src>` that a page script cannot touch, and can also be
pointed at Chrome's and Safari's picture-in-picture windows.

Costs, all engineering rather than research:

1. Geometry channel. The extension reports the video's content box (after
   `object-fit`), device pixel ratio, intrinsic size, fullscreen state, paused and
   picture-in-picture state, a "moving" flag during scroll or resize, and the rects of
   page elements that overlap the video with pointer events (control bars, caption
   containers). Chrome: native messaging host relaying to the app over a Unix socket.
   Safari: the extension's native handler lives in the app itself. Window position
   comes from the native side (window list or Accessibility notifications), so the
   page never needs to know where the window is.
2. Motion policy. During scroll, drag and resize the overlay hides and the original
   video shows through, then snaps back after about 80 ms of stillness. Nothing is
   ever misaligned because the underlying video is always the fallback.
3. Occlusion policy. If any other window sits between the browser and the overlay and
   intersects the video box, re-pin; if a window still intersects after re-pinning
   (a Chrome popup, a system dialog), mask or hide. Mission Control and Cmd-Tab are
   excluded with `.transient`, `.stationary`, `.ignoresCycle`.
4. Controls and captions. Everything inside the captured rectangle is enhanced and
   presented one to two frames late, including the site's control bar. Two options,
   both invisible: punch the reported control rects out of the overlay so the page's
   own controls show through, or hide the overlay while the pointer is inside the
   video box. Recommendation: punch-through, with pointer-hide as the fallback for
   sites whose controls cannot be identified.
5. Pixel mapping. On a Retina display the browser already paints a 1280x720 video
   into 2560x1440 physical pixels. The 2x output has exactly those dimensions, so the
   overlay must be aligned to physical pixels and use the display's backing scale.
   Non-integer CSS scaling, page zoom and rotated or transformed video need a resample
   step or a bail-out to the original video.
6. Audio. The overlay has no access to the page's audio, so the Live profile
   (16 to 33 ms) stays the default, as concluded in the previous research pass.

### B. In-page WebGPU engine (recommended as the second engine, not a replacement)

The extension inserts a canvas as a sibling of the `<video>`, sized to the same box,
`pointer-events: none`, below the site's controls, and runs a small network in WGSL on
each frame delivered by `requestVideoFrameCallback`. The `<video>` keeps decoding,
playing audio, seeking and rendering captions.

Verified on this machine: Chrome and Safari 26 both import the playing video into
WebGPU (`importExternalTexture`) with `shader-f16`. At 2560x1440 output, Chrome
sustains 120 fps through eight full-resolution 3x3 passes and 101 fps through 32;
Safari sustains its 60 Hz cap through 32 passes and drops to 41 fps at 64. Chrome's
NV12 path is zero-copy for 8-bit video; 10-bit HDR takes a copy path.

What that buys and what it does not:

- It is literally inside the DOM box: no window tracking, no occlusion logic, no
  screen recording permission, works in fullscreen, follows every layout change, and
  the extension can delay the page's audio to match a longer latency budget.
- It cannot use Apple's models or the Neural Engine. WebNN, the only route to Core
  ML from a page, is not shipped in Chrome (origin trial disabled since March 2026,
  no restart date) and has no Safari implementation. The engine is whatever fits in
  WGSL: published Apple-silicon numbers for hand-written WebGPU networks are 120 fps
  for an 8-channel 7-layer net at 720p to 1440p, 40 fps at 16 channels, 15 fps at 28
  channels. That is the same class as the residual student we plan to train, so one
  trained model can ship in both engines, but the native engine will always afford a
  larger one.
- It cannot read tainted cross-origin `<video src>` without a `crossorigin` reload,
  and cannot read DRM video at all (neither can capture).
- Every real-time in-browser project that works (Anime4K-WebGPU, WebSR, Framegen)
  abandoned ONNX Runtime Web and TensorFlow.js for hand-written WGSL; framework
  inference measured at 0.5 fps for the same model. Plan on WGSL from the start.

### C. Rejected: native processing with in-page presentation over loopback video

Sending source frames from the page to the app and enhanced frames back through a
hardware encoder and WebCodecs would put native models inside the DOM box, but it
re-encodes the output, adds three to four frames of latency, doubles the IPC traffic
at 4K60, and only exists to avoid the window management in A. Revisit only if A's
motion or occlusion policies prove visible in practice.

## 4. Recommendation

Build A first, add B as the engine used when A cannot run, and ship one trained model
for both.

A is the only path that reaches RTX-class quality on this hardware, because the
Neural Engine and Apple's temporal models are reachable only from native code. Its
remaining problems are window bookkeeping, and every one of them now has a measured
solution. B costs one extension and no native code, so it is cheap insurance for
denied permissions, transformed or scaled video, and for the far larger population
that will never install a native app.

Decision gates before coding:

1. Geometry channel round trip under 4 ms in both browsers (native messaging in
   Chrome, in-app handler in Safari). If Safari's handler is slower, fall back to
   title tagging for Safari as today.
2. Shadow layer with hide-on-motion passes a blind viewing test on YouTube, Twitch and
   a plain `<video>` page: no visible seam, lag or double image during normal use.
3. Fullscreen at `alphaValue = 0.99` shows no visible ghosting on high-contrast text.

## 5. Order of work when coding resumes

1. Correctness fixes agreed in the previous review (data races, per-frame session
   creation, format description caching, first-frame interpolation guard, picker
   observer leak, mode enum). These are prerequisites for both engines.
2. Geometry channel: Chrome native messaging host, Safari handler inside Lucid,
   one message schema, replace title tagging.
3. Shadow layer presentation: pinning, window-list watcher, hide-on-motion,
   punch-through control rects, fullscreen alpha rule, Retina alignment, sharing
   exclusion, Mission Control exclusion. Picture-in-picture window as a second source.
4. Engine work from the previous plan: actor stage graph with bounded backpressure,
   PTS-based presentation, Apple temporal SR and temporal denoise benchmarks, then the
   Neural Engine residual student.
5. WebGPU engine: WGSL port of the student, canvas-sibling presentation, CORS reload
   recovery, audio delay for longer latency profiles.
6. Packaging: signed and notarized app that contains the Safari extension and the
   Chrome native messaging host manifest.

## 6. Limits that no architecture removes

- DRM video (Netflix, Disney+, Apple TV+) is black to capture and unreadable in the
  page. RTX VSR can do it only because it lives in the driver.
- Content that the browser stops painting (background tab, fully occluded window) has
  no frames to enhance; the alpha rule prevents us from being the cause.
- Detail that was never in the stream cannot be recovered; the largest gains remain
  480p to 1080p compressed sources, and the automatic strength rule must leave clean
  1080p and native 4K almost untouched.
