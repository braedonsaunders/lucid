# Browser color correction — September 4, 2026

An actual Chrome 152 WebCodecs round trip exposed darkened sRGB and Display P3 midtones. Gray 128 returned as 116; maximum error across the 12 flat patches was 13 RGB levels. The same Rec.709 chart had a maximum error of 4. Enhancement was bypassed, isolating native normalization and delivery.

`DecodedFrameSource` now retains sRGB encoding while converting gamut, matrix and range to the native NV12 contract. `EnhancedFrameSender` describes the actual buffer transfer/range instead of labeling every packet video-range Rec.709. It also preserves transfer metadata when resizing. `LearnedUpscaler` tags the Core ML RGB result with its input encoding before conversion back to NV12, clears recycled destination attachments and checks transfer success.

After the change, all three charts have maximum error 4; gray 128 returns as 128. See `browser-color-before.json` and `browser-color-after.json`. This is a pixel comparison against the original VideoFrame drawn into an sRGB canvas in the same browser, rather than an assumption about its color-management behavior.

The native suite passes 34 tests. New regressions cover midtone values while alternating sRGB/709 frames, full/video-range NV12 delivery with and without resizing, and transfer preservation through the bundled 256×144 Core ML model. Tests use an isolated xctestrun environment with `LUCID_EPHEMERAL=1`, bridge port 47931 and token port 47932; the user's running app remains separate.

Reproduce the browser check after building the app:

```sh
PLAYWRIGHT_SKIP_BROWSER_GC=1 playwright-cli -s=lucid-color-check open about:blank
python3 Tools/check_browser_color.py \
  --app .build/tests/Build/Products/Debug/Lucid.app/Contents/MacOS/Lucid \
  --session lucid-color-check \
  --report .build/color-check.json
playwright-cli -s=lucid-color-check close
```

The offline `--color-probe` entry point runs before app state and opens no port. The checker records executable and probe hashes and fails above four levels of error. It uses the host Playwright CLI and an explicitly supplied task-owned session. Close that session even if the check fails.

Limits: flat 8-bit SDR patches do not certify chroma edges, all wide-gamut colors, linear light, display ICC profiles, Safari or HDR. The browser chart bypasses enhancement; model encoding is covered separately by the native regression. This does not complete the release color gate.
