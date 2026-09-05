# Sustained installed Chrome companion: failed gate

Shipping weights delivered **25,918 actual canvas draw acknowledgments over 620.040 seconds: 41.80 fps**, with pooled capture-to-draw p95 **50.70 ms** and p99 **82.60 ms**. The sustained gate requires at least 57 fps and p95 at most 50 ms for at least 600 seconds. Duration and Off passed; cadence and latency failed.

This used the actual unpacked companion installed through Chrome CDP in an owned Chrome 152 profile, with service-worker messaging and an extension-origin presentation iframe. The extension copy changed only the isolated test ports. A 20-second, 1,200-frame 640×360 source looped at its original 60 fps. The native shipping model reconstructed 2560×1440 and the sender delivered 1280×720. The local fixture loaded no substitute companion scripts. Extension, app, fixture and harness hashes are in the report.

The harness recorded each real `presented` acknowledgment after the surface submitted its draw. This measures capture through canvas submission, not physical display scanout. It sampled 600 status snapshots in 30-second chunks; the full elapsed interval includes the gaps between chunks, and acknowledgments during those gaps count. The analyzer validates the trace digest, extension/session identity, strictly increasing unique sequence IDs, finite latency, and coverage of both measurement boundaries. Status-window FPS and mean status-window p95 are not used as the primary metrics.

The first four one-minute bins delivered 13.32, 38.62, 11.02 and 31.98 fps. Later full minutes delivered 51.23–54.65 fps. Video callback telemetry remained around 60/sec while submitted-frame rate fell, warranting capture/transport diagnosis before attributing this to inference. This was a shared workstation; no continuous system-load trace was collected, so it does not isolate the cause of the slowdown. Later precision experiments observed substantial unrelated CPU load and cannot retroactively establish this run's load.

Twenty native RSS samples ranged up to 185.48 MiB and ended at 175.47 MiB, below the initial 182.78 MiB. Only two browser RSS spot checks were collected; they do not prove sustained browser memory behavior. Short installed-extension ABBA runs in `browser-installed-60fps.json` also showed wide startup variation and do not establish a sustained pass.

Reproduce the analysis with:

```sh
python3 Tools/frontier_eval/analyze_browser_trace.py Benchmarks/frontier/browser-installed-shipping-10min/report.json --out /tmp/lucid-browser-gate.json
```

The adjacent evidence directory contains the report, compressed raw trace, gate output, sparse browser memory observations and exact executed harness. This covers one local Chrome 360p60 condition. Other resolutions, frame rates, third-party CSP, Safari and sustained browser-memory gates remain open. No weights, tuning or flicker filters changed.
