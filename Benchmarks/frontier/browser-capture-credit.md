# Capture scheduling under bursty video callbacks

The faster experimental 2× model originally delivered fewer browser frames than shipping. A 10-second candidate trace recorded 511 allowed capture attempts, 426 admissions and 85 rejections; **80 rejections occurred with zero pending captures**. Rejected callbacks commonly arrived approximately 8.3 ms after the preceding callback, below the 9.76–11.07 ms processing-derived interval. Chrome supplied approximately 50 callbacks per second overall. The previous minimum-spacing gate discarded unused capacity between callback bursts.

`CaptureGate` now retains at most one interval of idle credit. A long idle permits at most two immediate captures; sustained admissions remain paced by native processing cost, with at most two pending copies. Processing-cost changes preserve fractional credit/debt. Off, foreign sessions, expired connection leases and resets clear admission state. This does not change a temporal image filter. Eight companion tests pass, including jitter replay, sustained overload, lost acknowledgments and connection transitions. Chrome and Safari use identical policy bytes.

## Visible Chrome comparison

Both measurements use Chrome 152 on M4 Pro, a visible 900×700 viewport at device scale two, the same 640×360 50 fps looping source, and actual 1280×720 sender packets/canvas. Each test runs shipping/candidate/candidate/shipping, with five seconds warmup and 30 one-second status samples per run. The isolated native app uses CPU+GPU and Standard settings; candidate sharpening is the already frozen 0.4. Each run explicitly tests Off and closes its app, server and browser session. Fixture MessageChannels exercise the real companion scripts but do not establish installed-extension or third-party CSP behavior.

Only the served `stream-policy.js` hash changed between the two headed comparisons. App, model selection, video, HTML, other companion scripts, harness and browser configuration stayed fixed. The diagnostic trace hook was removed before the second comparison. Input hashes are recorded before each run completes. The exact executed harness is preserved as `presented-calibrated-browser-headed-command.py` and is also `Tools/frontier_eval/run_browser_candidate.py` at this commit.

| Model | Previous presented fps | Credit presented fps | Previous p95 latency ms | Credit p95 latency ms |
|---|---:|---:|---:|---:|
| Shipping | 46.02 | 50.52 | 27.10 | 39.05 |
| Experimental candidate | 42.38 | 50.83 | 19.26 | 24.60 |

These are means of two run means. The latency columns average the native status window's reported p95 values; they are **not pooled per-frame percentiles**. Individual credit runs report mean-window p95 values of 35.17/42.93 ms for shipping and 21.39/27.81 ms for candidate. Processing time also varies across runs. Output slightly above the nominal 50 fps reflects measured callback/timing windows on a looping fixture, not new source information.

Cadence rises **9.78% for shipping and 19.94% for the candidate**, reaching approximately the available callback rate. The added admitted frames also increase latency: **11.94 ms shipping and 5.34 ms candidate** in these averages. This is a throughput/latency tradeoff, not a latency optimization. Consecutive short before/after tests do not isolate thermal or system-load drift. Both shipping run averages remain under 50 ms, but one reported status-window p95 reaches 52.0 ms. This does not pass the release latency gate; the required ten-minute installed-extension measurement and 60 fps footage also remain unverified. The candidate still fails independent detail fidelity and is not promoted.

A separate earlier headless comparison delivered approximately 30 fps for both models despite the 50 fps source. It is retained in `presented-calibrated-browser-headless.json`; it must not be used to assert visible-window cadence. All raw headed results, trace events and the derived summaries remain in this directory.
