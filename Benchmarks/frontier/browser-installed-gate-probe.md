# Capture-credit diagnostic: slowdown not reproduced

An isolated extension copy appended `Tools/frontier_eval/capture_gate_probe.js` to the unchanged capture policy. It counted admission decisions and acknowledgment round trips without changing their results. The installed companion ran shipping weights on the same 360p60 local fixture with per-draw tracing for 120 status samples.

The result was **7,402 actual draws in 122.913 seconds: 60.22 fps, p95 33.50 ms**. Off passed. This is a short instrumented diagnostic, not a replacement for the failed ten-minute run. Capture acknowledgments averaged about 3 ms in the inspected windows. Final counters showed 7,936 reservations and acknowledgments, no expired pending frames and no unknown acknowledgments; only 16 readiness checks were pending-limited and ten were pacing-limited. Counters include warmup; each admitted frame checks readiness twice, so readiness checks are not distinct video callbacks.

The severe slowdown did not reproduce, and these results do not establish its cause. Keep the failed sustained result and the short successful diagnostic together. Continuous system-load observations and reproduction under contention are needed before attributing the earlier slowdown to a particular bridge operation. No production admission logic or temporal filters changed.

The adjacent directory preserves the full report, compressed acknowledgments, short diagnostic analysis and exact executed harness. The analysis explicitly uses a 120-second diagnostic duration; the release gate remains 600 seconds.
