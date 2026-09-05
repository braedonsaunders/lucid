# NanoVSR July 2026 pretrained comparison

The official [NanoVSR repository](https://github.com/filippawlicki/nanovsr/tree/b48e2bad89d01e22226eb8613bee25cb45623fae) and [paper](https://arxiv.org/html/2607.10495v1) release a bidirectional recurrent 4× model. This comparison uses the 644k weights at the linked v1.0 release. Source is MIT-licensed; the implementation remains in the isolated upstream checkout, not the shipping app. The loader verifies reviewed source and weight hashes, uses `weights_only=True` and loads every parameter strictly. It does not use the upstream utility's permissive checkpoint loader.

- Source SHA256: `a7eae614f0dbfb5d9532d312d2eebdae0c560fd34de7dae2384c3d1b14e10125`
- Weights SHA256: `515eec20ff589986fc0e0da18b363962b931008ebd848d0c8fb7e256ce200cf7`

The official route receives all 16 frames of each excerpt, including future information. The causal endpoint route carries only the forward feature state; its backward branch sees the current frame only. It exactly matches the official model's final output on each growing prefix (checked lengths 1–4), with zero observed CPU error and maximum CUDA error 0 (tolerance 0.0001). This is an execution adaptation using unchanged pretrained weights, not a retrained streaming model or novelty claim. Boundary-only use differs from the usual bidirectional training context.

The RTX 4080 evaluated 12 codec/bitrate excerpts from three development sources. Shipping and both NanoVSR modes use the same decoded pixels and the same explicit presentation adapter: true 4× output, RGB8 clamp/round, then bicubic to the common 2× reference. Spatial metrics cover 48 fixed frame pairs. Upstream pretraining overlap is unverified; this is not a final untouched holdout. No temporal filter or flicker optimization was performed.

| Mode | LPIPS improvement | DISTS improvement | Fine-correlation change, source-balanced | Decision |
|---|---:|---:|---:|---|
| Full bidirectional | 2.34% | 2.84% | −0.04722 | Reject |
| Past-only endpoint | 5.55% | 3.57% | −0.04342 | Reject |

Both routes fail the fine-detail guard on all three sources. The causal route clears aggregate perceptual minima, but its added detail energy is less faithful to the reference. In this codec-degraded screen, future context does not establish a quality advantage over the endpoint execution. That observation does not generalize to the paper's different training/evaluation tasks.

CUDA excerpt prediction takes roughly 26 ms/frame for the causal route and 31–33 ms/frame for full bidirectional processing at 640×360 input. These opportunistic per-excerpt measurements include no formal steady-state timing protocol and exclude capture, presentation and Mac execution. They must not be presented as live-Mac cadence or competitive hardware parity.

`nanovsr-evaluation.json` records all rows, decoded-pixel fingerprints, model/evaluator hashes, endpoint checks and timings; `nanovsr-gates.json` applies the unchanged spatial thresholds. The first supervisor attempt failed because FFmpeg was absent from its PATH, before scoring any frames. The corrected supervisor explicitly adds the existing `C:\ffmpeg\bin` and completed successfully. No existing GPU job was displaced. No weights were promoted and no NanoVSR dependency was added to the app.

The stored Windows text artifacts normalize CRLF to LF. `nanovsr-artifact-receipt.json` records both original remote and stored hashes; no metrics were changed.
