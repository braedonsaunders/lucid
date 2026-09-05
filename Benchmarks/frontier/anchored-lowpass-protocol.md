# Train a correction inside the fixed spatial constraint

The completed two-model decomposition produced 2.579% LPIPS and 7.322% DISTS improvement with all fine-detail guards passing, but missed the unchanged 3% LPIPS minimum. That fixed decomposition is rejected; there is no sigma or blend sweep. Its result motivates a different training hypothesis: let a small branch learn corrections through the same fixed spatial constraint, rather than filtering an already-trained unconstrained model after inference.

Keep the previous anchored-detail experiment's immutable folded shipping base, 32-channel/four-block correction branch, 29-source bank, cached 50/50 teacher target, seed 20260914, batch 4, crop 96, learning rate 0.00002, losses, pinned released PixRestore DINO discriminator and 8,000-step schedule. Change only the branch's output operator: fixed sigma-1, radius-3 separable Gaussian with replicated boundaries, before adding the correction to the base. No temporal state or filtering is added. The convolution kernel is a nontrainable checkpoint buffer; a distinct architecture tag prevents loading the weights without their filter.

The 2026 motivation remains [PixelIR's separate fidelity and detail optimization](https://arxiv.org/html/2608.30782v1) and [PixRestore's released adversarial guidance](https://arxiv.org/html/2608.16793v1). This compact implementation is not a reproduction of either architecture, and a fixed Gaussian is not a novel technique. Quality and measured Mac cost decide whether the combination is useful.

Before full training, verify nonzero-branch Core ML conversion and measure the native graph at 360p/720p. It must pass the existing RGB conversion tolerance and keep 720p prediction below 30 ms mean in this short feasibility probe; that is not an end-to-end release gate. Paired two-step CUDA smokes must match first-batch/discriminator hashes and preserve the base weights. Reuse the completed unfiltered anchored run as control only after verifying these inputs and protocol match.

Score the final raw checkpoint once on the frozen 48 full-frame development pairs and 96 bank-validation patches. No checkpoint or blend selection is allowed. Apply unchanged 3% LPIPS/DISTS minima, per-source perceptual guards and 0.01 fine-correlation limit. Any result remains development evidence and needs native trained correctness and fresh release footage before promotion. Preserve shipping weights and existing remote jobs.


## Feasibility and smoke results

The nonzero random branch passed Core ML correctness checks at both sizes. M4 Pro CPU+GPU prediction averaged 7.262 ms at 640×360 (p95 7.475 ms) and 27.728 ms at 1280×720 (p95 28.285 ms). Maximum RGB errors were 0.887 and 0.941 levels. These short, interleaved 20-sample measurements satisfy the predeclared native feasibility screen; trained correctness and sustained playback remain separate.

Both two-step CUDA arms completed and matched all six checked data/cache/checkpoint/initialization provenance fields. The full-training supervisor also checks the recorded settings and hashes against the historical unfiltered run before admitting the 8,000-step experiment. The new kernel stays fixed, the base remains frozen, and checkpoint loading without the required filter is covered by a failing-loader regression test. The full run is in progress; no quality result is implied by these checks.
