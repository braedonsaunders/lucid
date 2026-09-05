# Train a correction inside the fixed spatial constraint

The completed two-model decomposition produced 2.579% LPIPS and 7.322% DISTS improvement with all fine-detail guards passing, but missed the unchanged 3% LPIPS minimum. That fixed decomposition is rejected; there is no sigma or blend sweep. Its result motivates a different training hypothesis: let a small branch learn corrections through the same fixed spatial constraint, rather than filtering an already-trained unconstrained model after inference.

Keep the previous anchored-detail experiment's immutable folded shipping base, 32-channel/four-block correction branch, 29-source bank, cached 50/50 teacher target, seed 20260914, batch 4, crop 96, learning rate 0.00002, losses, pinned released PixRestore DINO discriminator and 8,000-step schedule. Change only the branch's output operator: fixed sigma-1, radius-3 separable Gaussian with replicated boundaries, before adding the correction to the base. No temporal state or filtering is added. The convolution kernel is a nontrainable checkpoint buffer; a distinct architecture tag prevents loading the weights without their filter.

The 2026 motivation remains [PixelIR's separate fidelity and detail optimization](https://arxiv.org/html/2608.30782v1) and [PixRestore's released adversarial guidance](https://arxiv.org/html/2608.16793v1). This compact implementation is not a reproduction of either architecture, and a fixed Gaussian is not a novel technique. Quality and measured Mac cost decide whether the combination is useful.

Before full training, verify nonzero-branch Core ML conversion and measure the native graph at 360p/720p. It must pass the existing RGB conversion tolerance and keep 720p prediction below 30 ms mean in this short feasibility probe; that is not an end-to-end release gate. Paired two-step CUDA smokes must match first-batch/discriminator hashes and preserve the base weights. Reuse the completed unfiltered anchored run as control only after verifying these inputs and protocol match.

Score the final raw checkpoint once on the frozen 48 full-frame development pairs and 96 bank-validation patches. No checkpoint or blend selection is allowed. Apply unchanged 3% LPIPS/DISTS minima, per-source perceptual guards and 0.01 fine-correlation limit. Any result remains development evidence and needs native trained correctness and fresh release footage before promotion. Preserve shipping weights and existing remote jobs.


## Feasibility and smoke results

The nonzero random branch passed Core ML correctness checks at both sizes. M4 Pro CPU+GPU prediction averaged 7.262 ms at 640×360 (p95 7.475 ms) and 27.728 ms at 1280×720 (p95 28.285 ms). Maximum RGB errors were 0.887 and 0.941 levels. These short, interleaved 20-sample measurements satisfy the predeclared native feasibility screen; trained correctness and sustained playback remain separate.

Both two-step CUDA arms completed and matched all six checked data/cache/checkpoint/initialization provenance fields. The full-training supervisor also checks the recorded settings and hashes against the historical unfiltered run before admitting the 8,000-step experiment. The new kernel stays fixed, the base remains frozen, and checkpoint loading without the required filter is covered by a failing-loader regression test. The completed quality result follows below; feasibility alone did not establish useful reconstruction.


## Completed training and evaluation: rejected

The RTX 4080 completed all 8,000 steps in **5.3078 minutes**, with the immutable base checked at each saved checkpoint. On 48 full-frame pairs, LPIPS improves **1.2577%** and DISTS **0.7616%**; fine-detail guards pass, but both perceptual minima fail. On 96 bank-validation patches, LPIPS improves **1.4072%** and DISTS only **0.00067%**; Sintel also loses 0.02024 fine correlation and fails the detail guard.

This does not improve the earlier unconstrained branch's full-frame result (1.27% LPIPS / 1.03% DISTS). Training through the fixed filter did not recover the two-model diagnostic's DISTS improvement or meet the joint gates. No checkpoint selection, additional filter tuning or promotion followed. The trained weights remain in the isolated experiment and `.build/anchored-lowpass-candidate/`; shipping weights are unchanged.

The adjacent full training receipt/log/experiment and development/bank reports preserve all results. `anchored-lowpass-controls.json` verifies matching manifests, checkpoint hashes and per-frame control metrics against both earlier evaluations. `anchored-lowpass-artifact-receipt.json` records original Windows and stored LF-normalized hashes. Eight architecture/objective tests pass, including gradients through the filter, immutable base/kernel, traced round-trip and rejection of an incorrectly tagged checkpoint. The initial native feasibility result remains an untrained-graph measurement; trained native quality is not claimed for this rejected checkpoint.
