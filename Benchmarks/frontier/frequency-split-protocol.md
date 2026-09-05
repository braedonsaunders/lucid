# Fixed spatial decomposition test

The regional diagnostic found that perceptual training increases fine energy while losing reference edge alignment. Before another training run, test whether the existing candidate's perceptual gain survives when its highest-frequency changes are attenuated. This is a spatial reconstruction diagnostic, not flicker filtering.

Use the existing reference-detail experiment's fixed 80% checkpoint (`C:/lucid/reference-detail-20260905/calibrated/blend-0.8.pth`) and unchanged shipping weights. Both are converted to the same RGB8 2× presentation domain used by the frozen 48-frame development scorer. Construct exactly one new output: `shipping + Gaussian(candidate - shipping)`. Fix sigma=1 output pixel, radius=3, separable RGB filtering and replicated boundaries before evaluation, based on the existing sigma-1 detail diagnostic. Do not select a different sigma, candidate, or blend after seeing this run. References supply scores only, never predictions or per-image parameter selection.

This does not mathematically preserve the legacy fine-correlation metric: Gaussian filtering is not an ideal frequency projector, and quantization/clamping remain. Require the unchanged aggregate 3% LPIPS/DISTS improvements, per-source perceptual guard and 0.01 fine-correlation limit. A pass would justify testing a constrained cheap residual branch, not shipping this two-network combination. A failure rejects this fixed decomposition. No model training, shipping promotion, temporal changes, or novel-method claim is implied.

Record both unmodified controls on the same frozen pixels and compare their metrics/hashes with the completed reference-detail report. Preserve the full result even if it fails. Any subsequent learned replacement still needs fresh data, native conversion correctness and end-to-end performance.


## Completed: below the promotion gate

The fixed decomposition improved source-balanced **LPIPS 2.579%** and **DISTS 7.322%**. All per-source perceptual and fine-detail guards pass. Fine-correlation changes are CrowdRun −0.000231, FourPeople +0.000254 and Johnny +0.000450. The sole spatial gate failure is LPIPS improvement below the frozen 3% minimum. The minimum was not relaxed and no sigma, blend or candidate sweep followed.

Compared with the unfiltered candidate's 6.939% LPIPS / 10.106% DISTS improvements and face-detail failures, this operation gives up much of the LPIPS benefit while retaining DISTS improvement and restoring fine fidelity. It establishes a measured tradeoff, not a passing model or proof that a learned low-frequency branch will succeed. The two-network operation is not integrated into native playback.

`frequency-split-evaluation.json` contains all 144 frame/model rows, hashes and scorer metadata. `frequency-split-gate.json` applies the unchanged thresholds; `frequency-split-controls.json` verifies both checkpoint hashes and per-frame controls against the completed reference-detail report. `frequency-split-artifact-receipt.json` records original Windows and LF-normalized artifact hashes. Two filter tests pass. The RTX 4080 job completed successfully, and shipping weights and temporal filters are unchanged.
