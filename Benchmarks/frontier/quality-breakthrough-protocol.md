# Ranked quality experiments

Continue the ranked research/innovation list from thread `thr_jbe5uyie3u` under
the native Codex goal. Optimize native delivered quality, keep one shipping
model, and retain the existing comparison gates. No commercial SDK outputs,
weights, or benchmark numbers are included in these experiments or receipts.

## Pipeline correction that changes the experiment

`EnhancementPipeline.process` runs `DetailEnhancer.preprocess` before
`LearnedUpscaler.upscale`. Both debanding and motion-aligned TAA are in
`preprocess`, at decoded source resolution. Sharpening and tone/grain stages
run after reconstruction. Earlier references to all of these as “post-stages”
were inaccurate. The measured native gap combines changed model inputs and
output processing; it is not entirely an output filter tax.

This also invalidates the mechanism proposed for flip-cycle self-ensemble:
there is no accumulator after the model to average orientation-dependent
predictions. Four flips are four orientations, not an eight-way ensemble, and
the CPU buffer flips have a runtime cost even though the inference count stays
one. The feature remains disabled by default.

## Ordered work

1. **LDL and EMA:** r27 is the original approximate formulation. Correct the
   reference formulation and test r28 at the already chosen weight 100, with
   2,000 and 4,000 steps. Score both student and EMA checkpoints.
2. **Flip cycle:** the prior two native runs are complete. The tested mechanism
   and a subsequent actual four-pass static ensemble are closed as breakthrough
   candidates on the measured sources; do not enable by default.
3. **Stage-aware training:** first isolate source-stage adaptation, with three
   consecutive LR frames through the deband/TAA proxy. Compare matched 1,000-step
   fine-tunes from `v4_w0075_2k`, using the same three-frame sampler for both arms.
   Then test native outputs. The output-stage proxy is implemented separately
   behind `--native-output-stages`; its training/evaluation follow the input
   ablation. The input-only arm must not be described as the full pipeline.
4. **Learned pre-cleaner:** pending. Derive clean pre-encode LR supervision from
   the declared resize/encode process; preserve raw input as a residual path.
5. **Degradation conditioning:** pending. Begin with an oracle versus an
   input-only blockiness estimator; do not deploy oracle information.
6. **Recurrent frame feeding:** pending. Warp previous predictions using
   input-derived motion and verify cut/resize/history-reset behavior.
7. **Confidence head:** pending. Test per-pixel stage strength and calibration,
   including the actual pre/post model placement of each controlled stage.
8. **Clean-LR consistency and AESOP:** pending. Exact resize provenance matters;
   a generic RGB downsample is not automatically the bank's pre-encode target.
9. **Larger base pretraining bank:** pending. Keep source-disjoint evaluation,
   licence provenance, and disk requirements explicit before acquisition.

## LDL implementation and verification

The [author's LDL loss](https://github.com/csjliang/LDL/blob/master/basicsr/losses/LDL_loss.py)
uses channel-summed absolute residuals, reflected unbiased 7×7 window variance,
patch variance raised to 1/5, and the EMA comparison mask. The
[training implementation](https://github.com/csjliang/LDL/blob/master/basicsr/models/srganArtifactsDis_model.py)
differentiates through the current residual map and saves EMA weights.

r27 detached that map, used population variance with truncated edge windows,
and did not save its EMA. It remains an interpretable approximate-loss trial,
not a faithful paper reproduction. `--ldl-formulation legacy` preserves its map
behavior. The corrected default uses the reference math, a small floor on patch
variance to avoid undefined gradients at exact reconstruction, and named EMA
state updates. Both student and EMA weights are now saved in loadable files.
Tests compare values and gradients against the reference definition, check
constant/perfect reconstruction, masking, and EMA parameters/buffers.

The completed r27 raw development scores (LPIPS / DISTS gain against the original
SPAN comparator) are 16.82% / 18.70% at weight 30, 2k; 15.84% / 18.06% at weight
100, 2k; and 16.21% / 19.21% at weight 100, 4k. The existing r17 no-LDL 2k recipe
scored 17.11% / 19.08%. None establishes a joint improvement over that recipe.
At 4k, the raw legacy LDL candidate still fails the bank-validation fine
correlation floor. The fixed 80% mixtures pass the development and bank gates
but yield smaller perceptual improvements. All six gate receipts are retained
under `quality-breakthrough/ldl*-gate.json`. These are not native release results.

## Flip-cycle result

Each native run has 960 frames per model across the eight existing holdout
sources. Against the identical model with guarded native stages and no flips:

| Model | LPIPS change | DISTS change |
|---|---:|---:|
| Shipping `big2k` | −0.116% | −0.159% |
| 33-source `v4_2k` | −0.558% | −0.072% |

Negative is better. The small aggregate gains are mixed per source. On shipping,
Sunflower improves 1.40% LPIPS / 1.52% DISTS, while OldTownCross worsens 0.44% /
0.62% and InToTree worsens 0.28% / 0.85%. This does not establish the intended
ensemble mechanism, a material gain, temporal stability, or zero latency cost.
`quality-breakthrough/flip-cycle-comparison.json` preserves the complete
per-source comparison and hashes of the full reports. No promotion.

An actual four-pass static ensemble was then tested on 48 development frames,
using shipping `big2k` and `v4_w0075_2k` on the same MPS scorer. Each prediction is
unflipped and quantized to RGB8 before averaging. Unlike frame cycling in the
current app, this actually averages all four orientations of identical content.
It costs four model evaluations and is not a native temporal experiment.

| Model | LPIPS change | DISTS change |
|---|---:|---:|
| Shipping `big2k` | +3.596% | +1.901% |
| 33-source `v4_2k` | +5.111% | +3.243% |

Positive is worse. Every development source worsens on both metrics. This
rejects the proposed averaging mechanism on this screen; it does not establish
a universal result for every scene or model. The full 240-row report and
per-source comparison are `static-flip-ensemble-evaluation.json` and
`static-flip-ensemble-comparison.json` in `quality-breakthrough/`.

## Stage proxy verification and limits

`native_stages.py` follows the production deband, block motion, and TAA kernels.
`native_output_stages.py` adds the fixed 2x shipping sharpen, tone and adaptive
grain recipe, with neutral chroma saturation.
`NativeStageBench.swift` extracts and compiles those actual kernel definitions,
without maintaining a second Metal implementation. Four deterministic fixtures
cover translated texture, a plateau, a cut, and a stationary frame.
All twenty source/output stage comparisons pass the 1e-5 absolute tolerance;
maximum error is 5.216e-6. Sharpening differs by at most 1.20e-7 and tone/grain by
1.79e-7. Receipts are `quality-breakthrough/source-stage-metal-parity.json` and
`quality-breakthrough/full-stage-metal-parity.json`.

This validates FP32 kernel math, not VideoToolbox RGB↔NV12 conversion, production
fast math, chroma siting, long-running history, or end-to-end quality. The training
proxy approximates RGB/420 packing, quantizes UNORM8 intermediates and stores TAA
history and motion fields in FP16. The frozen r28 input-only snapshot used FP32
motion-field confidence; subsequent snapshots emulate FP16 field storage too.
Each three-frame crop starts with reset history. Crop boundaries
and the absence of warmed full-frame context are additional limitations. No HR
information enters the motion search or LR stages.

The source stages precede the model, so gradients through those stages are not
needed for the current input-adaptation arm. They remain differentiable where
the native decisions allow it, with straight-through quantization, for later
cleaner experiments. Output-stage losses do backpropagate into the SR model,
including the critic and LDL terms when enabled. The output proxy uses the
decoded LR to set adaptive grain amplitude, never the HR target. Full-frame
statistics and spatial grain phase differ from a training crop and must still
be validated natively.

## Execution and admission

`Tools/experiments/run_quality_breakthrough_r28.ps1` fixes the four training arms
and their development/bank evaluations. It uses a fresh directory and refuses
to start alongside existing Python jobs. Stage and control both use three-frame
sampling; their first decoded sequence hashes must match. Metadata records exact
source and checkpoint hashes. Training completion alone is not quality evidence.

Evaluate only complete reports with matched frame identities, declared model
weights and comparator pixels. Development screens choose which candidates merit
native evaluation; they do not constitute release admission. A repeated holdout
is regression evidence, not fresh independent validation. Keep the current
shipping assets until a candidate delivers a material native improvement with
the existing detail and per-source safeguards and acceptable runtime cost.

For stage-aware candidates, raw checkpoint scores are diagnostic only: these
models are explicitly optimized for transformed inputs and/or presented outputs.
Their raw scores cannot substitute for native evaluation. The r29 script fixes
source-only (FP16 field), output-only, and combined 1,000-step arms; it is prepared
separately from the immutable running r28 snapshot.
