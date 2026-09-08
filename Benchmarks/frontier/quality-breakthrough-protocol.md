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
4. **Learned pre-cleaner:** implemented; matched 2k raw/native-input arms complete.
   Clean-LR error improves about 5%, but raw SR development distances worsen.
   Native-input arm still needs native evaluation. Preserve the raw residual path.
5. **Degradation conditioning:** implemented; constant, estimated, and oracle
   arms are fixed in r31. Compare reference-assisted damage maps with local
   input-only estimates; never deploy oracle information.
6. **Recurrent frame feeding:** prototype and state-lifetime tests implemented;
   r32 fixes the first 2k training probe. Training results and native integration
   remain pending. Warp previous predictions using decoded-input motion.
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

Corrected reference-LDL weight-100 student / EMA checkpoints complete at 2k and
4k. All four pass the development and source-disjoint bank guards. Their raw
development LPIPS/DISTS gains against the original SPAN comparator are
14.83%/18.00%, 11.34%/13.76%, 15.03%/18.44%, and 14.16%/17.45%, respectively.
None jointly beats the existing no-LDL development result. The 2k student on
960 repeated regression frames improves aggregate LPIPS 0.34% and DISTS 2.20%
against current `big2k`, but Sunflower LPIPS worsens 9.92%. The 2k EMA worsens
aggregate LPIPS 1.40% while improving DISTS 0.41%. No promotion. The 4k weights
have development/bank evidence only; do not imply they received native or 960
regression testing. Receipts are `ldlref*-gate.json` and
`ldlref100_2k-holdout-comparison.json`.

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
source-only (FP16 field), output-only, and combined 1,000-step arms; all training
and raw development/bank evaluations completed separately from the immutable
r28 snapshot. Core ML random-image conversion errors are below one RGB level;
timings collected alongside other GPU work are not latency admission evidence.

The r28 source-control/input-stage native comparison is complete: 960 frames per
candidate, with identical comparator and reference pixels verified by hash.
Input-stage training worsens LPIPS 1.76% and improves DISTS 1.26% against the
matched control. Against shipping `big2k`, it worsens LPIPS 1.61% and improves
DISTS 1.52%, with LPIPS regressions of 20.11% on Rush Hour and 24.56% on Sunflower.
Even the control has large regressions on those sources. Reject this candidate
for promotion. `source-stage-native-comparison.json` retains all source deltas;
`source-stage-matched-training.json` proves matched sampling and initialization.
The r29 FP16-field source arm reaches the same conclusion: LPIPS worsens 1.49%
and DISTS improves 1.28% against the matched control (1.35% worse / 1.54% better
against shipping). `source-fp16-native-comparison.json` records that native run;
960 identical comparator scores were reused after full pixel/provenance checks.

`score_native_holdout.py --reuse-scores RGB_DIRECTORY REPORT` reuses measurements
only when current output/reference pixel hashes match a complete prior report
with the same metric implementation, Torch version and device. It validates
manifest binding, full sample coverage, metrics and conflicting duplicate pixel
pairs. Current image/reference bytes are still hash-checked. This saves repeated
scoring of unchanged comparators without relabeling candidate measurements.

## Pre-cleaner experiment

Inspired by [RealBasicVSR's cleaning stage](https://arxiv.org/abs/2111.12704), the
independent three-convolution cleaner has 3,203 parameters, an identity
initialization, and an RGB residual bounded to 1/8 of the range. The shipping
`big2k` SR weights remain frozen. r30 trains two matched 2,000-step arms: decoded
RGB and three-frame deband/TAA-proxy inputs. The final frame receives clean-LR
L1 plus signed-Sobel supervision. Both use the same aligned sampler and targets.

Targets are FFmpeg Lanczos 2x RGB reductions of the bank's HR patches, before
codec and chroma subsampling. They are **not bit-exact pre-encode YUV**. Four LR
border pixels are excluded; an interior crop/full-frame resize check differs by
at most one RGB level. Cache manifests bind source splits, bank identity, target
recipe, FFmpeg version and target hashes. Texture present in the clean target is
retained as signal, not labeled noise.

Source-disjoint clean-LR validation MSE improves 5.07% for decoded-input training
and 4.75% for native-input training. Raw SR development LPIPS/DISTS instead worsen
0.332%/0.378% and 0.303%/0.230%. The native-input model still requires native
evaluation. These results show that cleaner LR reconstruction does not itself
establish better SR. `precleaner-screen.json` binds both runs and source deltas.
Tests cover identity, bounded corrections, frozen SR gradients, loadable
checkpoints, improving a known corruption, alignment, and cache provenance.

## Degradation-conditioning experiment

[DASR](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Unsupervised_Degradation_Representation_Learning_for_Blind_Super-Resolution_CVPR_2021_paper.html)
motivates conditioning on degradation; this experiment is not its architecture
or contrastive training recipe. r31 freezes `big2k` and learns bounded gain/bias
modulation of its first feature map. Compare a constant two-channel condition,
a small local RGB/curvature estimator, and explicit reference-assisted maps.
Oracle maps contain locally averaged clean-LR absolute error and curvature error.
They include chroma/resize/stage error, not solely codec blockiness. Curvature
alone also responds to real texture; no codec-grid alignment is assumed.

All three arms share 2,000 steps, three-frame native-input sampling, initialization
and reconstruction losses. The estimator learns reference-only damage labels;
its SR condition is detached so calibration and reconstruction remain distinct.
Validation compares the same frozen backbone and candidate on source-disjoint
third-frame patches. The oracle is a mechanism diagnostic, not a mathematical
upper bound or deployable checkpoint. Ordinary inference and the shared loader
reject oracle use. Estimated inference accepts only current RGB. Identity in
FP32/BF16, frozen-backbone preservation, crop locality and checkpoint round trips
are tested. Fold the backbone in FP32 once before training to prevent autocast
from refreshing its supposedly frozen fused convolution caches.

All three r31 arms completed with unchanged backbones and identical first
decoded batches, initialization and target-cache hashes. Constant, estimated
and oracle conditions worsen validation LPIPS by 0.681%, 0.900% and 1.128%, and
DISTS by 0.350%, 0.224% and 0.190%. The estimate lowers condition MAE from 0.233
to 0.163, but that calibration gain does not improve SR. Reject this modulation
recipe for promotion; this does not rule out every degradation-conditioned
architecture. `degradation-conditioning-screen.json` preserves all three runs.

## Recurrent frame-feeding prototype

[FRVSR](https://openaccess.thecvf.com/content_cvpr_2018/html/Sajjadi_Frame-Recurrent_Video_Super-Resolution_CVPR_2018_paper.html)
motivates feeding the previous reconstructed frame into the current network.
The first Lucid probe freezes shipping `big2k` and adds a zero-initialized,
bias-free history convolution at the first feature map. A direct 4x space-to-depth
pack converts the previous 2x RGB image to 48 channels at SPAN trunk resolution.
This has the same information as a two-step pack, but its channel order is the
direct 4x order and must be preserved in a future native port.

Correspondence uses decoded LR, never the target or generated detail. Native
integer block motion seeds a half-pixel refinement around both the integer and
zero offsets. The zero seed matters: integer matching can choose an unrelated
location when the true match falls between pixels. Unlike TAA, this prototype
does not discard all small motion as stationary. This is experimental motion
code, not a claim of production-kernel parity or a universal flow solution.

The runner rejects occlusions and low-confidence cuts and explicitly resets on
first frame, seek/gap, stream change, resize, device/dtype change or caller reset.
History is the RGB8 reconstruction before sharpening/grading. Quantization uses
a straight-through gradient during unrolled training. The generic still-image
loader rejects recurrent checkpoints; a dedicated loader and sequence runner
preserve the state contract. Seven tests cover identity, frozen gradients,
integer/half-pixel correspondence, cuts, lifetime resets, storage, unrolled
backpropagation and checkpoint round trips.

r32 fixes a 2,000-step, three-frame FP32 training probe with the native input
proxy, current-frame backbone frozen and final-frame reconstruction supervision.
Validation uses full bank sequences to expose longer-history drift and compares
spatial metrics and temporal residual error with the same frozen backbone.
Native SR warping, model-interface integration, latency and delivered quality
are not implemented or established by these prototype tests.
