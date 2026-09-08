# Ranked quality experiments

Continue the ranked research/innovation list from thread `thr_jbe5uyie3u` under
the native Codex goal. Optimize native delivered quality and keep one shipping
model. Retain the existing comparison measurements as diagnostics, alongside
visual review: a threshold alone must not discard a visibly better result.
No commercial SDK outputs,
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
   Native-input arm is also natively evaluated: mixed perceptual result, no promotion.
   Preserve the raw residual path.
5. **Degradation conditioning:** implemented; constant, estimated, and oracle
   arms are fixed in r31. Compare reference-assisted damage maps with local
   input-only estimates; never deploy oracle information.
6. **Recurrent frame feeding:** frozen and joint/perceptual probes are complete,
   including held-out Sintel and REDS. History does not improve over the trained
   current-frame branch; native integration remains untested. Warp previous
   predictions using decoded-input motion.
7. **Confidence head:** implemented and first 2k proxy probe complete; learned
   policies worsen perceptual quality. Native integration is not justified yet.
8. **Clean-LR consistency and AESOP:** implemented and unit-tested; r36 fixes
   matched loss ablations. Explicit full-chroma resize differs from pre-encode YUV.
9. **Larger base pretraining bank:** assembled and evaluated against a matched
   original-bank control, including native output. Expanded-critic and equal-source
   follow-ups investigate the observed distribution tradeoff. Preserve
   source-disjoint evaluation and licence provenance.

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

The r29 output-only arm improves native LPIPS 1.27% and DISTS 2.40% against its
matched control, the first joint native aggregate improvement in these ablations.
Against shipping it improves 1.41%/2.66%, but Rush Hour LPIPS worsens 18.27% and
Sunflower 24.79%, so it cannot ship. `output-stage-native-comparison.json` records
all source deltas. The input-stage ablation did not supply this gain. The fixed
r33 follow-up repeats control/output-only training from current shipping `big2k`,
rather than carrying the v4 initialization's large source-specific regressions.
The r33 script is prepared; it has not yet been launched.

The combined r29 arm has finished native export and RGB decoding: 1,920 frozen
rows (960 candidate and 960 comparator) are ready in
`.build/quality-breakthrough-r29/full_stages_1k-native-rgb`. Its spatial scoring
is still pending. All training/export/scoring processes started for the current
checkpoint have exited. The next checks are that combined-arm score, the r33
shipping-initialized output-stage ablation, native pre-cleaner evaluation, and
the remaining confidence/consistency/AESOP/larger-bank items.

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

The completed r32 probe preserves the frozen backbone. Validation on all 16
frames of each source-disjoint bank patch sequence reduces mean RGB MSE 1.03%,
last-frame MSE 1.06%, and temporal residual-change L1 0.73%. Spatial samples at
the first/middle/final frames improve PSNR by 0.052 dB but worsen LPIPS 2.293%
and DISTS 0.136%. No promotion. `recurrent-screen.json` records the experiment
and deltas. This frozen-backbone reconstruction-loss probe does not rule out
joint or perceptually supervised recurrent training.

## Completed combined-stage and cleaner native results

The r29 combined source/output-stage arm improves native source-balanced LPIPS
1.760% and DISTS 4.461% against current `big2k`, and 1.619% / 4.213% against its
matched r28 control. It still worsens Sunflower LPIPS 25.85%, RushHour 8.90%, and
PedestrianArea 3.43% against `big2k`. No promotion. The first-frame sampler and
initial model differ from shipping's original training; r33 repeats the control
and output-only ablation from actual shipping weights, and r35 adds both stages
using the same immutable r33 code and schedule. `full-stage-native-comparison.json`
contains the complete comparison.

The native-input pre-cleaner's completed 960-frame sender-output comparison
improves DISTS 0.683% but worsens LPIPS 0.168% against its frozen `big2k` backbone.
The cleaner improves its low-resolution reconstruction objective, but does not
establish a joint perceptual improvement after playback stages. No promotion;
`precleaner-native-comparison.json` records the result. Joint cleaner/SR training
remains untested.

## Confidence head: implemented, first probe rejected

`ConfidenceSPAN` adds a 4,624-parameter log-variance head to frozen SR features.
Its mean prediction is unchanged. The head learns Gaussian RGB error NLL with
the mean detached, bounds log variance to [-12, 0], and uses a fixed confidence
mapping `0.0025 / (variance + 0.0025)`. r34 trains 2,000 steps from current `big2k`
on three-frame native-input proxy sequences. Only the head changes.

Current-frame confidence controls post-SR sharpening. A separate causal policy
warps the previous frame's confidence to control pre-SR debanding; cuts, unknown
history and rejected correspondence fall back to normal shipping strength.
Debanding never uses a future or not-yet-computed SR prediction. Validation
compares both policies with fixed shipping strengths and a constant half-strength
control over all 16 frames of each validation patch sequence.

Post-only confidence worsens LPIPS 0.811% / DISTS 0.154%; the causal policy worsens
0.651% / 0.192%. Even the constant half-strength control scores better than both
learned policies on these perceptual metrics. Calibration also underestimates
error in the populated 0.05–0.10 and 0.10–0.20 predicted-standard-deviation bins.
`confidence-screen.json` includes calibration, temporal summaries and provenance.
Seven tests cover mean identity, frozen gradients, analytic NLL, stage ordering,
causality, reset fallback and checkpoint loading. This is a tested proxy, not a
native confidence-output model or evidence for shipping it. No promotion.

## Clean LR and AESOP fidelity implementation

The clean-LR consistency term compares downsampled prediction with downsampled HR,
not with the codec-damaged decoded input. Both use the same half-pixel Lanczos-3
RGB operator, straight-through RGB8 quantization and four-pixel LR border exclusion.
Its interior agrees with FFmpeg `lanczos+full_chroma_inp` within one RGB level on
three random-image geometries. Plain FFmpeg `lanczos` subsamples input chroma
internally; the older pre-cleaner target cache therefore is deliberately not reused.
This is an explicit clean RGB surrogate, not a claim of recovering bit-exact
pre-encode YUV samples. FFmpeg's [RGB input conversion](https://github.com/FFmpeg/FFmpeg/blob/master/libswscale/input.c)
contains the half-chroma conversion path that motivated this distinction.

AESOP uses the [authors' published implementation](https://github.com/2minkyulee/AESOP-SR)
and synthetic 100,000-step autoencoder checkpoint, pinned by SHA256. It compares
L1 **after the decoder**, with frozen weights and a detached target branch;
prediction gradients pass through the full autoencoder in FP32. The published
4x-bottleneck teacher accepts same-size input/output images for this 2x SR loss.
It has not been pretrained locally for the 2x codec domain. The vendored graph
preserves state keys, records its upstream commit and Apache 2.0 licence, and
loads `params_ema` strictly. `aesop-provenance.json` binds the source and weights.

When enabled, AESOP replaces both existing pixel L1 terms with 1.1× decoded-image
L1; signed Sobel, FFT and paired-critic losses remain. Teacher initialization
preserves the discriminator RNG. Default training retains the original floating-
point operation order. Six tests cover this control, replacement semantics,
frozen/target gradients, rejection of bottleneck comparison and clean-LR parity.
r36 fixes three independent 1,000-step arms from `big2k`: clean-LR weight 1,
AESOP alone, and both. The reference control is r33's matched `big_control_1k`.
The teachers add no SR inference cost. Training and native quality remain to be
measured; implementing the losses is not a quality claim.

## Larger-bank base pretraining: data and memory implementation

r37 expands the existing pinned REDS selection from eight to 60 sequences and
uses all 100 sharp frames per sequence. The same seeded ordering preserves the
first two validation IDs; REDS4 remains excluded. The public archive is fetched
by bounded ranges with each frame's exact ZIP identity, size and CRC checked,
then SHA256 receipts. Grouped ranges reduce request overhead while preserving
these checks and resuming completed frames. Lossless FFV1 masters must pass an
RGB comparison for every input PNG. This acquisition uses `D:` on the authorized
worker, which had about 1.3 TiB free, rather than its nearly full system drive.
The official dataset is CC BY 4.0; the original provenance remains in
`reds-archive-receipt.json` and `reds-training-subset.md`.

`build_large_stream_bank.py` freezes a source/recipe specification and retains
checked per-source receipts for interruption recovery. Planned expansion is
18 full-frame codec windows × 10 spatial patches × 58 training REDS sequences
= 10,440 new 16-frame examples. Composing with v4's 1,152 training examples gives
11,592, or 10.06× v4. This is 10× patch-sequence count, not 10× independent footage;
windows can overlap within a clip. It also changes source distribution and
master quality. Those effects cannot be attributed to count alone. The first
two REDS sequences and all original v4 validation sources stay out of training.

The uncompressed pairs exceed the worker's 32 GiB RAM. `mmap_training_bank.py`
materializes checked `.npy` pairs, preserves source splits and publishes the
manifest only when complete. The loader validates bytes, maps read-only arrays
and copies only sampled crops. Reference-only supervision reuses the sampled HR
target rather than eagerly duplicating every sequence. Four tests verify exact
sampler/RNG equivalence, read-only memory maps, source/sequence conflicts,
changed arrays and incomplete-bank rejection. No completed larger bank or
pretraining quality result is claimed yet.

The completed r33 native follow-up from current shipping weights also fails the
per-source requirement. Its raw-objective control improves mean LPIPS 0.474% and
DISTS 0.223% over shipping, but worsens RushHour LPIPS 13.43% and Sunflower 14.66%.
The output-stage arm worsens mean LPIPS 0.056% while improving DISTS 2.733% over
shipping; Sunflower LPIPS worsens 20.07%, RushHour 11.88% and PedestrianArea 7.12%.
Against its control, output-stage training worsens LPIPS 0.532% and improves
DISTS 2.516%. Initialization alone does not resolve the regressions. Neither is
promoted. `shipping-init-output-native-comparison.json` records both arms.

r36's three loss ablations completed with identical initialization, first sampled
sequence, first model output and initial discriminator hashes. On development48,
clean LR alone worsens LPIPS 1.166% / DISTS 1.761% against the matched control;
AESOP improves 1.599% / 1.121%; the combination improves LPIPS 0.562% but worsens
DISTS 0.746%. AESOP alone improves 4.684% / 5.227% against shipping on that screen,
with all three sources improving. It is selected for native measurement, without
changing its weights or stage settings. `fidelity-development-screen.json` binds
all three reports. These reused development samples are not release evidence.

r38 fixes the larger-bank base control before the new-bank outcome: original
pretrained SPAN weights, folded to direct 2x, 40,000 reconstruction steps at
batch 16/crop 96/AdamW 1e-4/cosine decay/seed 20260918, followed by 2,000 paired-
critic steps at batch 4 and learning rate 2e-5. The critic weight is 0.0075.
Both final critic passes use the same original v4 data, so that comparison
isolates the changed **base-pretraining** bank rather than changing both stages
at once. The larger arm must use the identical architecture, initialization,
schedule and objectives. This is continuation of pretrained features, not
training a new model from random initialization. The initial mapped v4 bank
contains exactly the original pairs; materialization completed successfully.

r35's combined-stage follow-up from `big2k` is complete: native mean LPIPS worsens
0.100% and DISTS improves 3.935% against shipping. It worsens Sunflower LPIPS
19.45%, PedestrianArea 7.41% and RushHour 6.88%. Relative to the matched control,
LPIPS worsens 0.577% and DISTS improves 3.720%. No promotion;
`shipping-init-full-stage-native-comparison.json` contains the source results.

AESOP's completed native run improves aggregate LPIPS only 0.496% and DISTS
0.019% against shipping. It worsens RushHour LPIPS 24.74% / DISTS 17.99%, Sunflower
26.02% / 8.02%, and PedestrianArea 5.72% / 9.84%. Against the matched control,
LPIPS improves 0.023% while DISTS worsens 0.204%. The development advantage does
not establish native quality. No promotion; `aesop-native-comparison.json`
records the complete comparison. This first fixed-weight probe does not rule out
joint stage training, domain-specific AE pretraining or other fidelity weights.

All 6,000 r37 sharp frames and all 60 lossless masters have passed their checks.
The first eight source IDs retain the earlier order and REDS 154/073 remain the
two validation sources. `reds-expansion-sources.json` records their provenance
and frame/master receipt hashes. Expanded codec construction is now separate
from acquisition, and remains incomplete until its final manifest is published.

The roomy `D:` volume is a rotational disk. Concurrent corpus writes exposed a
random-read bottleneck in the first mapped-bank control. r38 was stopped early
without selecting or interpreting its partial weights. r38b rebuilds the same
v4 pairs on the `I:` SSD, checks that the manifest hash is identical, and restarts
the full unchanged seed/schedule from the original pretrained initialization.
The expanded mapped bank will also use that SSD; sequential source preparation
remains on `D:`. The available SSD space was about 122 GiB before these copies.

## Training arithmetic follow-up

A fixed 48-crop diagnostic compared the same `big2k` weights with unfused
training and fused evaluation graphs. Against fused FP32 evaluation, BF16
training arithmetic differs by 0.328 RGB levels on average; FP16 differs by
0.039 and FP32 by 0.017. These are forward differences, not quality improvements.
The receipt defines its aggregation explicitly: `max_rgb` is the mean of each
crop's maximum, not the maximum over the dataset.

r40 fixes a matched 1,000-step BF16/FP32 SR-forward ablation from `big2k`, using
the r33 sampler, original v4 data, seed and objectives. The paired critic remains
BF16 in both arms. `--sr-precision fp32` changes only SR, EMA and temporal model
forwards; the default preserves existing arithmetic. The r38b/r39 larger-bank
comparison keeps its original immutable trainer snapshot. Numerical agreement
alone cannot establish native quality; this probe must be evaluated normally.

The completed FP32 arm worsens development LPIPS 0.411% and DISTS 0.064% against
its new BF16 control. Initialization, bank, first sampled tensors and initial
critic hashes match; only the intended first SR output differs. There is no
measured advantage supporting further native work on this precision setting.
The new BF16 control also differs from r33 despite matching those initial
hashes. Its first critic gradient differs in the last floating-point bit and
later optimization diverges. The default reconstruction formula is unit-tested
bit-exact, but code snapshots differ; this is not a clean same-binary variance
estimate. Small single-run effects should not be treated as reliable gains.
Both comparisons are retained as `sr-precision-development-comparison.json`
and `bf16-control-repeat-comparison.json`.

The 4k LDL student now has a complete native result: aggregate LPIPS worsens
1.240% and DISTS 0.152% against shipping. RushHour worsens 30.25% / 21.91%;
Sunflower worsens 23.81% / 7.78%. Direct crop review shows rougher small-car
boundaries and texture changes, without an obvious detail breakthrough.
`ldlref100-4k-student-native-comparison.json` records all eight sources. The EMA
native comparison remains in progress.

## Joint recurrence follow-up

r41 removes the frozen-backbone limitation of the first recurrent probe.
Two 2,000-step arms optimize the same fused SR graph with identical source-stage
inputs, final-frame reconstruction and paired critic weight 0.0075. The control
disables history; the recurrent arm also learns the zero-initialized history
convolution. Backbone learning rate is 2e-5 and history rate 2e-4 with cosine
decay. SR and warps remain FP32; the training-only critic uses BF16. This is a
matched test of recurrence under joint perceptual training, not an isolated
comparison of fused and reparameterized optimization.

Validation includes the unchanged starting model, the trained current-frame
branch and the full 16-frame recurrent sequence. In the control, the latter
explicitly has history disabled. Nine tests now cover the original frozen
behavior, joint gradients, and independence from earlier frames in the control.
Positive patch evidence would still require native state, warp-cost and visual
validation before any deployment.

r41 is complete. Its original bank validation has one source, Sintel. A second
evaluation adds REDS 154/073, selecting the first patch from all 18 codec windows
per source before looking at the recurrent outcome. Source identity, family and
master hashes are checked against training to prevent overlap. Across these
three sources, recurrence improves LPIPS 0.452% against its matched control but
worsens DISTS 0.753% and fine correlation 0.863%. Disabling history in the same
trained checkpoint slightly improves all three aggregate measurements. Direct
unscaled crop review shows no substantial recovery of missing detail. The
comparison is retained in `joint-recurrence-comparison.json`; native integration
remains deferred. This does not rule out recurrence with broader training data
or a different correspondence method. The joint checkpoint round-trip is also
tested after both backbone and history weights have actually been updated.

The completed 4k LDL EMA native comparison worsens LPIPS 0.845% while improving
DISTS only 0.155%. RushHour LPIPS worsens 16.18% and Sunflower 15.52%. Together
with the student result, this closes the missing 4k native measurement without
establishing a breakthrough. `ldlref100-4k-ema-native-comparison.json` records all
sources. The matched 40k-base/2k-critic control is now being measured natively.

The expanded mapped bank is complete: 11,592 training and 396 validation
sequences from 93 sources, with Sintel and REDS 154/073 held out. Its manifest
hash is `4f7ce2f230ee57dd60eabcbad3af7fb63e74e77992bcd650d206e01589feeae7`.
`expanded-bank-receipt.json` binds the original and expanded banks, composition,
geometry and materializer. r39 has started from the same pretrained checkpoint
and immutable training code as r38b. Matching 40k steps tests the bank change
at fixed training compute; it does not give the 10× larger bank equal epochs.
Longer training would be a separate compute-and-data experiment.

r38b's native control improves aggregate LPIPS 1.179% but worsens DISTS 3.413%
against shipping. Sunflower worsens LPIPS 41.30% / DISTS 29.12%, and RushHour
19.55% / 11.22%. Crop review shows somewhat crisper architecture but rougher
outlines on the bee and small cars, without recovered fine structure.
`base-control-native-comparison.json` preserves the complete comparison. The
expanded arm must beat shipping in useful delivered quality, not merely improve
this control's weaker result.

## Combined perceptual fidelity follow-up

The pinned [AESOP RRDB recipe](https://github.com/2minkyulee/AESOP-SR/blob/3d6fe1d95a0a2fbaf2365861b0ce9f725985c498/AESOP/options/train/AESOP/train_Synthetic_AESOP_RRDB.yml)
also uses multi-layer pre-ReLU VGG19 feature L1, artifact loss and EMA. r36
isolated the autoencoder replacement within Lucid's existing objective; it did
not reproduce that complete recipe. Earlier VGG work used weight 0.025 with a
different training route, before the current paired-critic and AESOP trials.

r42 prepares three fixed 1,000-step arms from `big2k`: VGG weight 1 + LDL weight 1;
the same with AESOP replacing pixel L1; and that combination with both native
stage proxies. All use the three-frame sampler and existing paired DINO critic
weight 0.0075. VGG construction preserves the critic RNG and has no inference
cost. Its frozen feature-state hash, layer weights and source are recorded.
This remains a local adaptation: SPAN, codec degradation, Sobel/FFT terms, 1.1
AESOP coefficient, critic, step budget and optional stages differ from the paper.
The three arms are prepared separately from the running immutable r39 snapshot.
Eight fidelity tests now include teacher freezing, prediction gradients, disabled
behavior and RNG preservation; 29 relevant tests pass together.

The fixed-compute larger-bank comparison is complete. r39 matches the r38b
trainer hashes and base initialization exactly; the final critic's first sample
and discriminator initialization also match. Development LPIPS/DISTS worsen
0.402%/1.194% against the control. Native regression worsens 3.823%/2.575% against
the control and 2.599%/6.075% against shipping. Sunflower worsens 61.21%/36.85%
against shipping. Direct crops show no recovered-detail breakthrough.

The same expanded checkpoint does improve held-out REDS 154/073: on all 720
stride-8 validation patches, raw LPIPS/DISTS improve 2.586%/2.474% against the
control and 3.289%/5.212% against shipping. Both REDS sources improve. This is a
distribution-dependent gain, not a universal failure of broader training data,
and it does not override the native regressions. The `expanded-bank-*` receipts
retain initialization parity, development, native and REDS comparisons.

r43 tests the next data change: start from the identical completed r39 base,
but perform the final 2k critic pass on the expanded bank instead of original
v4. Code, seed, batch, schedule and objectives remain fixed. This distinguishes
the original isolated base-pretraining test from adapting both training stages
to the expanded data.

r42's three development runs are complete with matching initial model, source
sample and discriminator hashes. VGG+LDL worsens LPIPS/DISTS 7.981%/9.497% against
shipping; adding AESOP worsens 8.263%/9.733%; adding both native stage proxies
worsens raw-output scores 12.235%/9.600%. These are not native measurements of
the stage-trained arm; its native run remains active. The VGG term dominates the
initial head gradient at this coefficient, so this probes a substantially
different objective rather than establishing an optimum loss balance.
`combined-fidelity-development.json` records the full results and teacher hashes.

r42's stage-trained combination is now measured natively. It worsens aggregate
LPIPS 8.665% and DISTS 6.460% against shipping. Direct crops show softened
architectural structure; the small LPIPS gains on RushHour and Sunflower do not
establish better overall delivered quality. No promotion;
`combined-fidelity-native-comparison.json` records every source.

r43's expanded-bank critic improves raw REDS LPIPS 15.871% and DISTS 12.398%
against shipping, while fine correlation falls 8.288%. Development LPIPS improves
6.560% and DISTS 0.976%, with fine correlation down 8.649%. Raw crops show stronger
palm and chair structure alongside extra texture. Native evaluation remains
active; the correlation change is a diagnostic to investigate visually, not an
automatic veto. The starting base weights, training code and discriminator
initialization match r39's original-bank critic exactly.

The composed bank has 36 sequences per original training source and 180 per REDS
source. Uniform sequence sampling therefore gives each REDS source five times
the weight of an original source. `--source-balanced` instead samples all 90
training sources equally, then their own sequences equally, without copying
image arrays. An exact bounded virtual index retains one RNG draw per example;
its 16,200 sampling bins are not additional training data. Three tests verify
uniform probabilities, unchanged batches/RNG for already balanced inputs, and
split/identity/order rejection. Together with the affected training tests, 27
tests pass. The earlier full relevant suite passed 65 tests before this addition.

r44 fixes two source-balanced 2k critic continuations with the existing recipe:
one from the r39 reconstruction base and one from current `big2k`. These test
data balance and whether the longer pixel-base stage was needed. Model geometry
and inference operations remain unchanged. The pending optional 60-versus-30fps
preference does not affect these same-architecture experiments; the current
performance target remains in force.

The r43 native measurement is complete: source-balanced LPIPS worsens 15.373%
and DISTS 26.809% against shipping. ParkJoy improves 19.901%/18.775% and InToTree
17.962%/15.619%; unscaled crops show more separated foliage structure. The same
model adds conspicuous speckling around the bee and small cars, with Sunflower
worsening 241.050%/142.664% and RushHour 113.732%/109.334%. Both the original
three-crop gallery and additional natural-scene crops were reviewed, so the
decision reflects visible gains and defects rather than aggregate scores alone.
No promotion. `expanded-critic-native-comparison.json` retains all eight sources.
All 68 relevant unit tests pass after the source-sampling addition.
