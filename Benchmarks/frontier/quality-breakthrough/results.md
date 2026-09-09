# Ranked quality experiment results

All nine suggestions have working experimental implementations and completed
initial comparisons. No candidate has yet demonstrated a broad native quality
breakthrough. The repository app now defaults to full-search TAA motion with
the same `big2k` model and `debandGuard = 0.005`. This incremental change improves
native LPIPS/DISTS by 0.588%/1.340%; both distances improve on all eight sources.

That 0.588%/1.340% is the improvement attributable to this ranked continuation,
relative to the `big2k`/guard/legacy-motion baseline already present when the
native Codex goal began. Against the older SPAN4x predecessor, the current
Release output improves LPIPS/DISTS by 12.779%/17.226%; most of that improvement
predates this goal. These are reductions in perceptual distances, not literal
percentages of visual quality. The source and local Release build are updated;
the installed application has not been replaced. No new learned weights have
been promoted during this continuation.

Measurements below supplement direct image review. Lower LPIPS/DISTS is better;
small changes from a single training run are not strong evidence by themselves.
Native comparisons use 960 frames across eight regression sources, unless noted.
These repeatedly examined sources are regression evidence, not a fresh blind test.
Historical comparisons below use the preceding `big2k`/guard/legacy-motion
baseline unless explicitly stated otherwise.

| Rank | Implemented experiment | Completed evidence and current conclusion |
| --- | --- | --- |
| 1 | Reference LDL artifact map, differentiable residual weighting, EMA checkpoints | Student and EMA at 2k/4k evaluated. Native 4k student LPIPS +1.240%, DISTS +0.152%; EMA +0.845%/−0.155%. Visible rough outlines remain. [Receipt](ldlref100-4k-student-native-comparison.json) |
| 2 | Four-orientation cycling and actual static four-pass ensemble | Native accumulator precedes SR, so cycling cannot average model predictions. Actual ensemble also worsens raw development quality. Disabled. [Receipt](static-flip-ensemble-comparison.json) |
| 3 | Differentiable source deband/TAA and output stages, three-frame training | Proxy parity checked against Metal. Shipping-initialized full-stage native result +0.101%/−3.935%, with visible/source-level tradeoffs. [Receipt](shipping-init-full-stage-native-comparison.json) |
| 4 | Identity-initialized supervised clean-LR pre-cleaner | Clean-LR MSE improves about 5%; native LPIPS +0.168%, DISTS −0.683%. No visible breakthrough. [Receipt](precleaner-native-comparison.json) |
| 5 | Constant, estimated and reference-oracle degradation conditioning, frozen and joint backbone training | Frozen probes worsen perceptual distances. Joint estimated conditioning versus its matched SR-only control is effectively flat: LPIPS +0.016%/DISTS −0.056%; reference-only oracle −0.308%/−0.229%. Fixed visual differences remain small. No native promotion. [Joint receipt](joint-conditioning-comparison.json) |
| 6 | Warped previous-output recurrence, frozen and jointly trained backbone | Joint perceptual follow-up covers Sintel plus held-out REDS. Recurrence worsens LPIPS 0.039%/DISTS 0.307% versus its own trained current-frame branch. No native integration. [Receipt](joint-recurrence-comparison.json) |
| 7 | Gaussian-NLL confidence head controlling spatial stage strength | Constant/post/causal policies all worsen proxy perceptual quality. Calibration remains weak; no native integration. [Receipt](confidence-screen.json) |
| 8 | Clean-LR consistency, pinned AESOP teacher, combined VGG/LDL follow-up | AESOP native −0.496%/−0.019% versus shipping, with substantial individual-source regressions. Combined full-stage fidelity native +8.665%/+6.460%. [Receipt](aesop-native-comparison.json) |
| 9 | 60 additional source clips, 10.06× training-sequence bank, mapped loader | Fixed-compute larger base improves raw REDS but worsens native output. Adapting the critic too improves native foliage while adding conspicuous texture elsewhere. Both equal-source arms are natively evaluated. Direct continuation from shipping improves four native sources on both distances, but aggregate LPIPS/DISTS worsen 9.557%/10.792%; visible texture artifacts persist. [Receipt](expanded-critic-native-comparison.json) |

The expanded bank contains 11,592 training sequences across 90 training sources;
its 10× count includes overlapping windows and is not 10× independent footage.
The source-balanced sampler corrects unequal per-source sequence counts without
copying image arrays. All 68 relevant unit tests pass. Tests establish software
behavior, not perceptual improvement.

Unpromoted architecture experiments remain opt-in research tools. The motion
change applies globally; no model picker or content-specific policy was added.
See [the protocol](../quality-breakthrough-protocol.md) for recipes, provenance,
matched controls, limitations, and detailed follow-up results.

Combining source-balanced broader data with reference LDL100 and EMA improves
all five raw development/REDS sources on LPIPS and DISTS and visibly reduces
false texture. Native output remains mixed: EMA is +2.822%/+2.669% versus
shipping; student +5.945%/+6.078%. Both improve Ducks, OldTown, ParkJoy and
InToTree, while the other four sources regress. [EMA native evidence](balanced-ldl-ema-native-comparison.json).

A separate fixed three-patch temporal diagnostic finds that a .25 blend of
aligned previous clean HR improves MSE 9–12%, while blending previous SR
generally does not. Clean HR is strictly an oracle diagnostic, not an inference
input. This motivates investigating better preservation of observed information
in temporal state; it does not establish a deployable temporal quality gain.
[Transition-level evidence](history-information-diagnostic.json).

A matched decoded-observation history trial also fails to establish a temporal
quality advantage: enabling history worsens LPIPS/DISTS by 0.140%/0.464%
against its own trained current-frame branch. Generated-SR history worsens them
by 0.450%/0.062%. [Matched comparison](decoded-history-comparison.json).

Investigation identifies and corrects a recurrent motion-seed defect that
discarded valid low-contrast translations before subpixel refinement. Native TAA
and archived checkpoint semantics remain compatible; 70 relevant tests pass.
Fixed-weight replay reproduces all legacy results exactly. The correction
improves LPIPS by only about 0.018%, with subtle visual changes and no promotion.
[Replay evidence](recurrent-motion-replay-comparison.json). The previous-clean-HR
diagnostic confirms substantially better alignment on the fixed low-contrast
Sintel patch, but this oracle result is not a deployable quality gain.
[Alignment comparison](recurrent-motion-information-comparison.json).

Training both temporal models from the start with corrected motion also fails
to establish a quality advantage. Across the same 72 sequences, history worsens
LPIPS/DISTS versus each model's own current-frame branch: SR +0.456%/+0.170%,
decoded +0.883%/+0.551%. Fixed visual comparisons show small changes without a
clear detail breakthrough. [Corrected training](corrected-motion-training-comparison.json).

Joint clean-LR/SR training also provides no substantial gain: LPIPS/DISTS improve
only 0.063%/0.063% versus its matched SR-only control, with small visual changes.
[Joint cleaner evidence](joint-precleaner-comparison.json). A separate TAA motion
policy probe gives a stronger improvement on one fixed low-contrast patch;
broader comparison is required before interpreting it as a quality breakthrough.

The full TAA proxy replay gives modest overall gains: preserving the match-gain guard
improves LPIPS/DISTS by 0.221%/0.254%; retaining all integer matches improves
them by 0.298%/0.345%. Baseline rows reproduce exactly. The larger fixed-patch
gain is localized.
[TAA policy comparison](taa-motion-proxy-comparison.json).

Native follow-up verifies the same default RGB bytes on all 960 baseline frames.
The gain-check alternative improves LPIPS/DISTS by 0.281%/0.645%; full search
improves them by 0.588%/1.340%, with improvements in both distances across all
eight sources. Fixed images show small local changes, not dramatic recovery of
reference detail. [Native comparison](taa-motion-native-comparison.json).

On three selected consecutive-frame cases, full search lowers RGB/luma MSE but
raises adjacent-frame residual-change L1 by 0.339%. This diagnostic tradeoff is
recorded alongside the spatial improvement; it is not a perceptual flicker
score. [Temporal evidence](taa-motion-native-temporal.json). Counterbalanced
enhancement-stage timing averages 8.72 ms for full search, with a maximum run
p95 of 10.52 ms. Browser transport and presentation are outside that measurement.

Full search is now the single default in the source and rebuilt Release app.
Without an environment override, the rebuilt default reproduces all 360
consecutive full-search images exactly. The installed app was not replaced.
[Default verification and decision](taa-motion-default-admission.json).

Joint conditioning now has four completed matched 2,000-step arms using the
updated full-search source proxy. All 216 initial metric rows match exactly
across arms and reproduce the earlier full-search replay. The SR-only control
improves LPIPS/DISTS by 0.735%/1.758% versus its initialization; almost all of the
conditioned models' gains versus initialization are therefore shared with
ordinary continued SR training. Constant conditioning versus that control is
−0.050%/+0.035%, estimated +0.016%/−0.056%, oracle −0.308%/−0.229%.
Estimated-map calibration improves, but does not translate into a substantial
reconstruction gain in this recipe. Fixed Sintel/palm/chair patches remain
visually close to the SR-only control. The oracle is reference-only and cannot
be deployed; none of these four arms has native quality or runtime admission.
Twenty-seven focused tests pass locally and on the GPU host, followed by a
real CUDA forward/backward smoke check. All training processes have exited.
[Joint conditioning evidence](joint-conditioning-comparison.json).

Crossing raw versus full-search-filtered inputs with decoded history also fails
to establish a recurrence gain. History worsens LPIPS/DISTS versus its own
current-frame branch in both cases: filtered+0.383%/+0.528%, raw+0.155%/+0.095%.
Fixed images remain close to the controls. The raw SR-only control is more
promising: versus unchanged big2k with current preprocessing, development
LPIPS/DISTS improve1.523%/2.007%, with both better on all three sources, although
fine correlation falls1.670%. It proceeds to native comparison alongside an
unchanged-weight raw-pipeline control; no new weights or settings are promoted
from these proxy results. [Matched four-arm evidence](raw-filtered-recurrence-comparison.json).

The raw-input native follow-up is now complete. Disabling deband/TAA and the
associated output grain with unchanged big2k worsens LPIPS/DISTS by7.262%/4.793%
versus the current full-search default. The trained raw-input control recovers
0.994%/1.097% relative to that raw configuration, but remains6.196%/3.643% worse
than the current default. Ducks, ParkJoy and InToTree improve on both distances;
Sunflower worsens45.882%/25.887%. Six fixed native crops show small local changes
without broad recovery of reference detail. Neither model nor settings are
promoted. [Completed native evidence](raw-input-native-comparison.json).
