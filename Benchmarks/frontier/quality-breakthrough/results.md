# Ranked quality experiment results

All nine suggestions have working experimental implementations and completed
initial comparisons. No candidate has yet demonstrated a broad native quality
breakthrough. Shipping remains `big2k` with `debandGuard = 0.005`.

Measurements below supplement direct image review. Lower LPIPS/DISTS is better;
small changes from a single training run are not strong evidence by themselves.
Native comparisons use 960 frames across eight regression sources, unless noted.
These repeatedly examined sources are regression evidence, not a fresh blind test.

| Rank | Implemented experiment | Completed evidence and current conclusion |
| --- | --- | --- |
| 1 | Reference LDL artifact map, differentiable residual weighting, EMA checkpoints | Student and EMA at 2k/4k evaluated. Native 4k student LPIPS +1.240%, DISTS +0.152%; EMA +0.845%/−0.155%. Visible rough outlines remain. [Receipt](ldlref100-4k-student-native-comparison.json) |
| 2 | Four-orientation cycling and actual static four-pass ensemble | Native accumulator precedes SR, so cycling cannot average model predictions. Actual ensemble also worsens raw development quality. Disabled. [Receipt](static-flip-ensemble-comparison.json) |
| 3 | Differentiable source deband/TAA and output stages, three-frame training | Proxy parity checked against Metal. Shipping-initialized full-stage native result +0.101%/−3.935%, with visible/source-level tradeoffs. [Receipt](shipping-init-full-stage-native-comparison.json) |
| 4 | Identity-initialized supervised clean-LR pre-cleaner | Clean-LR MSE improves about 5%; native LPIPS +0.168%, DISTS −0.683%. No visible breakthrough. [Receipt](precleaner-native-comparison.json) |
| 5 | Constant, estimated and reference-oracle degradation conditioning | All three frozen-backbone development probes worsen perceptual distances. Joint backbone adaptation remains untested. [Receipt](degradation-conditioning-screen.json) |
| 6 | Warped previous-output recurrence, frozen and jointly trained backbone | Joint perceptual follow-up covers Sintel plus held-out REDS. Recurrence worsens LPIPS 0.039%/DISTS 0.307% versus its own trained current-frame branch. No native integration. [Receipt](joint-recurrence-comparison.json) |
| 7 | Gaussian-NLL confidence head controlling spatial stage strength | Constant/post/causal policies all worsen proxy perceptual quality. Calibration remains weak; no native integration. [Receipt](confidence-screen.json) |
| 8 | Clean-LR consistency, pinned AESOP teacher, combined VGG/LDL follow-up | AESOP native −0.496%/−0.019% versus shipping, with substantial individual-source regressions. Combined full-stage fidelity native +8.665%/+6.460%. [Receipt](aesop-native-comparison.json) |
| 9 | 60 additional source clips, 10.06× training-sequence bank, mapped loader | Fixed-compute larger base improves raw REDS but worsens native output. Adapting the critic too improves native foliage while adding conspicuous texture elsewhere. Both equal-source arms are natively evaluated. Direct continuation from shipping improves four native sources on both distances, but aggregate LPIPS/DISTS worsen 9.557%/10.792%; visible texture artifacts persist. [Receipt](expanded-critic-native-comparison.json) |

The expanded bank contains 11,592 training sequences across 90 training sources;
its 10× count includes overlapping windows and is not 10× independent footage.
The source-balanced sampler corrects unequal per-source sequence counts without
copying image arrays. All 68 relevant unit tests pass. Tests establish software
behavior, not perceptual improvement.

Architecture and pipeline experiments remain opt-in research tools. No model
picker, content-specific deployment policy, or shipping asset has been added.
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
