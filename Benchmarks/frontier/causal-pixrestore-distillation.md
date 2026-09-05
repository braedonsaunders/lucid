# Reference-checked PixRestore distillation pilot

The August 2026 PixRestore-S teacher, evaluated at its trained 512-pixel size, improves development LPIPS/DISTS substantially. The fast causal student has not matched that quality; doubling its width gave negligible improvement over a matched control. This experiment tests whether teacher outputs improve the existing fast student without adding inference computation.

The teacher uses the exact pinned upstream code, configuration, DINO encoder and weights from the measured baseline. `cache_pixrestore_teacher.py` verifies these identities before loading. It processes only training sequences; the cache records every output hash and the source-bank manifest hash. Validation and evaluation clips are not teacher-training targets.

The new context bank retains the same 21 source identities and family split as the previous bank. It has 160 training / eight validation sequences, each with 16 aligned frames and 256-pixel LR / 512-pixel HR patches, encoded at full-frame size before cropping. This supplies the teacher's trained context size. Larger patches do **not** add independent scene diversity: there are still 42 source windows with four correlated crops each. The original training and shipping assets remain intact.

Two 4,000-step runs start from the same 32-channel checkpoint, seed 20260913, learning rate 0.0001, batch four, crop 96, curriculum and DINO weight 0.03. Both load the same cache and apply exactly the same crop, flip, transpose and frame selection. The candidate adds teacher weight one; the control uses zero. This is a pilot, not a replacement claim.

For a selected frame, the teacher's and fixed Lanczos floor's local mean absolute RGB errors are measured against the known training reference, averaged over a 9×9 neighborhood. Confidence is `clamp((floor_error - teacher_error) / (floor_error + 0.001), 0, 1)`. The added objective is mean confidence-weighted absolute student/teacher error. Confidence and teacher pixels are detached. Regions where the teacher is less accurate than interpolation contribute nothing; the original reference objectives remain active everywhere. This guard is not a guarantee of perceptual accuracy.

The teacher is cached offline and discarded from student inference. The learned graph, persistent Core ML state, parameter count and deployment interface are unchanged. This does not establish that the compressed student preserves the teacher's quality; that is the experiment's measured question.

Eighteen experiment tests pass, including aligned cache augmentation, rejection of wrong-bank or validation targets, rejection of changed cache bytes, and gradient checks for useful versus harmful teacher regions. The remote command is `causal-pixrestore-distillation-command.ps1`; launch/source identities are in `causal-pixrestore-distillation-launch.json`.

## Matched pilot result

The RTX 4080 cached all 2,560 training targets, then completed the candidate/control in 3.74/3.60 minutes, both with exit code zero. `causal-pixrestore-matched-evaluation.json` records the completed development screen:

| Variant | LPIPS ↓ | DISTS ↓ | Fine correlation ↑ |
|---|---:|---:|---:|
| Lanczos | 0.345604 | 0.128056 | 0.509324 |
| Matched control | 0.338843 | 0.127766 | 0.512886 |
| Teacher weight 1 | 0.338704 | 0.127661 | 0.512914 |

The teacher adds only **0.041% LPIPS** and **0.082% DISTS** improvement over the control. Most improvement over the initialization comes from further ordinary training. This candidate is not promoted.

`causal-teacher-gradient-diagnostic.json` examines four training-only sequences at fixed initialization, in FP32 on CPU. Confidence averages 0.010–0.050; the added teacher gradient norm is only 4.6–15.1% of the L1 gradient norm. This comparison omits the other reference/DINO terms and is not a full-objective gradient ratio. A weight-10 run tests whether stronger supervision within the same gate helps, reusing the exact completed control rather than spending GPU time repeating it. `causal-pixrestore-strong-command.ps1` records that arm.

The gate itself favors pixel accuracy, while the teacher's advantage is perceptual. A separate `--teacher-policy unfiltered` ablation is available to test whether the gate discards useful guidance. It retains all original reference losses. Such an experimental arm still needs controlled quality and detail evaluation; omitting a training gate is not a shipping promotion.

## Core ML conversion and resolution coverage

The weight-1 checkpoint converts successfully with persistent state at all four tested input sizes. Thirty-sample synchronous Python/Core ML mean/p95 times on M4 Pro are 2.842/3.042 ms at 640×360, 5.006/5.923 at 960×540, 8.670/9.584 at 1280×720, and 13.847/14.470 at 1920×1080; every output dimension is doubled. The six-frame independently evolved Torch/Core ML checks include history resets and stay below 0.725 RGB levels. `causal-pixrestore-native.json` records samples, checkpoint identity and converter checks.

These are graph timings, not browser cadence or full app latency. The separate native Swift state-representation comparison establishes the Swift benefit; the new graph profiles establish conversion across resolutions. Useful reconstruction quality remains the blocker to enabling this route.

## Stronger and unfiltered supervision: completed

The weight-10 guarded run completed 4,000 steps in 3.65 minutes. LPIPS / DISTS / fine correlation are **0.338836 / 0.127235 / 0.512988**: essentially unchanged LPIPS and 0.42% better DISTS than the matched control. The unfiltered weight-3 run completed in 3.64 minutes and scores **0.341428 / 0.125673 / 0.512317**. It improves DISTS 1.64%, but worsens LPIPS 0.76% and slightly reduces fine correlation. Reports are `causal-pixrestore-strong-evaluation.json` and `causal-pixrestore-unfiltered-evaluation.json`. Neither clears the joint quality gate; no weights are promoted.

## Shipping comparison at a common presentation size

`causal-shipping-presentation-evaluation.json` compares the same 640×360 inputs and 1280×720 references. Shipping produces its genuine 4× output, clamped/rounded to RGB8, then explicitly resized with PIL bicubic to 2×. The evaluator rejects incorrect native scales before that declared adapter. This is a weight-only comparator; native postprocessing, NV12 conversion and browser scaling are not simulated.

| Variant | LPIPS ↓ | DISTS ↓ | Fine correlation ↑ |
|---|---:|---:|---:|
| Shipping 4× presented at 2× | 0.302483 | 0.128347 | 0.564311 |
| Guarded teacher weight 10 student | 0.338836 | 0.127235 | 0.512988 |
| Tiled PixRestore teacher, separate identical spatial frames | 0.276422 | 0.113650 | 0.525790 |

Shipping remains much stronger than the students on LPIPS and detail. PixRestore improves perceptual distances over shipping but loses fine correlation. A stronger training target must address that tradeoff; simply making another small student fine-tune is not supported by these results. All screens reuse three development identities and are not independent release evidence.
