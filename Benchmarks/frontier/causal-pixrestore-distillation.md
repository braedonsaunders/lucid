# Reference-checked PixRestore distillation pilot

The August 2026 PixRestore-S teacher, evaluated at its trained 512-pixel size, improves development LPIPS/DISTS substantially. The fast causal student has not matched that quality; doubling its width gave negligible improvement over a matched control. This experiment tests whether teacher outputs improve the existing fast student without adding inference computation.

The teacher uses the exact pinned upstream code, configuration, DINO encoder and weights from the measured baseline. `cache_pixrestore_teacher.py` verifies these identities before loading. It processes only training sequences; the cache records every output hash and the source-bank manifest hash. Validation and evaluation clips are not teacher-training targets.

The new context bank retains the same 21 source identities and family split as the previous bank. It has 160 training / eight validation sequences, each with 16 aligned frames and 256-pixel LR / 512-pixel HR patches, encoded at full-frame size before cropping. This supplies the teacher's trained context size. Larger patches do **not** add independent scene diversity: there are still 42 source windows with four correlated crops each. The original training and shipping assets remain intact.

Two 4,000-step runs start from the same 32-channel checkpoint, seed 20260913, learning rate 0.0001, batch four, crop 96, curriculum and DINO weight 0.03. Both load the same cache and apply exactly the same crop, flip, transpose and frame selection. The candidate adds teacher weight one; the control uses zero. This is a pilot, not a replacement claim.

For a selected frame, the teacher's and fixed Lanczos floor's local mean absolute RGB errors are measured against the known training reference, averaged over a 9×9 neighborhood. Confidence is `clamp((floor_error - teacher_error) / (floor_error + 0.001), 0, 1)`. The added objective is mean confidence-weighted absolute student/teacher error. Confidence and teacher pixels are detached. Regions where the teacher is less accurate than interpolation contribute nothing; the original reference objectives remain active everywhere. This guard is not a guarantee of perceptual accuracy.

The teacher is cached offline and discarded from student inference. The learned graph, persistent Core ML state, parameter count and deployment interface are unchanged. This does not establish that the compressed student preserves the teacher's quality; that is the experiment's measured question.

Eighteen experiment tests pass, including aligned cache augmentation, rejection of wrong-bank or validation targets, rejection of changed cache bytes, and gradient checks for useful versus harmful teacher regions. The remote command is `causal-pixrestore-distillation-command.ps1`; launch/source identities are in `causal-pixrestore-distillation-launch.json`. Completion, matched development scores and independent evaluation remain pending.
