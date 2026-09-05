# Frozen reconstruction with a learned detail branch

The broader-data adversarial model still improves perception by sacrificing detail. This experiment separates trainable refinement from a fixed reconstruction base. It is an engineering adaptation inspired by the recently reviewed PixelIR direction, not its flow teacher, transformer student, trained model or a novelty claim.

`AnchoredDetail` keeps the folded shipping 2× reconstruction immutable. Source 2×2 phases and reconstructed 4×4 phases are concatenated at half source resolution. Four narrow convolution blocks condition a learned RGB residual on both. The additional branch has **21,584 parameters**, zero-initialized output, and no frame history, stochastic input or temporal filter. Freezing the base does not guarantee the added detail is correct; it prevents learned refinement from changing the original reconstruction weights.

Three tests check exact zero-residual identity including borders, an effective learning path through the branch with bitwise unchanged anchor state, and nonzero tracing/checkpoint round trips through the shared evaluation loader. At each training checkpoint the trainer checks the complete frozen anchor against its first-forward SHA; unexpected changes abort the experiment.

## Native feasibility

A deliberately nonzero random branch prevents the compiler from erasing the added work. Forty synchronous predictions after ten warmups alternate between each graph on M4 Pro CPU+GPU. The proper shipping input is pinned to SHA `fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65`. All three graphs remain within one RGB level of their own FP32 reference. This is conversion/cost evidence, not learned image quality or browser cadence.

| Input | Shipping 4× mean/p95 ms | Folded 2× mean/p95 ms | Untrained anchored 2× mean/p95 ms |
|---|---:|---:|---:|
| 640×360 | 9.62 / 10.10 | 5.93 / 6.16 | 7.26 / 7.56 |
| 1280×720 | 36.41 / 37.34 | 21.35 / 21.78 | 27.34 / 27.86 |

The branch costs 1.34 ms at 360p and 5.99 ms at 720p. Shipping outputs four times the input dimensions; the other graphs output twice the dimensions. These graph times exclude downsampling, preprocessing, packet transport and presentation. A first command accidentally selected historical `span_ch32u` weights; that run is excluded. The accepted run uses `span_ch32utc` and an explicit expected-SHA guard. `anchored-detail-native-probe.json` and `anchored-detail-native-profiler-snapshot.py` preserve the actual accepted result and executed profiler source.

## Training comparison fixed before execution

Use the composed 29-source bank, exact existing teacher/shipping caches, shipping initialization, 8,000 steps, seed 20260914, batch four, crop 96, learning-rate schedule and losses from the completed broader-data adversarial control. Train only the conditional branch. The primary result is the **raw final anchored checkpoint versus raw final coupled control and shipping** on the same frozen development frames. The previous fixed 80% coupled candidate is additional context; no anchored gain sweep is planned from these results.

Branch initialization occurs inside a saved/restored Torch RNG scope, preserving the subsequent discriminator initialization stream. Two-step coupled and anchored smoke runs must record matching initial discriminator hashes and first source/reference/target batch hashes before the main anchored run starts. They also exercise CUDA, checkpoint loading and immutable-anchor checks. The completed coupled 8,000-step run is reused; changing architecture does not justify wasting another identical full control run.

A quality win still needs source-disjoint validation, proper native postprocessing and fresh promotion footage. The failed eight-source holdout remains historical regression evidence. No shipping weights or native tuning defaults change for this experiment.
