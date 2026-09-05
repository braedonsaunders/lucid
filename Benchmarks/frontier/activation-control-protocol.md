# Training-only activation-control diagnostic

Frozen September 5 before measurement. The just-completed subspace experiment preserves selected linear weight responses but fails final image detail guards. Before another training run, inspect whether input-dependent feature control has a plausible intervention surface in the existing fast graph.

The [September 3 SPARK paper](https://arxiv.org/html/2609.03813v1) studies dominant activation channels in diffusion-transformer SR models. It motivates a diagnostic; it does not establish that a small convolutional model has the same concentration or that changing its channels improves restoration. No author code/checkpoint was found in the inspected paper or targeted search. This is not a SPARK implementation.

Use frozen shipping weights `fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65` with the already-defined area-folded 2× head and fused convolutions. Use bank `b11ccdaba5ed388ec615da4270058b6441369fb6325a6e5e6680472e98d6d2f0`, selecting the first frame of the first sorted sequence from each **training** source, verifying bytes and source identity. No validation or holdout images enter selection or measurement.

At each of the six block outputs, collect source-balanced per-channel mean absolute activation and squared energy. Rank by absolute magnitude; report the fraction of energy in the leading 1, 4 and 8 of 32 channels. A second pass compares an unchanged graph with a fixed 0.95 gain on four top-ranked, bottom-ranked or fixed-seed random channels at all six main block outputs. Preserve the other returned block tensors. Use only this gain and channel count; no selection sweep or optimizing coefficients from these results.

Record output change, reference RGB8 MSE and reference fine correlation. These measure intervention sensitivity on training sources, not generalization, perceptual preference or a quality gate. No weights change, controller is trained or new runtime is installed. The unmodified repeat must be pixel-identical. If the premise lacks support, record that rather than assigning the paper's behavior to Lucid by analogy.
