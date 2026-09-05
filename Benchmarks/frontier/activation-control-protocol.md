# Training-only activation-control diagnostic

Frozen September 5 before measurement. The just-completed subspace experiment preserves selected linear weight responses but fails final image detail guards. Before another training run, inspect whether input-dependent feature control has a plausible intervention surface in the existing fast graph.

The [September 3 SPARK paper](https://arxiv.org/html/2609.03813v1) studies dominant activation channels in diffusion-transformer SR models. It motivates a diagnostic; it does not establish that a small convolutional model has the same concentration or that changing its channels improves restoration. No author code/checkpoint was found in the inspected paper or targeted search. This is not a SPARK implementation.

Use frozen shipping weights `fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65` with the already-defined area-folded 2× head and fused convolutions. Use bank `b11ccdaba5ed388ec615da4270058b6441369fb6325a6e5e6680472e98d6d2f0`, selecting the first frame of the first sorted sequence from each **training** source, verifying bytes and source identity. No validation or holdout images enter selection or measurement.

At each of the six block outputs, collect source-balanced per-channel mean absolute activation and squared energy. Rank by absolute magnitude; report the fraction of energy in the leading 1, 4 and 8 of 32 channels. A second pass compares an unchanged graph with a fixed 0.95 gain on four top-ranked, bottom-ranked or fixed-seed random channels at all six main block outputs. Preserve the other returned block tensors. Use only this gain and channel count; no selection sweep or optimizing coefficients from these results.

Record output change, reference RGB8 MSE and reference fine correlation. These measure intervention sensitivity on training sources, not generalization, perceptual preference or a quality gate. No weights change, controller is trained or new runtime is installed. The unmodified repeat must be pixel-identical. If the premise lacks support, record that rather than assigning the paper's behavior to Lucid by analogy.

## Measured outcome

The CUDA diagnostic completed successfully on all 26 training sources, with identical unmodified repeats and no changed weights. Its results were retrieved through the newly verified Tailscale SSH alias. Four magnitude-ranked channels carry 26.3–55.6% of block energy; eight carry 46.9–75.8%. This is concentrated, but the selected four already occupy 12.5% of a 32-channel block, unlike the paper's much sparser DiT intervention.

| Fixed 5% attenuation | Mean absolute output change (0–1) | Mean reference fine-correlation change | Mean reference RGB8 MSE change |
|---|---:|---:|---:|
| Top four | 0.001295 | −0.004686 | +0.03252 |
| Random four | 0.000870 | −0.001044 | +0.04114 |
| Bottom four | 0.000217 | −0.000683 | −0.06879 |

Dominant-channel attenuation changes the output 1.49× more than the fixed random selection and 5.96× more than the bottom selection. All three perturbations lose average reference fine correlation. Therefore this establishes a sensitive intervention surface, not an improvement or a justified fixed enhancement setting. Any learned input-conditioned controller would need a fresh matched training experiment, native cost/correctness checks and unchanged validation gates. No gain or channel-count sweep was performed. See `activation-control-probe.json` for every source/layer and `activation-control-summary.json` for original remote hashes.
