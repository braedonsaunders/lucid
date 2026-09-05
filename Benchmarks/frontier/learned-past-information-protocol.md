# Learned past-information diagnostic — September 5, 2026

Declared before execution. The raw decoded-pixel history probe failed its fixed
mechanism gate. [DGAF-VSR, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Rethinking_Diffusion_Model-Based_Video_Super-Resolution_Leveraging_Dense_Guidance_from_Aligned_CVPR_2026_paper.html)
motivates testing learned information instead of only interpolated input pixels.
This experiment uses the frozen shipping model's subpixel RGB predictions, not
DGAF's hidden features or diffusion pipeline; it is not a reproduction or novelty
claim.

Run the unchanged checkpoint in CUDA FP32 with no learned-weight updates. Regroup
its 4x RGB prediction into twelve channels at 2x source resolution using
PixelUnshuffle(2). No output phase or channel is selected from quality results.
Fit a small fixed linear residual readout with the previous spatial pixel features
plus these current-frame prediction channels. Compare current-only, unaligned
past and motion-aligned past arms. Past channels are residuals against the current
channels, using t-2 and t-1 only. Flow, bidirectional consistency and photometric
visibility use decoded LR inputs only; invalid history becomes current prediction.
Use the same common interior for all metrics, never a visibility-selected region.

Keep the previous 26 training identities, 13/13 hash-ordered fit/check split, first
sequence per identity, frames 4/8/12, stride-eight fitting, 16-pixel metric border,
trace-scaled ridge 0.001, RGB8 MSE and fine correlation. Freeze before execution:
aligned must improve MSE >=1%, fine correlation >=0.002, and both metrics on >=7/13
sources against shipping, current-only and unaligned controls. The original
raw-pixel probe remains a separate preserved result. Do not tune channels, ridge,
flow, endpoints or gates after observing scores.

These identities were previously used for neural training, and collection families
overlap across fit/check. This is a mechanism screen, not independent quality
validation, LPIPS/DISTS promotion, a trained replacement, native performance or
flicker work. A passing result only justifies a subsequent controlled experiment;
a failing result does not rule out other learned representations. No shipping
model or resolution coverage changes follow automatically.

## Fixed endpoint result

The passwordless durable job runs through `lucid-gpu-vpn` on RTX 4080, preserving
existing jobs and the shared environment. All seven alignment/readout tests pass.
The diagnostic itself takes 21.75 seconds; the complete task takes about 27 seconds.
All 156 check rows are present, all downloaded hashes match, and all 39 shipping
control rows reproduce the earlier probe exactly. Independent local recomputation
matches every comparison within 1e-12.

| Comparator | Aligned MSE improvement | Fine-correlation change | Sources improving both |
|---|---:|---:|---:|
| Shipping | 1.0347% | +0.010514 | 10/13 |
| Current-only learned prediction | 0.9755% | +0.004185 | 11/13 |
| Unaligned learned history | 1.2314% | +0.005171 | 11/13 |

**The mechanism gate fails:** MSE improvement over the current-only arm is below
1%. The additional fine correlation is measurable, but the frozen joint rule does
not pass. No full training, coefficient selection or shipping replacement follows.
The raw reports, job, source manifest and independent verification are in
`learned-past-information-results/`. Diagnostic weights remain in
`.build/learned-past-r1-results/result/diagnostic-linear-weights.pth` and the durable
remote result directory; their hash is retained in the report. The GPU is idle
after completion. This establishes neither independent perceptual quality nor
native temporal-feature performance.
