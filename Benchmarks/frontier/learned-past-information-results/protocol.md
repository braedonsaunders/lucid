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
