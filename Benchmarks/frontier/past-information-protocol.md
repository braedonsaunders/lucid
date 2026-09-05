# Past-frame reference-information probe

Frozen September 5, 2026, before executing the data probe. The coarse local controller passes native cost and detail guards but gains less than 1.4% LPIPS and 0.7% DISTS in both development screens. Test whether additional decoded observations supply useful reference detail before training another reconstruction network. This is a diagnostic linear-prediction experiment, not a proposed shipping architecture or novelty claim.

The [CVPR2026 dense-guidance paper](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Rethinking_Diffusion_Model-Based_Video_Super-Resolution_Leveraging_Dense_Guidance_from_Aligned_CVPR_2026_paper.html) motivates checking alignment and the information retained during upscaled warping. This probe does not reproduce its feature-domain diffusion method. OpenCV DIS and ridge regression are established measurement tools; they must not be described as September2026 inventions or deployed Mac performance.

Pin the existing 26 training identities and shipping-output cache by manifest SHA256. Select the first lexicographic sequence per training source and frames 4/8/12, using only t−2 and t−1 as history. Sort source IDs by SHA256 of `20260905-past-information:` plus ID, then alternate 13 fit and13 check identities. Neural training has previously seen every identity, and the Netflix collection family spans the two partitions. This is source-separated fitting of a new diagnostic predictor, not an untouched quality holdout. No development-validation or external holdout pixels enter the probe.

Compare frozen cached shipping output, a spatial linear residual predictor, an unaligned-history predictor, and a motion-aligned-history predictor. The latter two have identical 111-feature dimensions. Common features contain local 3×3 current-interpolation residuals and baseline high-frequency features. History adds 3×3 differences between each previous interpolation and the current interpolation, visibility channels and a bias. Only decoded LR images determine DIS flow, forward/backward checks and photometric visibility; no HR-derived correspondence or mask is allowed. Estimate motion at LR, scale it to the 2× grid, and warp bicubic-upsampled past samples. Invalid history is replaced with current interpolation.

Fit RGB reference residuals with a single trace-scaled ridge coefficient 0.001. Accumulate deterministic CUDA matrix products from an 8-pixel sampling grid inside a 16-pixel boundary, then solve in CPU float64. Do not sweep the ridge, select frames, tune coefficients after checking, or fit on check sources. Score every arm on the same complete 16-pixel-trimmed region, never on flow-selected regions. Record RGB8 MSE, the existing fine-band correlation, visibility coverage, source means and every frame row.

For this mechanism alone, require the aligned arm to reduce source-balanced MSE by at least 1%, increase fine correlation by at least 0.002, and improve both on at least 7 of 13 check identities against **each** of shipping, spatial and unaligned controls. These are diagnostic thresholds, not substitutes for the existing perceptual/native/browser promotion gates. A failed probe does not prove history lacks information; it rejects this particular predictor/alignment premise as immediate support for a larger run. No temporal smoothing or flicker loss is introduced.

Before execution, test known backward-flow direction and visibility, identity alignment, feature coordinates, and recovery of a known linear signal. The remote task checks existing Python jobs and retains exclusive GPU ownership for this bounded diagnostic. Preserve all existing jobs, shipping weights, rejected endpoints and prior artifacts.

The three local math/alignment tests pass. The initial remote task stops before data processing because OpenCV is absent. Preserve its import-error log. A fresh r2 directory adds only the verified official OpenCV 4.14.0.94 Windows headless wheel in its private dependency directory, matching local OpenCV 4.14.0. The existing remote NumPy 2.5.2 satisfies the wheel requirement. Source, data, ridge, split and mechanism limits remain unchanged; no package is installed into the shared training venv. All three remote tests then pass and the CUDA diagnostic starts. The wheel's download URL and SHA256 are retained in `past-information-opencv-receipt.json`.

## Completed result

The bounded CUDA diagnostic finishes in 15.31 seconds after all three remote tests pass. All 156 check rows are present: 13 source identities × 3 frames × 4 arms. Downloaded outputs match remote SHA256 receipts; independent local aggregation reproduces all source-balanced metrics and the failed mechanism gate.

| Aligned predictor compared with | RGB MSE reduction | Fine-correlation gain | Sources improving both |
|---|---:|---:|---:|
| Cached shipping | 0.639% | 0.007064 | 9/13 |
| Spatial predictor | 0.592% | 0.000463 | 10/13 |
| Unaligned history | 0.946% | 0.000718 | 7/13 |

Alignment helps this fitted predictor, but its incremental effect over the spatial control is small. Most of the fine-correlation gain over shipping is also achieved by the spatial predictor. The aligned route misses the fixed MSE minimum against all controls and the incremental fine-correlation minimum against both fitted controls. This does not justify launching a larger temporal training run from this mechanism check. It does not establish that nonlinear feature reconstruction, different footage or better correspondence could not help. No perceptual quality, native latency or release promotion claim follows, and no shipping weight changes.
