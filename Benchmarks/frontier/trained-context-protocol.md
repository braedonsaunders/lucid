# Trained-controller context diagnostic

Specified before inspecting diagnostic results, September 5, 2026. The previous context probe tested the frozen convolutional anchor only. The trained dynamic controller uses global mean and absolute-mean features, so crop conditioning can differ even where convolution padding has no influence. This is a concrete untested mechanism, not a conclusion about the failed quality endpoints.

Use the same first sorted training sequence and first frame for each of 26 training identities, verified against the immutable diversity-bank manifest and each original NPZ hash. Extract those first frames losslessly for a local MPS diagnostic while the RTX 4080 scores the independent native experiment. No validation or evaluation examples are used. Retain both fixed 8,000-step strict-deterministic unconstrained/constrained endpoints.

For each model, compare full-patch inference cropped to the even-aligned center96 region against ordinary crop inference. Then rerun the identical crop with its controller coefficients frozen to those computed from the full context. Ordinary-crop versus fixed-coefficient-crop isolates conditioning; full-context versus fixed-coefficient-crop measures residual padding/context effects. Report errors at output margins 8/32/64/80 and all coefficient differences. These MAEs are not additive causal percentages. No weights, loss coefficients, checkpoint endpoints or evaluation gates change.

Only after inspecting this training-only mechanism probe should a spatially conditioned alternative be considered. This diagnostic itself authorizes neither new training nor promotion and makes no novelty or flicker claim.

## Completed mechanism result

Both fixed endpoints complete on MPS FP32 with identical training-only samples. At output margin64, the unconstrained model changes by 0.3675 RGB levels on average between full and cropped contexts; changing only its pooled coefficients reproduces 0.3675 levels, while fixed-coefficient padding contributes about 0.000003 levels. The constrained endpoint shows 0.1050 / 0.1050 / 0.000003 levels respectively. The effect persists at margin80. Thus global conditioning produces a real interior crop-context dependence not covered by the earlier frozen-backbone diagnostic.

This does not establish the dominant cause of failed perceptual/detail gates, nor a quality gain from any proposed fix. It supports an isolated local-conditioning prototype whose crop locality and native cost must be checked before spending another training run. All endpoint hashes, sample identities, per-source/coefficient differences and scripts are retained. No flicker objective is introduced.
