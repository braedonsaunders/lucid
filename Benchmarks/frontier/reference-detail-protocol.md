# Reference-grounded detail supervision: fixed experiment

The expanded-data 2× student improved full-frame development LPIPS by 6.75% and DISTS by 10.17% at the already-fixed 80% mixture, but lost fine-detail correlation on FourPeople and Johnny. Its current signed-gradient and FFT objectives both imitate the shipping/PixRestore target mixture. The next controlled test changes **only those two targets to the HR reference**, retaining their 0.2 and 0.05 weights. Pixel regression still uses the teacher mixture plus the existing 0.1 reference term. The pinned August 2026 PixRestore DINO adversary and all inference operations remain unchanged. This is a supervision ablation, not a novelty claim.

Use the exact composed 29-source bank, shipping and teacher caches, folded shipping initialization, seed 20260914, 8,000 steps, batch four, crop 96, learning rate 0.00002 and cosine schedule from the completed expanded-data coupled control. Both arms use the coupled 2× graph. Matching two-step smoke runs must reproduce the discriminator initialization and first-batch hashes; the default objective must remain bitwise equivalent in a unit test. Reuse the completed 8,000-step control after these checks.

Evaluate the raw final checkpoint and the predetermined 80% trained / 20% folded initialization mixture on the same 48 full-frame development pairs and the 96 declared bank-validation pairs. Do not search mixture coefficients. Each comparison uses shipping as baseline and the frozen 3% LPIPS/DISTS aggregate minimum, per-source perceptual guards and 0.01 maximum fine-detail correlation loss. Report both failures and passes. These are development results; a winner still needs controlled native conversion, postprocessing and fresh source-disjoint release evaluation. The existing failed native holdout remains regression evidence.

No shipping weights are overwritten. The RTX 4080 supervisor checks for existing Python jobs and uses a fresh output directory with an exclusive owner lock. All source/data/checkpoint hashes and exact commands are recorded before training. No temporal or flicker loss is introduced.

## Completed result

Training completed all 8,000 steps in **5.84245 minutes** on the RTX 4080. Both smoke arms matched the initial discriminator hash and the first source/reference/teacher batch hash. The full-frame and bank-validation evaluation processes exited successfully.

| Variant and evaluation | LPIPS improvement | DISTS improvement | Detail / promotion result |
|---|---:|---:|---|
| Raw, 48 full-frame pairs | 9.57% | 14.80% | Fails fine correlation on all three sources |
| Fixed 80%, 48 full-frame pairs | 6.94% | 10.11% | Fails FourPeople and Johnny fine correlation |
| Raw, 96 validation patches | 9.41% | 9.33% | Fails all three detail guards and Sintel LPIPS |
| Fixed 80%, 96 validation patches | 10.95% | 6.76% | Passes these three patch sources only |

The fixed 80% full-frame fine-correlation deltas are CrowdRun −0.00869, FourPeople −0.01321 and Johnny −0.01532, against a maximum permitted drop of 0.01. These results are close to the previous mixture-detail control and do not remove its quality tradeoff. No model is promoted.

The weighted adversarial/head reconstruction gradient ratio was **0.149 at step 1, 9.112 at step 200 and 3.512 at step 1,000**. These are measured gradients at the output head, not a causal proof about the entire model. They motivate testing a bounded adversarial gradient contribution next: the intended initially modest adversarial signal did not remain modest. No further hyperparameter or coefficient was selected from this evaluation.

Sixteen focused Python tests passed across trace analysis, validation export, objective compatibility/gradients, folding, checkpoint interpolation and quality gates. The adjacent `reference-detail-*` receipts preserve both smoke runs, the executed evaluation command, training metadata/log, completed evaluation reports and gate outputs. Each development gate checks sample identities against the corresponding prior frozen evaluation and verifies matching manifest and shipping-checkpoint hashes.
