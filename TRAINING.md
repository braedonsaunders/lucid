# The upscaler: what was measured, and what was tried and failed

This is the record of how Lucid's model got to where it is. It exists because
most of what follows was learned expensively, is not visible in the code, and
would otherwise have to be learned again. It replaces `RESUME.md`, which was an
operational note for a training run that has since finished.

Every number here was measured on one machine — an Apple M4 Pro, macOS 26 — and
nothing has been measured on any other chip. See
`Lucid/Resources/DeviceCapabilities.json`.


## What ships

`SPAN_x4_ch32u`, ~1.03M parameters, six fixed input sizes, Core ML on the
Neural Engine. Trained here from scratch. Not a fine-tune of a published
checkpoint — there is no published checkpoint for this architecture.

Held-out validation at the end of the 60000-step run: **32.49 dB PSNR,
0.8930 fine**.


## The honest comparison

A model's own validation set is held-out *patches of its own corpus*, which
makes two models trained on different corpora incomparable — part of the gap
between their curves is "different test set". `Tools/eval_checkpoint.py` exists
to settle that: it scores any number of checkpoints on the same disjoint clip,
next to a plain Lanczos upscale.

Scored on 120 pairs of `crowd_run` at 3840×2160, footage in neither training
corpus, fine-band correlation with the truth:

| | fine | mid | coarse | PSNR |
|---|---|---|---|---|
| Lanczos anchor | 0.2288 | 0.5317 | 0.8145 | 25.17 |
| Same architecture, animation-only corpus | 0.2521 | 0.5517 | 0.8165 | 25.43 |
| **ch32u — what ships** | **0.2720** | **0.5697** | **0.8210** | **25.58** |

It wins every tier and every band — **+19% fine-band over the anchor**.

**Do not write this up as "the corpus fixed everything."** Decomposed at matched
steps (22k against the control's 24k), the corpus diversity is worth about
**+0.009** and the extra 38000 steps about **+0.011**. The corpus was roughly
half the gain. Both mattered; neither was the whole story.

Against the models it replaced, on the 1080p bench references at 480×270:

| | fine | mid | coarse | ms |
|---|---|---|---|---|
| ch28 — previously shipping | 0.2214 | 0.6406 | 0.9467 | 14.7 |
| ch48 | 0.2278 | 0.6424 | 0.9466 | 21.8 |
| **ch32u** | **0.2242** | **0.6528** | **0.9475** | **7.92** |

Better than ch28 on all three bands at 1.86× the speed, and it beats the much
larger ch48 on mid and coarse at a third of the cost.

**That table flatters ch32u and should carry this caveat wherever it is
quoted:** the bench references are Big Buck Bunny, which *is* in ch32u's
training corpus. The `crowd_run` numbers above are the trustworthy ones,
because that footage is in neither corpus.


## Why fine-band correlation, and not PSNR

Fidelity metrics mislead on this problem. Measured against ground truth, plain
bicubic correlates *better* with the truth at every spatial band than either
Apple's scaler or this pipeline — perceptual super-resolution synthesises
plausible detail, it does not recover real detail.

Band **energy** is worse than useless: it cannot tell recovered detail from
invented detail, and it misled this project twice. Correlation in the finest
band can, which is why it is the number everything is judged by.

One trap worth knowing: correlation is **scale-invariant**. A converted variant
that came back nearly black — output mean 17.7 where torch said 92.0 — still
scored above the Lanczos anchor. Always check the output level, never just the
metric. (`Tools/convert_span.py` documents the fp16 cause.)


## The architecture change, and why it was necessary

Every published efficient-SR model runs its trunk at the input resolution.
Measured on the Neural Engine, that trunk is **activation-bandwidth bound, not
MAC bound**:

* Latency scales with channels, not channels squared — ch16 0.63×, ch20 0.75×,
  ch24 0.90× against ch28.
* int8 **weight** quantisation moved 30.6 ms to 29.3 ms. Nothing. Weights are
  not the traffic.

So the lever is pixels, not channels. A `PixelUnshuffle(2)` before the trunk
runs the 18 trunk convolutions at quarter area and lets the head do ×8 instead
of ×4. At 640×360 that is **27.9 ms → 12.6 ms at slightly more capacity**
(1.06M parameters against 1.03M). Nobody ships weights for that shape, which is
why training from scratch was the only route.


## The ladder

Model only, ANE, M4 Pro, against the ch28 ladder it replaced:

| input | ch28 | **ch32u** |
|---|---|---|
| 256×144 | 4.5 ms | **2.38 ms** |
| 320×180 | 7.0 ms | **3.54 ms** |
| 432×240 | 13.6 ms | **6.40 ms** |
| 480×270 | 14.7 ms | **7.92 ms** |
| 640×360 | 26.7 ms | **12.53 ms** |
| 864×480 | 48.2 ms | **23.50 ms** |

Roughly half the time and better on every band. 864×480 dropping under budget is
what opened 480p, and 480p is what reaches the window Edge uses.

Width must be a multiple of 16. The Neural Engine tiles along width and an
unaligned width pays for an entire extra pass — 426×240 costs 23.0 ms against
432×240's 13.6 ms for the same picture. Height alignment buys nothing; 480×272
measured *slower* than 480×270 by exactly the two extra rows.


## The corpus, and the four rebuilds it took

`Tools/make_degradation.py` builds pairs by pushing real crops through real
codecs — libx264 and libvpx-vp9 at 90 kbps to 1.5 Mbps, GOP 30 to 250 — rather
than by bicubic downsampling. Stock checkpoints are trained on bicubic, and that
mismatch is the single biggest lever there is: it is not what a 240 kbps VP9
stream does to a picture.

**The corpus was rebuilt four times, and every fault was distributional.** Every
one passed every structural check — counts, file sizes and HR == 4×LR were
correct every single time:

1. Pairs generated at 2× when 4× was needed.
2. HR and LR decoded by separate seeks, landing on **different frames** — 57% of
   pairs were the same shot at a different moment. A model trained on those
   learns to invent motion. The only symptom was a validation PSNR that looked
   merely disappointing. Cause: `-ss` before `-i` is a keyframe seek.
3. 100% animation. A model that has never seen skin, film grain, sensor noise or
   a real lens.
4. Then 95% live action — the same mistake with the sign flipped, caused by
   choosing uniformly over files when one class had 19 and the other had 2.

Two defences came out of this and both should stay. `pair_correlation()` in
`Tools/train_span.py` downscales each HR and correlates it against its LR,
rejecting anything below 0.85 and hard-exiting if more than a fifth fail — a
real pair survives a round trip. And `report_composition()` prints content,
tier, codec, bitrate and GOP distribution *before* a minute is spent training.

If you write a contributing guide, the lesson belongs in it: **shape is what
goes wrong, so shape is what gets printed.**


## Dead ends — do not repeat these

* **Pre-deblocking the source before the model does nothing.** 0.228 → 0.226.
  The network removes compression damage itself; cleaning the input first only
  softens what it is about to reconstruct from.
* **Apple's `VTLowLatencySuperResolutionScaler` measures below a Lanczos
  anchor** (0.174 against 0.180), inventing a mottled foliage texture that is
  not in the source. Sharpening on top of it made things worse still (0.152).
  It is not in the shipping path and there is no fallback to it.
* **Transcoding the Chimera clips to a short GOP did not fix corpus generation
  speed.** The cost was frame count on 60fps sources, not seeking. Sizing the
  decode window by frame rate is what fixed it.
* **Blender's mirror only has Tears of Steel at 720p** — too small to supply
  2560×1440 HR crops.
* **archive.org's CC-licensed 4K is overwhelmingly YouTube rips**, already
  compressed, and useless as ground truth.
* **Running Core ML jobs on the ANE and GPU concurrently does not add their
  throughput.** Shared memory pressure took aggregate throughput *down*, 15.5
  fps to 11.8.


## Open questions

* **int8 activations (W8A8) — untested, and the best unclaimed lead.** int8
  *weight* quantisation was ruled out because it bought nothing, and that result
  is exactly what points at activations: if the trunk is activation-bandwidth
  bound, then quantising the activations is the thing that should move it.
  Conversion-side only, no retraining. Potentially large.
* **Chroma siting measures exactly 0.0000 on every band**, even after the bench
  was fixed to honour the setting. Either it genuinely does nothing on this
  content or there is a gap in the harness. It is on because it is free and the
  reasoning is sound, not because it is proven. Do not claim it either way.
* **Temporal input to the model.** Independently the top-ranked proposal from
  four external model reviews of this repository. A checkpoint exists
  (`span_ch32ut`, carrying an extra `frames` key) but is not converted and is
  not what the app loads.
* **Everything is one machine.** No number here has been reproduced on any chip
  other than an M4 Pro.


## Things that were proposed and deliberately not done

Several reviews recommended deleting the detail stack along with the Apple
scaler path. That was refused, and the refusal was right: the later ablation
showed the temporal stage alone earns **0.008 fine**, which is more than half of
what the entire continuous-parameter tuning sweep bought. Measure a stage before
removing it.

The Apple scaler path itself is kept in the code for one reason — the offline
comparison shooter needs it to produce the stills that justify not using it.
`EngineKind.apple` and `.lucid` exist for that and the app never runs them.
