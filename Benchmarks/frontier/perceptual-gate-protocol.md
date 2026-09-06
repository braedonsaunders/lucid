# Perceptual promotion gate — 2026-09-05

Every reconstruction candidate trained this month improved source-balanced LPIPS and DISTS, several by 5–18%, and every one was rejected by the frozen development gate's requirement that no source lose more than 0.01 fine-band correlation against shipping. The measured NVIDIA VFX SDK output, the competitive target, improves LPIPS/DISTS by 23–25% / 18–22% on the identical 48 pairs and fails the same guard on CrowdRun and Johnny. A guard that rejects the target is not measuring the goal.

Fine correlation is scale-invariant. It measures whether fine structure is real and cannot reward synthesized detail by construction, so a gate anchored to shipping's correlation admits only models that synthesize nothing. `gate_perceptual.py` keeps the perceptual minimums (≥3% source-balanced LPIPS and DISTS gain, ≤2% per-source perceptual regression) and replaces the shipping-relative correlation guard with two guards a good synthesizer passes and a bad one fails:

- **Hallucination floor.** Per-source fine correlation must not fall below the Lanczos anchor, which synthesizes nothing. Below the anchor, added fine energy is invented rather than recovered. The earlier temporal-weight sweep found the same boundary empirically: at weight 10 the model fell below the anchor and both perceptual metrics worsened.
- **Embellishment cap.** Per-source fine-band energy must not exceed the reference by more than 10%.

Applied to the retained reports without changing any pixels:

| Candidate | Set | LPIPS | DISTS | Frozen gate | Perceptual gate |
|---|---|---:|---:|---|---|
| NVIDIA VFX HIGH | 48 dev | +23.14% | +17.86% | fails (2 sources) | **passes** |
| NVIDIA VFX ULTRA | 48 dev | +25.00% | +22.22% | fails (2 sources) | **passes** |
| NVIDIA VFX BICUBIC | 48 dev | −11.23% | −3.33% | fails | fails (10 reasons) |
| paired critic raw | 48 dev | +10.05% | +17.98% | fails (2 sources) | **passes** |
| paired critic 80% | 48 dev | +7.85% | +11.81% | fails (Johnny) | **passes** |
| paired critic raw | 96 bank | +8.98% | +10.89% | fails | fails (Sintel LPIPS −5.7%, below anchor) |
| paired critic 80% | 96 bank | +11.10% | +7.60% | **passes** | **passes** |

The perceptual gate separates the SDK's AI modes from its bicubic mode by a wide margin and rejects the raw checkpoint on the animation source where it genuinely regresses. It is a development screen only: the native delivery holdout, temporal stability and fresh-source evidence remain required before any weights ship.

Receipts: `paired-critic-*-perceptual-gate.json`, `nvidia-vfx-rgb48-perceptual-gate.json`; tests in `test_gate_perceptual.py`.

## Teacher finding

`teacher-complementarity.json` (the cached tiled PixRestore teacher actually used for training) scores LPIPS 0.2764 / DISTS 0.1136 on the 48 pairs: +8.6% / +11.5% over shipping, fine correlation 0.526, above the Lanczos anchor. The fixed 50/50 shipping/teacher mixture that forms every student's pixel-regression target scores only 0.2937 / 0.1198 (+2.9% / +6.6%). The paired-critic raw student reaches 0.2720 / 0.1053, past both the mixture and the teacher itself: the gain comes from the critic and reference terms, and the mixture target now pulls the student back toward shipping. (An earlier single-pass PixRestore configuration in `.build/pixrestore-spatial-evaluation.json` scored 0.290 / 0.1315; it is not the cached teacher.) The next controlled run removes the mixture and regresses to the HR reference (`--intended reference`).
