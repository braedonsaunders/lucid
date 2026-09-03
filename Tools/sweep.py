#!/usr/bin/env python3
"""Finds the value each continuous control should have, by measuring.

`ablate.py` answers "is this stage worth running at all". This answers "and at
what setting", which is a different question and the one the tuning panel
actually asks.

Two kinds of control, and only one has a right answer:

  Fidelity   sharpen, detail, deblock, lobe, micro. These try to reconstruct
             what was lost, so ground truth decides. Swept, peak reported.

  Grade      black point, contrast, saturation. These deliberately depart from
             the source; the reference has no grade, so every fidelity metric
             will prefer neutral. Sweeping them measures what the look costs,
             not what it should be. Reported as a cost curve, not a
             recommendation.

Coordinate descent, not a grid: the controls interact (sharpen and lobe most
obviously), so each pass sweeps one control with the others held at the best
values found so far, and a second pass catches what the first pass's ordering
missed.

  .venv-convert/bin/python Tools/sweep.py --frames 8
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablate import run, ROOT  # noqa: E402

FIDELITY = [
    ("sharpness", "Sharpen", [0.0, 0.15, 0.3, 0.5, 0.7]),
    ("fine",      "Detail",  [0.0, 0.15, 0.3, 0.5]),
    # The panel's "Deblock" is sourceDeblock (pre-scale, 0-0.08). `deblock` is a
    # separate post-upscale control that is not on the panel but is active.
    ("sourceDeblock", "Deblock (panel)", [0.0, 0.02, 0.04, 0.08]),
    ("deblock",   "Deblock (post)", [0.0, 0.15, 0.35, 0.6]),
    ("lobeScale", "Lobe",    [0.3, 0.5, 0.7, 1.0]),
    ("micro",     "Micro",   [0.0, 0.1, 0.25, 0.5]),
]
GRADE = [
    ("blackPoint", "Black",    [0.0, 0.02, 0.04]),
    ("contrast",   "Contrast", [0.0, 0.1, 0.2]),
    ("saturation", "Colour",   [1.0, 1.05, 1.12]),
]


def sweep(label, key, values, base, clip, reference, frames, learned):
    """Sweeps one control, returning the value with the highest fine-band
    correlation. Ties break toward the smaller value: two settings that measure
    the same are not equally good, and the one that does less is the one that
    can go wrong in fewer ways."""
    rows = []
    for value in values:
        tuning = dict(base); tuning[key] = value
        result, (ms, _) = run(tuning, clip, reference, frames, learned)
        if result is None:
            print(f"    {value:>6}  bench failed"); continue
        rows.append((value, result, ms))
        print(f"    {value:>6}  fine {result['fine']:.4f}  mid {result['mid']:.4f}  "
              f"coarse {result['coarse']:.4f}  PSNR {result['psnr']:.2f}  "
              f"{ms or 0:.2f} ms", flush=True)
    if not rows:
        return base.get(key), None
    best = max(rows, key=lambda r: (round(r[1]["fine"], 4), -r[0]))
    return best[0], best[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="bbb-360p-350k.mp4")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--tuning", default=os.path.join(ROOT, "Tools/tuning.json"))
    parser.add_argument("--grade", action="store_true",
                        help="also sweep the grade controls, as a cost curve")
    args = parser.parse_args()

    clip = os.path.join(ROOT, "TestSite", args.clip)
    reference = os.path.join(ROOT, "TestSite", "bbb-1080p60-1700k.mp4")
    for path in (clip, reference):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")

    with open(args.tuning) as fh:
        base = json.load(fh)
    start, (start_ms, upscale_ms) = run(base, clip, reference, args.frames, True)
    if start is None:
        raise SystemExit("baseline bench failed")
    print(f"clip {args.clip}, {args.frames} frames, ch32u on the Neural Engine")
    print(f"starting point: fine {start['fine']:.4f}  mid {start['mid']:.4f}  "
          f"PSNR {start['psnr']:.2f}  (upscaler {upscale_ms or 0:.2f} ms, "
          f"detail {start_ms or 0:.2f} ms)\n")

    for pass_index in range(args.passes):
        print(f"── pass {pass_index + 1} ──")
        for key, label, values in FIDELITY:
            print(f"  {label} ({key}), currently {base.get(key)}")
            best, result = sweep(label, key, values, base, clip, reference,
                                 args.frames, True)
            moved = "" if best == base.get(key) else "   <-- changed"
            print(f"    best: {best}{moved}\n")
            base[key] = best

    final, (final_ms, _) = run(base, clip, reference, args.frames, True)
    print("── result ──")
    for key, label, _ in FIDELITY:
        print(f"  {label:9s} {base[key]}")
    if final:
        print(f"\n  fine {start['fine']:.4f} -> {final['fine']:.4f}   "
              f"mid {start['mid']:.4f} -> {final['mid']:.4f}   "
              f"PSNR {start['psnr']:.2f} -> {final['psnr']:.2f}   "
              f"detail {start_ms or 0:.2f} -> {final_ms or 0:.2f} ms")

    if args.grade:
        print("\n── grade: what the look costs, not what it should be ──")
        for key, label, values in GRADE:
            print(f"  {label} ({key}), currently {base.get(key)}")
            sweep(label, key, values, base, clip, reference, args.frames, True)
            print()

    out = os.path.join(ROOT, ".build/sweep-result.json")
    with open(out, "w") as fh:
        json.dump(base, fh, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
