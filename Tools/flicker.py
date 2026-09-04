#!/usr/bin/env python3
"""Measures how much the picture shimmers between frames.

Every other metric in this project scores one frame against one reference, and
is therefore structurally blind to the axis this measures. That blindness has
already cost us once: the per-frame table said the temporal stage was harmful on
every clip, because a stage that trades a little per-frame sharpness for
frame-to-frame steadiness can only ever look like a loss to an instrument that
sees one frame at a time.

It matters more now. A GAN synthesises plausible texture per frame
independently - nothing tells frame N what frame N-1 invented - so adversarial
training buys detail and pays for it in stability. That is the shimmer a viewer
notices, and until this tool existed nobody could say whether it was there or
how much.

  flicker = mean |output[t] - output[t-1]| over pixels the REFERENCE holds still

The mask is the load-bearing part. Taking it from the reference rather than from
our own output means it cannot be influenced by anything we did: it isolates
regions that genuinely should not be changing, so movement there is our
invention rather than the scene's. Measuring our output against itself would
simply reward a model that changes less, which a blur does perfectly.

Two ways in. `--stems` scores models the app already carries, through the real
pipeline, and so includes everything the detail stages do. `--checkpoints`
scores .pth files directly in torch, before conversion - which is the only way
to ask whether a temporal model is worth the plumbing it would need, since the
app has no multi-frame input path to run it through yet.

  .venv-convert/bin/python Tools/flicker.py --frames 12
  .venv-convert/bin/python Tools/flicker.py --checkpoints Model/weights/*.pth
"""
import argparse, json, os, shutil, subprocess, sys, tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, ".build/release/Build/Products/Release/Lucid.app/Contents/MacOS/Lucid")

# Pixels the reference moves by less than this between frames are treated as
# static. In 0-255 luma: below roughly one level is sensor and codec noise in the
# reference itself, not motion.
STATIC = 1.2


def render(stem, clip, reference, frames, tuning):
    """Runs the bench and returns its output and reference frames, in order."""
    out = tempfile.mkdtemp(prefix="flicker-")
    path = os.path.join(out, "tuning.json")
    with open(path, "w") as fh:
        json.dump(tuning, fh)
    env = dict(os.environ, LUCID_TUNING=path, LUCID_LEARNED="1")
    if stem:
        env["LUCID_MODEL_STEM"] = stem
    p = subprocess.run([APP, "--bench", clip, reference, out, "0", str(frames)],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        print(p.stdout[-500:], p.stderr[-500:])
        shutil.rmtree(out, ignore_errors=True)
        return None, None
    outputs, refs = [], []
    for name in sorted(f for f in os.listdir(out) if f.endswith("-detail.png")):
        stem_id = name.split("-")[0]
        ref = os.path.join(out, f"{stem_id}-reference.png")
        if not os.path.exists(ref):
            continue
        a = Image.open(os.path.join(out, name)).convert("L")
        r = Image.open(ref).convert("L")
        if a.size != r.size:
            a = a.resize(r.size, Image.LANCZOS)
        outputs.append(np.asarray(a, dtype=np.float64))
        refs.append(np.asarray(r, dtype=np.float64))
    shutil.rmtree(out, ignore_errors=True)
    return outputs, refs


def decode(path, frames, width=None):
    """Consecutive frames from the start of a clip, as RGB images.

    Consecutive is the whole point and is easy to get wrong: seeking per frame
    returns frames from slightly different decode states, which reads as
    flicker that is ours when it is ffmpeg's. One decode, one pass, in order."""
    out = tempfile.mkdtemp(prefix="flicker-frames-")
    scale = ["-vf", f"scale={width}:-2:flags=lanczos"] if width else []
    subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(frames)]
                   + scale + ["-fps_mode", "passthrough", os.path.join(out, "%04d.png")],
                   check=True)
    names = sorted(os.listdir(out))
    images = [Image.open(os.path.join(out, n)).convert("RGB").copy() for n in names]
    shutil.rmtree(out, ignore_errors=True)
    return images


def render_checkpoint(path, clip, reference, frames, device):
    """Runs a checkpoint over consecutive frames, feeding a temporal model its
    own history the way playback would.

    History is the frames that actually preceded this one, oldest first. The
    first frames of the clip have none, so the current frame is repeated - the
    same degenerate case as a scene cut, and the honest one: it tells the model
    nothing new rather than handing it an unrelated shot."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_span import Unshuffled

    state = torch.load(path, map_location="cpu", weights_only=False)
    history = state.get("frames", 1)
    model = Unshuffled(state.get("channels", 32), frames=history,
                       version=state.get("version", 1)).eval().to(device)
    model.load_state_dict(state["model"] if "model" in state else state)

    sources = decode(clip, frames)
    refs = [np.asarray(f.convert("L"), dtype=np.float64) for f in decode(reference, frames)]
    tensors = [torch.from_numpy(np.asarray(f, dtype=np.float32) / 255).permute(2, 0, 1)
               for f in sources]

    outputs = []
    with torch.no_grad():
        for i in range(len(tensors)):
            window = [tensors[max(i - k, 0)] for k in range(history - 1, -1, -1)]
            x = torch.cat(window, dim=0).unsqueeze(0).to(device)
            y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            outputs.append(Image.fromarray((y * 255).round().astype(np.uint8)))

    aligned = []
    for image, ref in zip(outputs, refs):
        grey = image.convert("L")
        if grey.size != (ref.shape[1], ref.shape[0]):
            grey = grey.resize((ref.shape[1], ref.shape[0]), Image.LANCZOS)
        aligned.append(np.asarray(grey, dtype=np.float64))
    return aligned, refs[:len(aligned)], history


def flicker(outputs, refs):
    """Mean frame-to-frame change in the output, over pixels the reference holds
    still. Also returns how much of the frame that mask covers, because a number
    computed over 2% of the picture is not comparable to one over 60%."""
    values, coverage = [], []
    for i in range(1, len(outputs)):
        moved = np.abs(refs[i] - refs[i - 1])
        static = moved < STATIC
        if static.sum() < 1000:
            continue
        values.append(float(np.abs(outputs[i] - outputs[i - 1])[static].mean()))
        coverage.append(float(static.mean()))
    if not values:
        return None, None
    return float(np.mean(values)), float(np.mean(coverage))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="crowdrun-360p-350k.mp4")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--tuning", default=os.path.join(ROOT, "Tools/tuning.json"))
    parser.add_argument("--stems", nargs="*",
                        default=["SPAN_x4_ch32ul1_", "SPAN_x4_ch32u_"],
                        help="model stems to compare, in order")
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help=".pth files to score in torch instead of via the app")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    known = {"crowdrun-360p-350k.mp4": "crowdrun-1080p.mp4",
             "dinner-360p-350k.mp4": "dinner-1080p.mp4"}
    clip = os.path.join(ROOT, "TestSite", args.clip)
    reference = os.path.join(ROOT, "TestSite", known.get(args.clip, ""))
    for p in (clip, reference):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    with open(args.tuning) as fh:
        tuning = json.load(fh)

    print(f"{args.clip}, {args.frames} frames")
    print("flicker is mean frame-to-frame change over pixels the reference holds")
    print("still, in 0-255 luma. Lower is steadier.\n")
    print(f"{'model':28s} {'flicker':>9s} {'static area':>12s}")
    results = {}

    if args.checkpoints:
        import torch
        device = torch.device(args.device if args.device != "mps"
                              or torch.backends.mps.is_available() else "cpu")
        # The anchor: a plain Lanczos upscale invents nothing, so whatever it
        # shimmers is the source's own noise passing through. A model below this
        # line is steadier than the footage it was given.
        sources = decode(clip, args.frames)
        refs = [np.asarray(f.convert("L"), dtype=np.float64)
                for f in decode(reference, args.frames)]
        anchor = [np.asarray(f.convert("L").resize((refs[0].shape[1], refs[0].shape[0]),
                                                   Image.LANCZOS), dtype=np.float64)
                  for f in sources]
        value, coverage = flicker(anchor, refs)
        print(f"{'lanczos':28s} {value:9.4f} {coverage*100:11.1f}%")
        for path in args.checkpoints:
            outputs, refs, history = render_checkpoint(path, clip, reference,
                                                       args.frames, device)
            value, coverage = flicker(outputs, refs)
            label = f"{os.path.basename(path).replace('.pth', '')} ({history}f)"
            results[label] = value
            print(f"{label:28s} {value:9.4f} {coverage*100:11.1f}%")
    for stem in args.stems if not args.checkpoints else []:
        outputs, refs = render(stem, clip, reference, args.frames, tuning)
        if not outputs:
            print(f"{stem:28s}  render failed"); continue
        value, coverage = flicker(outputs, refs)
        if value is None:
            print(f"{stem:28s}  no static regions - wrong clip for this test"); continue
        results[stem] = value
        print(f"{stem:28s} {value:9.4f} {coverage*100:11.1f}%")

    if len(results) == 2:
        a, b = list(results.values())
        names = list(results.keys())
        change = (b - a) / a * 100
        print(f"\n{names[1]} shimmers {abs(change):.1f}% "
              f"{'MORE' if change > 0 else 'less'} than {names[0]}")


if __name__ == "__main__":
    main()
