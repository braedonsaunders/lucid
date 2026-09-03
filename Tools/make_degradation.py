#!/usr/bin/env python3
"""Builds a super-resolution training corpus that matches our content.

Each pair: HR = a clean high-res crop of a source frame at 4x the LR
size; LR = that same crop downscaled, pushed through a REAL codec
(libx264 / libvpx-vp9) as a short segment at a streaming bitrate, then
decoded. No synthetic blur+JPEG: temporal compression artifacts are
the whole point. Resolution is stratified: every block of 5 pairs
covers the five model input tiers exactly once. The 240p tier encodes
at 426x240 (what YouTube ships) and is stretched to the 432x240 the
model sees only after the codec round-trip.

Layout (torch DataLoader friendly): <out>/hr/%06d.png,
<out>/lr/%06d.png with matching basenames, plus manifest.jsonl
recording exactly how each pair was made (source, timestamp, crop,
resolution, codec, bitrate, GOP, seed).

Deterministic given --seed. --count controls corpus size (small to
prove the loop, big overnight). Ends by reporting the distribution
ACTUALLY produced, not the intended one.

Local sources default to TestSite/*.mp4 and
.build/bench/reference-1080p.mp4. --fetch also pulls Big Buck Bunny
720p and Sintel (Blender, CC-BY) from archive.org into
.build/corpus-sources/.
"""
import argparse
import collections
import json
import math
import os
import random
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# HR must be clean: only genuinely high-quality material. Low-bitrate
# clips (bbb-240p-200k etc.) would bake compression artifacts
# into the target. The floor is per tier, not global: a source must be
# at least the HR crop it is asked for, checked when the tier is drawn.
CLEAN_SOURCES = [
    os.path.join(ROOT,
                 ".build/corpus-sources/bbb_sunflower_2160p_60fps_normal.mp4"),
    os.path.join(ROOT, ".build/corpus-sources/Sintel.2010.4k.mkv"),
]
FETCH_DIR = os.path.join(ROOT, ".build/corpus-sources")
FETCH_URLS = {
    # (zip member name, download URL): Blender ships 2160p as a zip.
    "bbb_sunflower_2160p_60fps_normal.mp4": (
        "https://download.blender.org/demo/movies/BBB/"
        "bbb_sunflower_2160p_60fps_normal.mp4.zip"),
    "Sintel.2010.4k.mkv": ("https://archive.org/download/SintelDCP_201512/"
                            "Sintel.2010.4k.mkv"),
}
# (model width, model height, encode width, encode height): the five LR
# sizes the 4x model takes, and the size each is encoded at. All tiers
# encode at the model size except 240p: YouTube ships 426x240, which is
# not a multiple of 16 and costs the Neural Engine a whole extra pass,
# so live it is stretched to 432x240 on the way into the model. The
# corpus reproduces that: real codec pass at 426x240, then resize.
LR_TIERS = [
    (256, 144, 256, 144),
    (432, 240, 426, 240),
    (480, 270, 480, 270),
    (640, 360, 640, 360),
    (320, 180, 320, 180),
]
CODECS = ["libx264", "libvpx-vp9"]
BITRATE_LO, BITRATE_HI = 90_000, 1_500_000
GOPS = [30, 60, 120, 250]
SEG_SECS = 2.0  # LR segment length; decoded frame taken from the middle.
SCALE = 4  # HR is 4x the LR: the fine-tune target is a 4x model.


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:6])}... failed: "
                           f"{r.stderr.strip()[-300:]}")
    return r


def probe_dims(path):
    r = sh(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-of", "csv=p=0", path])
    w, h, d = r.stdout.strip().split(",")
    if d == "N/A":  # some masters omit stream duration; use container.
        d = sh(["ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0", path]).stdout.strip()
    return int(w), int(h), float(d)


def fetch_sources():
    import zipfile
    os.makedirs(FETCH_DIR, exist_ok=True)
    got = []
    for name, url in FETCH_URLS.items():
        dest = os.path.join(FETCH_DIR, name)
        if url.endswith(".zip"):
            # Blender ships 2160p as a zip of one mp4; want the mp4.
            if os.path.exists(dest):
                got.append(dest)
                continue
            print(f"fetching {url} ...", flush=True)
            try:
                tmp = dest + ".zip"
                urllib.request.urlretrieve(url, tmp)
                with zipfile.ZipFile(tmp) as z:
                    mp4s = [i for i in z.namelist()
                            if i.endswith(".mp4")]
                    z.extract(mp4s[0], FETCH_DIR)
                    os.rename(os.path.join(FETCH_DIR, mp4s[0]), dest)
                    got.append(dest)
                os.remove(tmp)
            except Exception as e:  # noqa: BLE001 - report, keep locals
                print(f"  skipped ({e})")
            continue
        if os.path.exists(dest):
            got.append(dest)
            continue
        print(f"fetching {url} ...", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            got.append(dest)
        except Exception as e:  # noqa: BLE001 - report, keep locals
            print(f"  skipped ({e})")
    return got


def encode_segment(lr_raw, w, h, codec, bitrate, gop, tmp):
    seg = os.path.join(tmp, "seg.mkv")
    cmd = (["ffmpeg", "-y", "-v", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-framerate", "30",
            "-i", "-"]
           + (["-c:v", "libx264", "-preset", "medium"]
              if codec == "libx264"
              else ["-c:v", "libvpx-vp9", "-quality", "good",
                    "-cpu-used", "6"])
           + ["-b:v", str(bitrate), "-g", str(gop), "-pix_fmt", "yuv420p",
              seg])
    p = subprocess.run(cmd, input=lr_raw, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg {codec} encode failed: "
                           f"{p.stderr.decode()[-300:]}")
    return seg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, ".build/corpus"))
    ap.add_argument("--fetch", action="store_true",
                    help="download BBB/Sintel from archive.org as sources")
    ap.add_argument("sources", nargs="*")
    args = ap.parse_args()

    # Sources are the 4K masters only. The fetch dir is NOT globbed:
    # older compressed clips living there (720p AVI, 2048 Sintel) would
    # otherwise leak back in as HR for the low tiers - the exact dirty-
    # target bug the 720p floor existed to prevent.
    srcs = list(args.sources or CLEAN_SOURCES)
    if args.fetch:
        srcs += fetch_sources()
    srcs = [s for s in srcs if os.path.exists(s)]
    if not srcs:
        sys.exit("no source videos found")
    # No global floor: a source is judged per tier - it must cover the
    # HR crop it is asked for. Anything that cannot is named here.
    dims = {}
    for s in srcs:
        sw, sh_, dur = probe_dims(s)
        if sw < max(t[0] for t in LR_TIERS) * SCALE or \
                sh_ < max(t[1] for t in LR_TIERS) * SCALE:
            capped = [f"{t[0]}x{t[1]}" for t in LR_TIERS
                      if t[0] * SCALE <= sw and t[1] * SCALE <= sh_]
            if not capped:
                print(f"REJECTED {s}: {sw}x{sh_} cannot supply any "
                      f"HR crop - smallest needs "
                      f"{min(t[0] for t in LR_TIERS) * SCALE}x"
                      f"{min(t[1] for t in LR_TIERS) * SCALE}")
                continue
            print(f"LIMITED {s}: {sw}x{sh_} can only supply tiers "
                  f"{','.join(capped)}")
        dims[s] = (sw, sh_, dur)
    srcs = list(dims)
    if not srcs:
        sys.exit("no source can supply any HR crop")
    print(f"sources ({len(srcs)}): " +
          ", ".join(f"{os.path.basename(s)} {dims[s][0]}x{dims[s][1]}"
                    for s in srcs))

    hr_dir = os.path.join(args.out, "hr")
    lr_dir = os.path.join(args.out, "lr")
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)
    man_path = os.path.join(args.out, "manifest.jsonl")
    man = open(man_path, "w")

    rng = random.Random(args.seed)
    # Stratified resolution: every block of 5 pairs covers all five LR
    # tiers in a per-epoch shuffled order. Plain rng.choice can clump.
    tier_order = {}
    def lr_tier(i):
        epoch, pos = divmod(i, len(LR_TIERS))
        if epoch not in tier_order:
            e = list(LR_TIERS)
            random.Random(args.seed + epoch).shuffle(e)
            tier_order[epoch] = e
        return tier_order[epoch][pos]
    made = 0
    attempt = 0
    while made < args.count:
        attempt += 1
        if attempt > args.count * 50:
            sys.exit("too many skips - sources too small for LR sizes?")
        # HR is 4x the LR: the fine-tune target is a 4x model, so every
        # pair teaches the same mapping.
        mw, mh, ew, eh = lr_tier(made)
        cw, ch = mw * SCALE, mh * SCALE
        fit_srcs = [s for s in srcs
                    if dims[s][0] >= cw and dims[s][1] >= ch]
        if not fit_srcs:
            print(f"  tier {mw}x{mh} has no source covering "
                  f"{cw}x{ch} - skipped")
            continue
        src = rng.choice(fit_srcs)
        sw, sh_, dur = dims[src]
        t0 = rng.uniform(0, max(0.1, dur - SEG_SECS - 0.5))
        cx = rng.randint(0, (sw - cw) // 2) * 2 if sw > cw else 0
        cy = rng.randint(0, (sh_ - ch) // 2) * 2 if sh_ > ch else 0
        codec = rng.choice(CODECS)
        bitrate = int(10 ** rng.uniform(math.log10(BITRATE_LO),
                                        math.log10(BITRATE_HI)))
        gop = rng.choice(GOPS)

        hr_path = os.path.join(hr_dir, f"{made:06d}.png")
        lr_path = os.path.join(lr_dir, f"{made:06d}.png")
        # HR: clean crop of the segment's middle frame.
        sh(["ffmpeg", "-y", "-v", "error", "-ss", str(t0 + SEG_SECS / 2),
            "-i", src, "-frames:v", "1",
            "-vf", f"crop={cw}:{ch}:{cx}:{cy}", hr_path])
        # LR: same segment downscaled to the ENCODE size for a real
        # codec round-trip, then resized to the MODEL size. The two
        # differ only for 240p (426 encode -> 432 model); the resize
        # must be part of the pipeline, not the codec pass.
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", str(t0), "-i", src,
             "-t", str(SEG_SECS),
             "-vf", f"crop={cw}:{ch}:{cx}:{cy},scale={ew}:{eh}",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True)
        if p.returncode != 0:
            continue
        tmp = os.path.join(args.out, "tmp")
        os.makedirs(tmp, exist_ok=True)
        try:
            seg = encode_segment(p.stdout, ew, eh, codec, bitrate, gop,
                                 tmp)
            vf = (f"scale={mw}:{mh}" if (mw, mh) != (ew, eh)
                  else "null")
            sh(["ffmpeg", "-y", "-v", "error", "-ss",
                str(SEG_SECS / 2), "-i", seg, "-frames:v", "1",
                "-vf", vf, lr_path])
        except RuntimeError as e:
            print(f"  pair {made} skipped: {e}")
            continue
        rec = {"idx": made, "seed": args.seed, "source": src,
               "t0": round(t0, 3),
               "hr_crop": {"x": cx, "y": cy, "w": cw, "h": ch},
               "lr": {"w": mw, "h": mh, "encode_w": ew, "encode_h": eh,
                      "codec": codec, "bitrate": bitrate, "gop": gop}}
        man.write(json.dumps(rec) + "\n")
        made += 1
    man.close()

    # Actual distribution, measured from the manifest just written.
    recs = [json.loads(l) for l in open(man_path)]
    print(f"wrote {len(recs)} pairs to {args.out} (seed {args.seed})")
    for key in ("codec", "gop"):
        print(f"  {key}: {dict(sorted(collections.Counter(
            r['lr'][key] for r in recs).items()))}")
    print(f"  lr_size: {dict(sorted(collections.Counter(
        (r['lr']['w'], r['lr']['h']) for r in recs).items()))}")
    bins = collections.Counter(
        f"{b // 1000}k" if (b := r['lr']['bitrate']) < 1000 * 1000
        else f"{b / 1000000:.1f}M" for r in recs)
    print(f"  bitrate: {dict(sorted(bins.items()))}")
    print(f"  sources: {dict(sorted(collections.Counter(
        os.path.basename(r['source']) for r in recs).items()))}")


if __name__ == "__main__":
    main()
