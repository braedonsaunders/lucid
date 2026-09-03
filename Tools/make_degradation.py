#!/usr/bin/env python3
"""Builds a super-resolution training corpus that matches our content.

Each pair: HR = a clean high-res crop of a source frame at 2x the LR
size; LR = that same crop downscaled, pushed through a REAL codec
(libx264 / libvpx-vp9) as a short segment at a streaming bitrate, then
decoded. No synthetic blur+JPEG: temporal compression artifacts are
the whole point. Resolution is stratified: every block of 5 pairs
covers 144p/240p/270p/360p/480p exactly once.

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
# into the target, so anything below 720p is rejected, not used.
MIN_SOURCE_H = 720
CLEAN_SOURCES = [
    os.path.join(ROOT, ".build/bench/reference-1080p.mp4"),
    os.path.join(ROOT, "TestSite/bbb-1080p60-1700k.mp4"),
]
FETCH_DIR = os.path.join(ROOT, ".build/corpus-sources")
FETCH_URLS = {
    # NOTE: the archive.org "720p_surround.mp4" is really 640x360;
    # the AVI is the true 1280x720 master.
    "bbb-720p.avi": ("https://archive.org/download/BigBuckBunny_124/"
                     "Content/big_buck_bunny_720p_surround.avi"),
    "sintel-2048.mp4": ("https://archive.org/download/Sintel/"
                        "sintel-2048-stereo.mp4"),
}
# (width, height) LR operating points, 144p through 480p.
LR_SIZES = [(256, 144), (426, 240), (480, 270), (640, 360), (854, 480)]
CODECS = ["libx264", "libvpx-vp9"]
BITRATE_LO, BITRATE_HI = 90_000, 1_500_000
GOPS = [30, 60, 120, 250]
SEG_SECS = 2.0  # LR segment length; decoded frame taken from the middle.
SCALE = 2  # HR is 2x the LR: the fine-tune target is a 2x model.


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
    return int(w), int(h), float(d)


def fetch_sources():
    os.makedirs(FETCH_DIR, exist_ok=True)
    got = []
    for name, url in FETCH_URLS.items():
        dest = os.path.join(FETCH_DIR, name)
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

    srcs = (args.sources or list(CLEAN_SOURCES)
            + sorted(f for f in
                     ([os.path.join(FETCH_DIR, f)
                       for f in sorted(os.listdir(FETCH_DIR))]
                      if os.path.isdir(FETCH_DIR) else [])
                     if f.endswith((".mp4", ".avi"))))
    if args.fetch:
        srcs += fetch_sources()
    srcs = [s for s in srcs if os.path.exists(s)]
    if not srcs:
        sys.exit("no source videos found")
    dims = {}
    for s in srcs:
        sw, sh_, dur = probe_dims(s)
        if sh_ < MIN_SOURCE_H:
            print(f"REJECTED {s}: {sw}x{sh_} below {MIN_SOURCE_H}p floor "
                  f"- HR must be clean, low-bitrate clips would bake "
                  f"artifacts into the target")
            continue
        dims[s] = (sw, sh_, dur)
    srcs = list(dims)
    if not srcs:
        sys.exit("no source at or above 720p")
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
    # sizes in a per-epoch shuffled order. Plain rng.choice can clump
    # (a 4-pair run came out all 144p); the corpus must span 144p-480p.
    size_order = {}
    def lr_size(i):
        epoch, pos = divmod(i, len(LR_SIZES))
        if epoch not in size_order:
            e = list(LR_SIZES)
            random.Random(args.seed + epoch).shuffle(e)
            size_order[epoch] = e
        return size_order[epoch][pos]
    made = 0
    attempt = 0
    while made < args.count:
        attempt += 1
        if attempt > args.count * 50:
            sys.exit("too many skips - sources too small for LR sizes?")
        # HR is 2x the LR: the fine-tune target is a 2x model, so every
        # pair teaches the same mapping. A fixed HR size would mix 1.1x
        # through 3.75x factors across pairs.
        lw, lh = lr_size(made)
        cw, ch = lw * 2, lh * 2
        fit_srcs = [s for s in srcs
                    if dims[s][0] >= cw and dims[s][1] >= ch]
        if not fit_srcs:
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
        # LR: same segment downscaled, real codec round-trip.
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", str(t0), "-i", src,
             "-t", str(SEG_SECS),
             "-vf", f"crop={cw}:{ch}:{cx}:{cy},scale={lw}:{lh}",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True)
        if p.returncode != 0:
            continue
        tmp = os.path.join(args.out, "tmp")
        os.makedirs(tmp, exist_ok=True)
        try:
            seg = encode_segment(p.stdout, lw, lh, codec, bitrate, gop,
                                 tmp)
            sh(["ffmpeg", "-y", "-v", "error", "-ss",
                str(SEG_SECS / 2), "-i", seg, "-frames:v", "1",
                lr_path])
        except RuntimeError as e:
            print(f"  pair {made} skipped: {e}")
            continue
        rec = {"idx": made, "seed": args.seed, "source": src,
               "t0": round(t0, 3),
               "hr_crop": {"x": cx, "y": cy, "w": cw, "h": ch},
               "lr": {"w": lw, "h": lh, "codec": codec,
                      "bitrate": bitrate, "gop": gop}}
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
