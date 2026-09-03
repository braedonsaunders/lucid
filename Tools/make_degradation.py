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
import argparse, glob
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
# Two Blender animated films were the whole corpus, and a model trained only on
# animation has never seen skin, film grain, sensor noise or a real lens. The
# Netflix Chimera set (live action, 4096x2160, CC BY, on media.xiph.org) fixes
# that: faces, fast motion, water, crowds, fine texture, driving POV. They are
# high-bitrate VP9 rather than raw masters - visually lossless at the scales
# used here, and far cleaner than any re-encode, but worth knowing.
CLEAN_SOURCES = [
    os.path.join(ROOT,
                 ".build/corpus-sources/bbb_sunflower_2160p_60fps_normal.mp4"),
    os.path.join(ROOT, ".build/corpus-sources/Sintel.2010.4k.mkv"),
] + sorted(glob.glob(os.path.join(ROOT, ".build/corpus-sources/fast-netflix-*.mp4")))
# The Chimera clips ship with a single keyframe in 20 seconds - they are
# research encodes tuned for compression, not for seeking - so every -ss had to
# decode up to 1200 frames of 4K VP9 and pair generation collapsed to 1.2 per
# minute. They are transcoded once to CRF 12 with a 25-frame GOP, which is
# visually lossless at these crop sizes and makes every seek cheap.
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
# (model width, model height, encode width, encode height). The encode size is
# the real streaming size; the model size is that rounded up to a multiple of
# 16, because an unaligned input width costs the Neural Engine a whole extra
# tile pass. Only 240p and 480p differ.
#
# The target is Edge's window: enabled below 720p. That needs the 854x480 tier,
# whose HR crop is 3456x1920 - Sintel 4K is 4096x1744 and cannot supply it, so
# that tier draws only from the 2160-tall sources.
LR_TIERS = [
    (256, 144, 256, 144),
    (320, 180, 320, 180),
    (432, 240, 426, 240),
    (480, 270, 480, 270),
    (640, 360, 640, 360),
    (864, 480, 854, 480),
]
CODECS = ["libx264", "libvpx-vp9"]
BITRATE_LO, BITRATE_HI = 90_000, 1_500_000
GOPS = [30, 60, 120, 250]
# Segment length is now derived per source from its frame rate; see `seg`.
# Which frame of the decoded segment becomes the pair. Counted from the first
# decoded frame, so it means the same thing for both halves regardless of the
# source's frame rate or GOP. Far enough in that the codec has stopped coasting
# off the opening keyframe, and inside 2s at both 30 and 60fps.
PAIR_FRAME = 30
SCALE = 4  # HR is 4x the LR: the fine-tune target is a 4x model.


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:6])}... failed: "
                           f"{r.stderr.strip()[-300:]}")
    return r


def probe_dims(path):
    # Parsed by key, not by position: ffprobe emits these fields in its own
    # order regardless of the order they are requested in, which silently
    # swapped duration and frame rate.
    r = sh(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration,r_frame_rate",
            "-of", "default=noprint_wrappers=1", path])
    fields = dict(line.split("=", 1) for line in r.stdout.strip().splitlines()
                  if "=" in line)
    w, h = fields["width"], fields["height"]
    d, rate = fields.get("duration", "N/A"), fields.get("r_frame_rate", "30/1")
    if d == "N/A":  # some masters omit stream duration; use container.
        d = sh(["ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0", path]).stdout.strip()
    num, _, den = rate.partition("/")
    fps = float(num) / float(den or 1)
    return int(w), int(h), float(d), fps


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
        sw, sh_, dur, fps = probe_dims(s)
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
        dims[s] = (sw, sh_, dur, fps)
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
        # Pick the content class first, then a source inside it. Choosing
        # uniformly over files gives whichever class has more files: 19 Netflix
        # clips against 2 films made the corpus 95% live action, which is the
        # original animation-only mistake with the sign flipped. Animation is
        # real deployment content - anime, cartoons, game streams - and its flat
        # shaded regions compress differently from film grain.
        animated = [f for f in fit_srcs if "netflix" not in os.path.basename(f)]
        live = [f for f in fit_srcs if "netflix" in os.path.basename(f)]
        pool = live if (live and (not animated or rng.random() < 0.65)) else animated
        src = rng.choice(pool or fit_srcs)
        sw, sh_, dur, fps = dims[src]
        # Decode only what the pair needs. A fixed 2s window is 48 frames on a
        # 24fps source and 120 on a 60fps one, and exactly one frame is kept -
        # which collapsed generation to 1.2 pairs a minute once 60fps 4K sources
        # were added. Sizing the window by frame rate keeps the codec's
        # inter-frame behaviour (the pair frame still sits deep inside the GOP)
        # while decoding a third as much on the 60fps sources.
        seg = max(0.35, (PAIR_FRAME + 8) / max(fps, 1.0))
        t0 = rng.uniform(0, max(0.1, dur - seg - 0.5))
        cx = rng.randint(0, (sw - cw) // 2) * 2 if sw > cw else 0
        cy = rng.randint(0, (sh_ - ch) // 2) * 2 if sh_ > ch else 0
        codec = rng.choice(CODECS)
        bitrate = int(10 ** rng.uniform(math.log10(BITRATE_LO),
                                        math.log10(BITRATE_HI)))
        gop = rng.choice(GOPS)

        hr_path = os.path.join(hr_dir, f"{made:06d}.png")
        lr_path = os.path.join(lr_dir, f"{made:06d}.png")
        # HR and LR must be the SAME frame, and that is harder than it looks.
        # This used to seek three times - once for HR at t0+1s, once for the
        # raw pipe at t0, and once into the encoded segment - and `-ss` before
        # `-i` is a keyframe seek, so the 4K source and the re-encoded segment
        # landed on different frames. 57% of the resulting pairs showed the
        # same shot at a different moment: a model trained on those learns to
        # invent motion. Both halves now decode from the same start and pick
        # the same frame index, and nothing seeks into the encoded file.
        sh(["ffmpeg", "-y", "-v", "error", "-ss", str(t0),
            "-i", src, "-vf",
            f"crop={cw}:{ch}:{cx}:{cy},select=eq(n\\,{PAIR_FRAME})",
            "-frames:v", "1", "-fps_mode", "passthrough", hr_path])
        # LR: same segment downscaled to the ENCODE size for a real
        # codec round-trip, then resized to the MODEL size. The two
        # differ only for 240p (426 encode -> 432 model); the resize
        # must be part of the pipeline, not the codec pass.
        p = subprocess.run(
            # -fps_mode passthrough matters as much here as on the two
            # extraction commands. Without it ffmpeg conforms this output to a
            # constant frame rate, duplicating or dropping frames on a
            # variable-rate source - which shifts every index the encoder then
            # sees, so frame 30 of the encoded segment stops being frame 30 of
            # the decode that HR came from. That desync is independent of codec
            # and GOP, which is exactly what the residual mismatches measured:
            # 14.5% for both x264 and VP9, flat across every GOP length.
            ["ffmpeg", "-v", "error", "-ss", str(t0), "-i", src,
             "-t", str(seg),
             "-vf", f"crop={cw}:{ch}:{cx}:{cy},scale={ew}:{eh}",
             "-fps_mode", "passthrough",
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
            # No -ss: decode the segment from its start and take the same
            # frame index HR took, so the two are the same instant by
            # construction rather than by luck.
            select = f"select=eq(n\\,{PAIR_FRAME})"
            chain = select if vf == "null" else f"{select},{vf}"
            sh(["ffmpeg", "-y", "-v", "error", "-i", seg,
                "-vf", chain, "-frames:v", "1",
                "-fps_mode", "passthrough", lr_path])
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
