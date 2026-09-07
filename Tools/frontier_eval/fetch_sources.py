#!/usr/bin/env python3
"""Fetch bounded uncompressed evaluation excerpts, preserving samples with FFV1.

Source clips stay outside git. Record their provenance and the exact excerpt hash.
These clips are reserved for evaluation and must never enter the training bank.
"""
import argparse
import concurrent.futures
import hashlib
import json
import re
from pathlib import Path
import subprocess
import urllib.request

BASE = 'https://media.xiph.org/video/derf/'
SVT = BASE + 'vqeg.its.bldrdoc.gov/HDTV/SVT_MultiFormat/SVT_MultiFormat_v10.pdf'
SOURCES = [
    ('four_people', 'FourPeople_1280x720_60.y4m', 'Public domain (Xiph collection)', None),
    ('johnny', 'Johnny_1280x720_60.y4m', 'Public domain (Xiph collection)', None),
    ('ducks_take_off', 'ducks_take_off_1080p50.y4m', 'SVT: testing/developing/presenting technology standards; see accompanying terms', SVT),
    ('old_town_cross', 'old_town_cross_1080p50.y4m', 'SVT: testing/developing/presenting technology standards; see accompanying terms', SVT),
]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def y4m_layout(header):
    text = header.decode('ascii')
    if not text.startswith('YUV4MPEG2 '):
        raise ValueError('invalid Y4M header')
    width, height = (re.search(rf' {key}(\d+)(?:\s|$)', text) for key in ('W', 'H'))
    if width is None or height is None:
        raise ValueError('Y4M dimensions missing')
    w, h = int(width.group(1)), int(height.group(1))
    chroma = re.search(r' C([^\s]+)', text)
    alternative = re.search(r' XYSCSS=([^\s]+)', text)
    # YUV4MPEG2 defines absent C as 8-bit 420jpeg, also used by the raw SVT files.
    layout = chroma.group(1) if chroma else (alternative.group(1).lower() if alternative else '420jpeg')
    if min(w, h) < 2 or w % 2 or h % 2:
        raise ValueError(f'unsupported source sample layout: {text}')
    if layout in ('420', '420jpeg', '420mpeg2', '420paldv'):
        return w, h, w * h * 3 // 2
    if layout == '422':
        # 4:2:2 masters (most derf 1080p live action); the excerpt is converted
        # to 4:2:0 by FFmpeg when it is preserved, recorded in the receipt.
        return w, h, w * h * 2
    raise ValueError(f'unsupported source sample layout: {text}')


def download_ranges(url, destination, size, workers):
    """Bounded parallel reads for large raw excerpts, with strict range checks."""
    chunk = 16 * 1024 * 1024
    with destination.open('wb') as stream:
        stream.truncate(size)
    def transfer(start):
        end = min(size, start + chunk) - 1
        request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(request, timeout=30) as response:
            content_range = response.headers.get('Content-Range', '')
            if response.status != 206 or not content_range.startswith(f'bytes {start}-{end}/'):
                raise ValueError('server did not honor exact excerpt range')
            payload = response.read(end - start + 2)
        if len(payload) != end - start + 1:
            raise ValueError('incomplete or oversized excerpt range')
        # Independent file handles and non-overlapping offsets avoid shared seeks.
        with destination.open('r+b') as stream:
            stream.seek(start)
            stream.write(payload)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(transfer, range(0, size, chunk)))


def fetch(entry, out, frames, download_workers=1):
    name, remote, license_name, license_url = entry
    destination = out / f'{name}.mkv'
    receipt = out / f'{name}.json'
    url = BASE + 'y4m/' + remote
    if destination.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['frames'] == frames and saved['sha256'] == digest(destination):
            print(f'{name}: verified existing excerpt', flush=True)
            return saved
    temporary = out / f'{name}.partial.mkv'
    # The public raw streams are large. Request only the exact leading byte
    # range; direct FFmpeg HTTP probing can read far beyond a short excerpt.
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-511'}), timeout=30) as r:
        header = r.readline()
    w, h, frame_bytes = y4m_layout(header)
    byte_count = len(header) + frames * (6 + frame_bytes)
    raw = out / f'{name}.partial.y4m'
    download = ['curl', '-f', '-sS', '--max-time', '480', '--range', f'0-{byte_count-1}', '-o', str(raw), url]
    cmd = ['ffmpeg', '-nostdin', '-y', '-v', 'error', '-f', 'yuv4mpegpipe',
           '-i', str(raw), '-map', '0:v:0', '-frames:v', str(frames), '-an', '-pix_fmt', 'yuv420p',
           '-c:v', 'ffv1', '-level', '3', '-threads', '2', str(temporary)]
    try:
        if download_workers > 1:
            download_ranges(url, raw, byte_count, download_workers)
        else:
            subprocess.run(download, check=True, timeout=500)
        if raw.stat().st_size != byte_count:
            raise RuntimeError(f'{name}: server did not provide exact byte range')
        with raw.open('rb') as stream:
            if stream.readline() != header:
                raise RuntimeError('source changed during download')
            for _ in range(frames):
                if stream.read(6) != b'FRAME\n':
                    raise RuntimeError('variable Y4M frame tags require a full demux')
                stream.seek(frame_bytes, 1)
        subprocess.run(cmd, check=True, timeout=120)
        info = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,pix_fmt,r_frame_rate,nb_read_frames',
            '-of', 'json', str(temporary)]))['streams'][0]
        if int(info['nb_read_frames']) != frames:
            raise RuntimeError(f'{name}: incomplete excerpt: {info}')
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        raw.unlink(missing_ok=True)
    result = {'id': name, 'path': str(destination.resolve()), 'url': url,
              'license': license_name, 'license_url': license_url,
              'collection': BASE, 'first_frame': 0, 'frames': frames,
              'transform': 'first frames; lossless FFV1; no spatial or pixel-format conversion',
              'sha256': digest(destination), 'stream': info, 'command': cmd,
              'download_command': download if download_workers == 1 else None,
              'download_ranges': {'workers': download_workers, 'chunk_bytes': 16 * 1024 * 1024,
                                  'bytes': byte_count} if download_workers > 1 else None,
              'use': 'evaluation only; excluded from training'}
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(f'{name}: {frames} frames, {destination.stat().st_size / 1e6:.1f} MB', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('.build/frontier-eval-sources'))
    ap.add_argument('--frames', type=int, default=48)
    ap.add_argument('--names', nargs='*', help='subset of registered source ids')
    args = ap.parse_args()
    if not 16 <= args.frames <= 500:
        ap.error('--frames must be between 16 and 500')
    args.out.mkdir(parents=True, exist_ok=True)
    terms = args.out / 'SVT_MultiFormat_v10.pdf'
    if not terms.exists():
        with urllib.request.urlopen(SVT, timeout=30) as r:
            terms.write_bytes(r.read())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        entries = [s for s in SOURCES if not args.names or s[0] in args.names]
        if not entries:
            ap.error('no registered sources selected')
        results = list(pool.map(lambda e: fetch(e, args.out, args.frames), entries))
    (args.out / 'sources.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
