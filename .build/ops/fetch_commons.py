#!/usr/bin/env python3
"""Fetch CC0 / CC-BY videos (>=1080p, 15-120 s) from Wikimedia Commons with receipts."""
import json, sys, time, urllib.parse, urllib.request, hashlib, subprocess
from pathlib import Path
API = 'https://commons.wikimedia.org/w/api.php'
UA = {'User-Agent': 'LucidResearchFetch/1.0 (open-source video super-resolution research; bsaunders@rassaun.com)'}
out = Path('.build/commons-sources'); out.mkdir(exist_ok=True)
def api(params):
    q = urllib.parse.urlencode(dict(params, format='json'))
    with urllib.request.urlopen(urllib.request.Request(API + '?' + q, headers=UA), timeout=60) as r:
        return json.load(r)
CATEGORIES = ['Category:Videos in 4K resolution', 'Category:Videos of nature', 'Category:Videos of sports', 'Category:Videos of cities',
              'Category:Videos of people', 'Category:Videos of animals', 'Category:Drone videos', 'Category:Timelapse videos',
              'Category:Videos of trains', 'Category:Videos of street scenes']
OK_LICENSES = ('cc0', 'cc-by-4.0', 'cc-by-3.0', 'cc-by-sa-4.0', 'cc-by-sa-3.0', 'pd')
rows, seen = [], set()
budget_bytes, total = 6 * 1024**3, 0
for cat in CATEGORIES:
    try:
        members = api({'action': 'query', 'list': 'categorymembers', 'cmtitle': cat, 'cmtype': 'file', 'cmlimit': 200})['query']['categorymembers']
    except Exception as e:
        print('FAIL cat', cat, e, flush=True); continue
    titles = [m['title'] for m in members if m['title'].lower().endswith(('.webm', '.mp4', '.ogv'))]
    for i in range(0, len(titles), 25):
        chunk = titles[i:i+25]
        try:
            info = api({'action': 'query', 'titles': '|'.join(chunk), 'prop': 'imageinfo', 'iiprop': 'url|size|extmetadata|mime|sha1', 'iiextmetadatafilter': 'LicenseShortName|License|Artist|Credit'})
        except Exception as e:
            print('FAIL info', e, flush=True); continue
        for page in info['query']['pages'].values():
            ii = (page.get('imageinfo') or [None])[0]
            if not ii: continue
            w, h, size, dur = ii.get('width', 0), ii.get('height', 0), ii.get('size', 0), ii.get('duration', 0)
            lic = (ii.get('extmetadata', {}).get('License', {}).get('value') or ii.get('extmetadata', {}).get('LicenseShortName', {}).get('value') or '').lower()
            if h < 1080 or not (15 <= dur <= 180) or size > 900 * 1024**2 or size < 20 * 1024**2: continue
            if not any(lic.startswith(k) for k in OK_LICENSES): continue
            if page['title'] in seen: continue
            seen.add(page['title'])
            if total + size > budget_bytes: continue
            name = hashlib.sha1(page['title'].encode()).hexdigest()[:10] + Path(page['title']).suffix.lower()
            dst = out / name
            if not dst.exists():
                try:
                    subprocess.run(['curl', '-sSL', '--fail', '--retry', '2', '-A', UA['User-Agent'], '-o', str(dst), ii['url']], check=True, timeout=1800)
                except Exception as e:
                    print('FAIL dl', page['title'], e, flush=True); dst.unlink(missing_ok=True); continue
            total += size
            rows.append({'id': 'commons_' + name.split('.')[0], 'title': page['title'], 'url': ii['url'], 'license': lic, 'artist': ii.get('extmetadata', {}).get('Artist', {}).get('value', '')[:200],
                         'width': w, 'height': h, 'seconds': dur, 'bytes': size, 'path': str(dst.resolve()), 'category': cat})
            print('OK', page['title'][:70], f'{w}x{h}', f'{dur:.0f}s', lic, flush=True)
            (out / 'sources.json').write_text(json.dumps(rows, indent=2) + '\n')
            if len(rows) >= 24: break
        if len(rows) >= 24: break
    if len(rows) >= 24: break
print('COMMONS-DONE', len(rows), 'files', round(total/1024**3, 2), 'GB', flush=True)
