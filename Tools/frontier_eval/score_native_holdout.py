#!/usr/bin/env python3
"""Score already frozen RGB native outputs against their frozen references."""
import argparse
import json
import math
from pathlib import Path
from PIL import Image
import torch
from evaluate_sequences import Scorer,digest,save,summarize


def reusable_scores(frame_directory, report_path, device):
    """Reuse only scores tied to identical RGB/reference bytes and metric code."""
    source = Path(frame_directory) / 'manifest.json'
    manifest = json.loads(source.read_text())
    report = json.loads(Path(report_path).read_text())
    if (not manifest.get('complete') or not report.get('complete') or
            report['manifest_sha256'] != digest(source) or
            report['scorer_sha256'] != digest(Path(__file__).with_name('evaluate_sequences.py')) or
            report['torch'] != str(torch.__version__) or report['device'] != device):
        raise ValueError('score cache provenance or execution environment differs')
    key = lambda row: (row['sequence_id'], row['variant'], row['frame'])
    frames = {key(row): row for row in manifest['rows']}
    scores = {key(row): row for row in report['rows']}
    if len(frames) != len(manifest['rows']) or len(scores) != len(report['rows']) or frames.keys() != scores.keys():
        raise ValueError('score cache is incomplete or contains duplicate samples')
    result = {}
    for identity, frame in frames.items():
        row = scores[identity]
        if (row['source_id'] != frame['source_id'] or
                set(row['metrics']) != {'lpips', 'dists', 'psnr_y', 'detail_energy', 'fine_correlation'} or
                not all(math.isfinite(v) for v in row['metrics'].values())):
            raise ValueError('score cache contains invalid measurements')
        pixels = (frame['image_sha256'], frame['reference_sha256'])
        if pixels in result and result[pixels] != row['metrics']:
            raise ValueError('identical cached pixels have conflicting scores')
        result[pixels] = row['metrics']
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames',type=Path,required=True)
    ap.add_argument('--references',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--development-presentation',action='store_true')
    ap.add_argument('--reuse-scores', nargs=2, metavar=('RGB_DIRECTORY', 'REPORT'),
                    help='Reuse validated scores only for identical RGB and reference image hashes')
    args=ap.parse_args()
    source=args.frames/'manifest.json';manifest=json.loads(source.read_text())
    split='development-presentation' if args.development_presentation else 'quality-holdout'
    if not manifest['complete'] or manifest['split']!=split:raise ValueError('complete matching frozen native split required')
    expected=set()
    for row in manifest['rows']:
        key=row['sequence_id'],row['variant'],row['frame']
        if key in expected:raise ValueError('duplicate native sample')
        expected.add(key)
        if digest(args.frames/row['file'])!=row['image_sha256']:raise ValueError('native image changed')
        if digest(args.references/row['reference'])!=row['reference_sha256']:raise ValueError('reference changed')
    report={'purpose':'native sender-output spatial '+split+'; no browser presentation/cadence claim','split':split,
        'manifest_sha256':digest(source),'native_config_sha256':manifest['native_config_sha256'],
        'checkpoint_sha256':manifest['checkpoint_sha256'],'code_sha256':digest(__file__),
        'scorer_sha256':digest(Path(__file__).with_name('evaluate_sequences.py')),
        'torch':str(torch.__version__),'device':args.device,'rows':[],'complete':False}
    cache = reusable_scores(*args.reuse_scores, args.device) if args.reuse_scores else {}
    report['score_reuse'] = {'report_sha256': digest(args.reuse_scores[1]) if args.reuse_scores else None,
                             'identical_pixel_pairs': 0}
    scorer=Scorer(torch.device(args.device))
    for row in manifest['rows']:
        metrics = cache.get((row['image_sha256'], row['reference_sha256']))
        if metrics is None:
            with Image.open(args.frames/row['file']) as image:output=image.convert('RGB')
            with Image.open(args.references/row['reference']) as image:reference=image.convert('RGB')
            metrics = scorer.spatial(output,reference)
        else:
            report['score_reuse']['identical_pixel_pairs'] += 1
        report['rows'].append({**{k:row[k] for k in ['source_id','sequence_id','variant','frame']},'metrics':metrics})
        report['summary']=summarize(report['rows']);save(args.report,report)
        print(row['sequence_id'],row['variant'],row['frame'],flush=True)
    report['complete']=True;save(args.report,report)


if __name__=='__main__':main()
