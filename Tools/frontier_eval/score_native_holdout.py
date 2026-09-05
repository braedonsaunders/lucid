#!/usr/bin/env python3
"""Score already frozen RGB native outputs against their frozen references."""
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from evaluate_sequences import Scorer,digest,save,summarize


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames',type=Path,required=True)
    ap.add_argument('--references',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--device',default='cuda')
    args=ap.parse_args()
    source=args.frames/'manifest.json';manifest=json.loads(source.read_text())
    if not manifest['complete'] or manifest['split']!='quality-holdout':raise ValueError('complete frozen native holdout required')
    expected=set()
    for row in manifest['rows']:
        key=row['sequence_id'],row['variant'],row['frame']
        if key in expected:raise ValueError('duplicate native sample')
        expected.add(key)
        if digest(args.frames/row['file'])!=row['image_sha256']:raise ValueError('native image changed')
        if digest(args.references/row['reference'])!=row['reference_sha256']:raise ValueError('reference changed')
    report={'purpose':'native sender-output spatial holdout; no browser presentation/cadence claim','split':'quality-holdout',
        'manifest_sha256':digest(source),'native_config_sha256':manifest['native_config_sha256'],
        'checkpoint_sha256':manifest['checkpoint_sha256'],'code_sha256':digest(__file__),
        'scorer_sha256':digest(Path(__file__).with_name('evaluate_sequences.py')),
        'torch':str(torch.__version__),'device':args.device,'rows':[],'complete':False}
    scorer=Scorer(torch.device(args.device))
    for row in manifest['rows']:
        with Image.open(args.frames/row['file']) as image:output=image.convert('RGB')
        with Image.open(args.references/row['reference']) as image:reference=image.convert('RGB')
        report['rows'].append({**{k:row[k] for k in ['source_id','sequence_id','variant','frame']},'metrics':scorer.spatial(output,reference)})
        report['summary']=summarize(report['rows']);save(args.report,report)
        print(row['sequence_id'],row['variant'],row['frame'],flush=True)
    report['complete']=True;save(args.report,report)


if __name__=='__main__':main()
