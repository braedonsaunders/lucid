#!/usr/bin/env python3
"""Apply the original quality/detail thresholds to the frozen native pipeline."""
import argparse
import json
from pathlib import Path
from gate_frozen_holdout import compare_rows,digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['report','frames-manifest','config','spec','sequence-manifest','out']:
        ap.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args()
    report,manifest,config,spec=[json.loads(p.read_text()) for p in [args.report,args.frames_manifest,args.config,args.spec]]
    if not report['complete'] or not manifest['complete'] or report['split']!='quality-holdout':
        raise ValueError('complete native holdout required')
    if report['manifest_sha256']!=digest(args.frames_manifest) or manifest['native_config_sha256']!=digest(args.config):
        raise ValueError('frozen frame/config identity changed')
    if manifest['source_manifest_sha256']!=digest(args.sequence_manifest):raise ValueError('source sequence manifest changed')
    expected_models={'shipping4x':'fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65','candidate':config['candidate_sha256']}
    if report['checkpoint_sha256']!=expected_models or manifest['checkpoint_sha256']!=expected_models:
        raise ValueError('checkpoint identities differ from frozen pair')
    if report['native_config_sha256']!=manifest['native_config_sha256']:raise ValueError('scoring config changed')
    expected={(s['id'],f"{s['id']}-{codec}-{bitrate}",frame) for s in spec['sources']
        for codec in spec['conditions']['codecs'] for bitrate in spec['conditions']['bitrates']
        for frame in range(0,s['frames'],spec['conditions']['spatial_stride'])}
    for label in expected_models:
        keys=[(r['source_id'],r['sequence_id'],r['frame']) for r in manifest['rows'] if r['variant']==label]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('native sample coverage differs from frozen contract')
    result=compare_rows(report['rows'],expected,spec['gate'],candidate='candidate',baseline='shipping4x',other_labels=())
    result['status']='Native sender-output spatial holdout only. The separate raw-weight fidelity gate failed; browser cadence, rendering, broader content and release requirements remain.'
    result['evidence_sha256']={str(p):digest(p) for p in [args.report,args.frames_manifest,args.config,args.spec,args.sequence_manifest,Path(__file__)]}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['spatial_gate_pass','failure_reasons','perceptual_improvements']},indent=2))


if __name__=='__main__':main()
