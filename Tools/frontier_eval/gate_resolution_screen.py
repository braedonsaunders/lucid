#!/usr/bin/env python3
"""Gate a frozen spatial resolution screen against every declared interpolation floor."""
import argparse
import json
from pathlib import Path
from gate_frozen_holdout import compare_rows, digest


def evaluate(report, spec, manifest):
    if not report.get('complete') or report.get('split') != spec['split']:
        raise ValueError('complete report in the declared split required')
    if report['manifest_sha256'] != spec['frames_manifest_sha256']:
        raise ValueError('frozen decoded frame manifest changed')
    if report['sequence_manifest_sha256'] != manifest['sequence_manifest_sha256']:
        raise ValueError('sequence manifest differs')
    if report['checkpoint_sha256'] != {spec['candidate']: spec['candidate_sha256'], 'shipping': spec['shipping_sha256']}:
        raise ValueError('candidate or shipping checkpoint changed')
    if report['interpolation'] != spec['interpolation'] or report['presentation_adapters'] != ['shipping']:
        raise ValueError('comparison routes changed')
    expected={(source,f'{source}-{codec}-{rate}',frame) for source in spec['sources']
              for codec in spec['codecs'] for rate in spec['bitrates'] for frame in spec['frames']}
    for side in ['reference', 'degraded']:
        rows=[r for r in manifest['frames'] if r['side']==side]
        keys=[(r['source_id'],r['sequence_id'],r['frame']) for r in rows]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:
            raise ValueError('incomplete or unexpected frame export')
    labels={spec['candidate'],'shipping',*spec['interpolation']}
    if {r['variant'] for r in report['rows']} != labels:
        raise ValueError('unexpected comparison variants')
    comparisons={}
    for baseline in [*spec['interpolation'],'shipping']:
        comparisons[baseline]=compare_rows(report['rows'],expected,spec['gate'],
            candidate=spec['candidate'],baseline=baseline,
            other_labels=tuple(sorted(labels-{spec['candidate'],baseline})))
        comparisons[baseline]['status']='resolution development screen; native delivery and independent release coverage remain'
    return {'coverage_spatial_screen_pass':all(comparisons[x]['spatial_gate_pass'] for x in spec['interpolation']),
            'comparisons':comparisons,'shipping_replacement_gate_is_informational':True,
            'status':'spatial development evidence only; never production admission by itself'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['report','spec','manifest','out']:ap.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args()
    report,spec,manifest=[json.loads(p.read_text()) for p in [args.report,args.spec,args.manifest]]
    if digest(args.manifest)!=spec['frames_manifest_sha256']:raise ValueError('manifest file changed')
    result=evaluate(report,spec,manifest)
    result['evidence_sha256']={str(p):digest(p) for p in [args.report,args.spec,args.manifest,Path(__file__),Path(__file__).with_name('gate_frozen_holdout.py')]}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'coverage_spatial_screen_pass':result['coverage_spatial_screen_pass'],
        'comparisons':{k:{key:v[key] for key in ['spatial_gate_pass','perceptual_improvements','failure_reasons']} for k,v in result['comparisons'].items()}},indent=2))


if __name__=='__main__':main()
