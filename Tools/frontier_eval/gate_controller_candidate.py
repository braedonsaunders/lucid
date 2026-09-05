#!/usr/bin/env python3
"""Apply unchanged controller development gates with reproduced frozen controls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from gate_frozen_holdout import compare_rows

SHIPPING='fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65'
MANIFESTS={'development':('aac663ad088fede87e1ede68cb46501603fc0f0b066b5ae6c07726b6555d923f',48),
           'bank':('eb8ad3792cc1999f4dc593494f959656be28b815ee21d3a9094c47b683f3a7fe',96)}
GATE=dict(source_balanced_lpips_improvement_min=.03,source_balanced_dists_improvement_min=.03,
          per_source_perceptual_regression_max=.02,per_source_fine_correlation_drop_max=.01)


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(r):return r['source_id'],r['sequence_id'],r['frame']


def evaluate(report,control,mode,checkpoint_hash):
    manifest,count=MANIFESTS[mode]
    for value in (report,control):
        if (value.get('complete') is not True or value['split']!='development-validation'
                or value['manifest_sha256']!=manifest or value['presentation_adapters']!=['shipping']
                or value['checkpoint_sha256']['shipping']!=SHIPPING):
            raise ValueError('complete fixed development pixels, adapter and shipping required')
    if report['sequence_manifest_sha256']!=control['sequence_manifest_sha256']:
        raise ValueError('source sequence manifest changed')
    if report['checkpoint_sha256']!={'coarse':checkpoint_hash,'shipping':SHIPPING}:
        raise ValueError('evaluated checkpoint differs')
    if {r['variant'] for r in report['rows']}!={'coarse','shipping','lanczos'}:
        raise ValueError('unexpected variants')
    prior=[r for r in control['rows'] if r['variant']=='shipping']
    expected={key(r) for r in prior}
    if len(prior)!=count or len(expected)!=count:raise ValueError('control sample coverage changed')
    max_delta=0.
    for label in ('shipping','lanczos'):
        old_rows=[r for r in control['rows'] if r['variant']==label]
        old={key(r):r['metrics'] for r in old_rows}
        new_rows=[r for r in report['rows'] if r['variant']==label]
        new={key(r):r['metrics'] for r in new_rows}
        if len(old_rows)!=count or len(new_rows)!=count or set(old)!=expected or set(new)!=expected:
            raise ValueError('control coverage changed')
        for sample in expected:
            if set(old[sample])!=set(new[sample]):raise ValueError('control metrics changed')
            for metric,value in old[sample].items():
                delta=abs(value-new[sample][metric])
                if not math.isfinite(delta) or delta>1e-5:raise ValueError('frozen control failed to reproduce')
                max_delta=max(max_delta,delta)
    result=compare_rows(report['rows'],expected,GATE,candidate='coarse')
    result.update(gate=GATE,controls_max_absolute_delta=max_delta,promotion_authorized=False,
                  status='Repeated controller development screen; fresh independent and native/browser quality still required')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=MANIFESTS,required=True)
    for name in ('report','control','checkpoint','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():ap.error('fresh gate output required')
    result=evaluate(json.loads(args.report.read_text()),json.loads(args.control.read_text()),args.mode,digest(args.checkpoint))
    result['evidence_sha256']={str(p):digest(p) for p in (args.report,args.control,args.checkpoint,Path(__file__),Path(__file__).with_name('gate_frozen_holdout.py'))}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('spatial_gate_pass','perceptual_improvements','failure_reasons')},indent=2))


if __name__=='__main__':main()
