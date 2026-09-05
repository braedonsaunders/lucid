#!/usr/bin/env python3
"""Verify downloaded final controller weights and the matched training provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'Tools/experiments'))
from train_presented_detail import state_digest

FIELDS = ('bank_sha256', 'checkpoint_sha256', 'teacher_manifest_sha256',
          'shipping_manifest_sha256', 'probe_sha256', 'frozen_anchor_and_masks_sha256',
          'controller_initial_sha256', 'discriminator_initial_sha256',
          'first_batch_sha256', 'first_output_sha256')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoints', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    report = {'complete': False, 'arms': {}, 'matched_fields': list(FIELDS)}
    experiments = []
    for label in ('unconstrained', 'constrained'):
        directory = args.checkpoints/label
        experiment = json.loads((directory/'experiment.json').read_text())
        completion = json.loads((directory/'complete.json').read_text())
        path = directory/'step008000.pth'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if completion['steps'] != 8000 or completion['frozen_anchor_unchanged'] is not True:
            raise ValueError('training incomplete or reconstruction changed')
        if checkpoint['step'] != 8000 or checkpoint['architecture'] != 'activation_control2x':
            raise ValueError('wrong checkpoint')
        if checkpoint['experiment'] != experiment:
            raise ValueError('checkpoint provenance differs')
        if checkpoint['controller_config']['dynamic'] is not True:
            raise ValueError('arm labels differ')
        state = checkpoint['model']
        frozen = state_digest({k: v for k, v in state.items() if not k.startswith('controller.')})
        controller = state_digest({k[len('controller.'):]: v for k, v in state.items() if k.startswith('controller.')})
        if frozen != experiment['frozen_anchor_and_masks_sha256']:
            raise ValueError('downloaded frozen weights or masks differ from initialization')
        if controller != completion['controller_final_sha256'] or controller == experiment['controller_initial_sha256']:
            raise ValueError('controller is unchanged or differs from completion receipt')
        smoke = json.loads((Path(__file__).parent/f'local-fidelity-r4-{label}-smoke-experiment.json').read_text())
        for field in FIELDS:
            if experiment[field] != smoke[field]:
                raise ValueError('full training differs from verified smoke: '+field)
        if experiment['deterministic_algorithms'] is not True or experiment['args']['local_fidelity_constraint'] != (label == 'constrained'):
            raise ValueError('wrong execution mode or constraint flag')
        if label == 'constrained':
            if experiment['first_fidelity_baseline_sha256'] != smoke['first_fidelity_baseline_sha256']:
                raise ValueError('aligned shipping baseline changed')
            if completion['fidelity_multipliers'] is None or len(completion['fidelity_multipliers']) != 2 or not all(1 <= x < float('inf') for x in completion['fidelity_multipliers']):
                raise ValueError('invalid adaptive fidelity multipliers')
        experiments.append(experiment)
        report['arms'][label] = dict(completion, checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    downloaded_frozen_sha256=frozen, downloaded_controller_sha256=controller)
    for field in FIELDS:
        if experiments[0][field] != experiments[1][field]:
            raise ValueError('arms differ: '+field)
    report['complete'] = True
    report['verifier_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
