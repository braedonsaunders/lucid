#!/usr/bin/env python3
"""Matched dynamic/static activation-controller experiment; frozen reconstruction."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from train_causal_detail import batch, digest, load_bank
from teacher_distillation import load_teacher_cache
from train_presented_detail import reconstruction_objective, state_digest
from fold_shipping_head import fold_head
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_checkpoint import load
from architectures.subspace_adapter import fuse_convolutions
from architectures.activation_control import ActivationControl, config_from_probe


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('bank', 'shipping-cache', 'pixrestore-cache', 'init', 'probe', 'out',
                'pixrestore-repository', 'dino-repository', 'dino-checkpoint'):
        ap.add_argument('--'+key, type=Path, required=True)
    ap.add_argument('--mode', choices=('static', 'dynamic'), required=True)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--seed', type=int, default=20260914)
    ap.add_argument('--local-fidelity-constraint', action='store_true')
    ap.add_argument('--deterministic', action='store_true')
    args = ap.parse_args()
    if args.out.exists() or args.steps < 1:
        ap.error('fresh output and positive steps required')
    torch.set_num_threads(4)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    probe = json.loads(args.probe.read_text())
    if probe['bank_sha256'] != digest(args.bank/'manifest.json') or probe['shipping_sha256'] != digest(args.init):
        raise ValueError('probe and training identities differ')
    _, data = load_bank(args.bank)
    bank_hash = digest(args.bank/'manifest.json')
    shipping, shipping_receipt = load_teacher_cache(args.shipping_cache, bank_hash, data, digest)
    teacher, _ = load_teacher_cache(args.pixrestore_cache, bank_hash, data, digest)
    if shipping_receipt['checkpoint_sha256'] != digest(args.init):
        raise ValueError('cached shipping identity changed')
    mixed = {identity: np.rint(.5*pixels.astype(np.float32)+.5*teacher[identity].astype(np.float32)).astype(np.uint8)
             for identity, pixels in shipping.items()}
    if not args.local_fidelity_constraint:
        del shipping
    del teacher
    anchor, _, frames = load(args.init, 'cuda')
    if frames != 1:
        raise ValueError('single-frame initialization required')
    anchor = fold_head(anchor)
    fuse_convolutions(anchor)
    config = config_from_probe(probe, args.mode == 'dynamic')
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model = ActivationControl(anchor, **config).cuda().train()
    del anchor
    frozen = lambda: state_digest({k: v for k, v in model.state_dict().items() if not k.startswith('controller.')})
    frozen_hash = frozen()
    optimizer = torch.optim.AdamW(model.controller.parameters(), lr=.0002, weight_decay=0)
    from dino_adversary import DinoAdversary
    adversary = DinoAdversary(args.pixrestore_repository, args.dino_repository, args.dino_checkpoint)
    penalty = None
    if args.local_fidelity_constraint:
        from local_fidelity_constraint import violations, AdaptiveFidelityPenalty
        penalty = AdaptiveFidelityPenalty('cuda')
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True)
    experiment = {
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'bank_sha256': bank_hash, 'checkpoint_sha256': digest(args.init),
        'teacher_manifest_sha256': digest(args.pixrestore_cache/'manifest.json'),
        'shipping_manifest_sha256': digest(args.shipping_cache/'manifest.json'),
        'probe_sha256': digest(args.probe), 'frozen_anchor_and_masks_sha256': frozen_hash,
        'controller_initial_sha256': state_digest(model.controller.state_dict()),
        'discriminator_initial_sha256': state_digest(adversary.discriminator.state_dict()),
        'controller_parameters': sum(p.numel() for p in model.controller.parameters()),
        'controller_config': config, 'adversary': adversary.metadata,
        'recipe': {'batch': 4, 'crop': 96, 'lr': .0002, 'gan_weight': .005, 'teacher_mix': .5,
                   'detail_target': 'reference', 'border_excluded': 8, 'temporal_loss': False},
        'limitation': 'pooled conditioning is learned on crops; full-frame generalization must be measured',
        'precision': 'CUDA BF16 autocast, FP32 AdamW', 'torch': str(torch.__version__),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'gpu': torch.cuda.get_device_name(), 'purpose': 'matched experimental controllers; no promotion',
        'source_hashes': {str(p): digest(p) for p in [Path(__file__),
            Path(__file__).with_name('train_presented_detail.py'), Path(__file__).with_name('dino_adversary.py'),
            Path(__file__).resolve().parents[1]/'architectures/activation_control.py']}}
    experiment_path = args.out/'experiment.json'
    if penalty is not None:
        experiment['local_fidelity_constraint'] = {
            'domains': ['RGB squared error', 'signed RGB Sobel squared error'],
            'baseline': 'aligned cached full-context shipping RGB8 presentation',
            'tiles': 8, 'energy_floor': 1e-4, 'initial_multipliers': [1., 1.],
            'multiplier_rate': penalty.rate, 'quadratic_rho': penalty.rho,
            'scope': 'training data only; no inference operations or fidelity guarantee',
            'code_sha256': digest(Path(__file__).with_name('local_fidelity_constraint.py'))}
    if args.deterministic:
        experiment['deterministic_resize_sha256'] = digest(Path(__file__).with_name('deterministic_resize.py'))
        experiment['dino_supervision_sha256'] = digest(Path(__file__).with_name('dino_supervision.py'))
    experiment_path.write_text(json.dumps(experiment, indent=2)+'\n')
    started = time.monotonic()
    parameters = list(model.controller.parameters())
    for step in range(1, args.steps+1):
        baseline_rng = copy.deepcopy(rng) if penalty is not None else None
        x, reference, intended = (v[:, 0].cuda() for v in batch(data['train'], rng, 4, 1, 96, mixed))
        if penalty is not None:
            bx, br, baseline = (v[:, 0].cuda() for v in batch(data['train'], baseline_rng, 4, 1, 96, shipping))
            if not torch.equal(bx, x) or not torch.equal(br, reference):
                raise ValueError('cached shipping crop or augmentation differs')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(x)
        if step == 1:
            experiment['first_batch_sha256'] = state_digest({'source': x, 'reference': reference, 'intended': intended})
            experiment['first_output_sha256'] = state_digest({'output': output})
            if penalty is not None:
                experiment['first_fidelity_baseline_sha256'] = state_digest({'baseline': baseline})
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                if not torch.equal(output, model.anchor(x)):
                    raise ValueError('initial controller changes BF16 anchor output')
            experiment_path.write_text(json.dumps(experiment, indent=2)+'\n')
        output, reference, intended = (v.float()[:, :, 8:-8, 8:-8] for v in (output, reference, intended))
        reconstruction = reconstruction_objective(output, reference, intended, 'reference')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            gan, fake = adversary.generator_loss(output)
        loss = reconstruction+.005*gan
        if penalty is not None:
            fidelity_values = violations(output, reference, baseline.float()[:, :, 8:-8, 8:-8])
            loss = loss+penalty.loss(fidelity_values)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
        optimizer.step()
        if penalty is not None:
            penalty.update(fidelity_values)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            discriminator_loss = adversary.update(reference, fake)
        for group in optimizer.param_groups:
            group['lr'] = .0002*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        if step % 200 == 0:
            print(f'step {step}/{args.steps} loss={float(loss.detach()):.6f} controller_grad={float(norm):.6f} '
                  f'discriminator={discriminator_loss:.5f} minutes={(time.monotonic()-started)/60:.2f}', flush=True)
            if penalty is not None:
                print(f'  fidelity_violation={fidelity_values.detach().tolist()} '
                      f'multipliers={penalty.multipliers.tolist()}', flush=True)
        if step % 2000 == 0 or step == args.steps:
            if frozen() != frozen_hash:
                raise ValueError('frozen reconstruction or control mask changed')
            torch.save({'model': model.state_dict(), 'channels': 32, 'scale': 2, 'frames': 1,
                'version': model.version, 'step': step, 'architecture': 'activation_control2x',
                'controller_config': config, 'experiment': experiment}, args.out/f'step{step:06d}.pth')
            torch.save({'optimizer': optimizer.state_dict(), 'discriminator': adversary.discriminator.state_dict(),
                'discriminator_optimizer': adversary.optimizer.state_dict(), 'step': step,
                'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(),
                'fidelity_multipliers': penalty.multipliers if penalty is not None else None}, args.out/f'training{step:06d}.pth')
    (args.out/'complete.json').write_text(json.dumps({'steps': args.steps,
        'minutes': (time.monotonic()-started)/60, 'frozen_anchor_unchanged': frozen() == frozen_hash,
        'controller_final_sha256': state_digest(model.controller.state_dict()),
        'fidelity_multipliers': penalty.multipliers.tolist() if penalty is not None else None})+'\n')


if __name__ == '__main__':
    main()
