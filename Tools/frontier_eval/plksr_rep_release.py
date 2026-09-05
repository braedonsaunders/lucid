"""Load reviewed PLKSR-Rep release code/weights, without importing its trainer."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F

PINNED = {
    'plksr/archs/plksr_arch.py': 'e465706c387207971ccb01bb1aa151112817299725acd59c42fe45560db0b79c',
    'plksr/archs/plksr_rep_arch.py': 'e52c6af7bacc8968ea54ac63c1064dda2665d56586924b4d36b7f886d942081e',
    'scripts/inference_ntire.py': '056f15f40727aa59c4c5a6373ff8775a6d2879a528787fb044bb563ac6ba297b',
    'LICENSE': '6a285070bc9a9048e4a275cdb343fda8cfd92b05bfe65307609c26a5ff6870b8',
}
WEIGHTS = '2990afbf1460cf4b6cb6b34a5c2994ec6f4d3a64328565c87bd2fa85b23484af'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PublishedPadding(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # The release's top-level test.py defaults to 16 source pixels of reflect padding.
        return self.model(F.pad(x, (16, 16, 16, 16), mode='reflect'))[:, :, 64:-64, 64:-64]


def load_release(repository, weights, deploy=False):
    repository = Path(repository)
    for name, expected in PINNED.items():
        if digest(repository/name) != expected:
            raise ValueError(f'changed release source: {name}')
    if digest(weights) != WEIGHTS:
        raise ValueError('changed release checkpoint')
    names = ['basicsr', 'basicsr.utils', 'basicsr.utils.registry', 'basicsr.archs',
             'basicsr.archs.arch_util', 'lucid_plksr_release',
             'lucid_plksr_release.plksr_arch', 'lucid_plksr_release.plksr_rep_arch']
    previous = {name: sys.modules.get(name) for name in names}
    class Registry:
        def register(self):
            return lambda cls: cls
    try:
        for name in names[:6]:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        sys.modules['basicsr.utils.registry'].ARCH_REGISTRY = Registry()
        # Only an initializer shim; every parameter/buffer is replaced by strict loading.
        sys.modules['basicsr.archs.arch_util'].trunc_normal_ = nn.init.trunc_normal_
        modules = []
        for name, file in [(names[-2], 'plksr_arch.py'), (names[-1], 'plksr_rep_arch.py')]:
            spec = importlib.util.spec_from_file_location(name, repository/'plksr/archs'/file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules.append(module)
        model = modules[1].PLKSR_Rep(dim=64, n_blocks=12, upscaling_factor=4,
            ccm_type='DCCM', kernel_size=17, split_ratio=.25, use_ea=True).eval()
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        parameters = checkpoint.get('params_ema', checkpoint.get('params', checkpoint))
        model.load_state_dict(parameters, strict=True)
        if deploy:
            for module in list(model.modules()):
                if isinstance(module, modules[1].RepConv):
                    module.switch_to_deploy()
            # Author-supplied equivalent slicing form; with_idt is false in this release.
            modules[0].convert_plk_forward_for_coreml(model)
        return PublishedPadding(model).eval()
    finally:
        for name in names:
            if previous[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]
