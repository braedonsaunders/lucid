"""Frozen decoded-autoencoder L1, using the published AESOP reference weights."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from architectures.aesop_autoencoder import AutoEncoder_RRDBNet
from train_causal_detail import digest

PUBLISHED_SHA256 = '8a428a920a9e9d66362bf5da2d837e7ccabdbe2d6fb04d0a1b84b7fea9fd83c1'


class AutoencodedFidelity(nn.Module):
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.autoencoder.eval()
        return self

    def forward(self, prediction, reference):
        if prediction.shape != reference.shape or prediction.ndim != 4 or any(s % 4 for s in prediction.shape[-2:]):
            raise ValueError('matching BCHW images with geometry divisible by four required')
        # Preserve the reference graph in FP32, including gradients into SR.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            with torch.no_grad():
                target = self.autoencoder(reference.detach().float())
            decoded = self.autoencoder(prediction.float())
            if decoded.shape != prediction.shape or target.shape != reference.shape:
                raise ValueError('AESOP must compare decoded images, not bottleneck features')
            return F.l1_loss(decoded, target)


def load_aesop_loss(path, device):
    path = Path(path)
    if digest(path) != PUBLISHED_SHA256:
        raise ValueError('AESOP reference checkpoint does not match its published-source receipt')
    # Added teacher initialization must not alter the matched discriminator RNG.
    with torch.random.fork_rng(devices=[]):
        autoencoder = AutoEncoder_RRDBNet({}, dict(type='RRDBNet', num_in_ch=3,
            num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32))
        state = torch.load(path, map_location='cpu', weights_only=True)
        autoencoder.load_state_dict(state['params_ema'], strict=True)
    loss = AutoencodedFidelity(autoencoder).to(device)
    loss.metadata = {'checkpoint_sha256': PUBLISHED_SHA256, 'weights': 'params_ema',
        'upstream_commit': '3d6fe1d95a0a2fbaf2365861b0ce9f725985c498',
        'loss': 'L1 after frozen autoencoder decoder, FP32; target branch detached',
        'architecture': 'official 4x bottleneck, two encoder RRDBs and 23 decoder RRDBs',
        'scope': 'published synthetic AE trained 100k steps; not a locally pretrained 2x codec-domain AE',
        'inference_cost': 'none; teacher is not stored in the SR model',
        'source_sha256': digest(Path(__file__).resolve().parents[1] / 'architectures/aesop_autoencoder.py')}
    return loss
