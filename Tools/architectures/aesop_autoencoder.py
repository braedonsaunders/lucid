"""Standalone AESOP autoencoder graph, extracted from the official implementation.

Source: 2minkyulee/AESOP-SR commit 3d6fe1d95a0a2fbaf2365861b0ce9f725985c498.
Based on AESOP (MinKyu Lee et al.) and BasicSR/ESRGAN (Xintao Wang et al.).
Licensed under Apache-2.0; see AESOP_LICENSE.txt.
Changes: retained only required graph definitions; replaced registry/framework
imports with Torch and standard logging. Graph operations and state keys retained.
This model is training-only and is never included in shipping SR assets.
"""
import logging
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F, init
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.checkpoint as checkpoint


def get_root_logger():
    return logging.getLogger(__name__)


def pixel_unshuffle(x, scale):
    return F.pixel_unshuffle(x, scale)

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32, use_checkpoint=False):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        if self.use_checkpoint:
            body_feat = checkpoint.checkpoint(self.body, feat)
            body_feat = self.conv_body(body_feat)
        else:
            body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

class AutoEncoder_RRDBNet(nn.Module):
    """
    AutoEncoder architecture for AESOP loss
    """

    def __init__(self, enc_opt, dec_opt):
        super().__init__()
        dec_opt = deepcopy(dec_opt)
        enc_opt = deepcopy(enc_opt)
        dec_opt.pop('type')
        self.decoder = RRDBNet(**dec_opt)
        self.conv_first = nn.Sequential(nn.Conv2d(dec_opt['num_in_ch'], dec_opt['num_feat'] // 16, 3, 1, 1), nn.Conv2d(dec_opt['num_feat'] // 16, dec_opt['num_feat'] // 16, 3, 1, 1))
        self.down = nn.Sequential(nn.PixelUnshuffle(2), nn.PixelUnshuffle(2))
        self.body = make_layer(RRDB, num_basic_block=2, num_feat=dec_opt['num_feat'], num_grow_ch=dec_opt['num_grow_ch'])
        self.conv_last = nn.Sequential(nn.Conv2d(dec_opt['num_feat'], dec_opt['num_feat'], 3, 1, 1), nn.Conv2d(dec_opt['num_feat'], dec_opt['num_in_ch'], 3, 1, 1))
        self.dec_is_frozen = False
        self.enc_is_frozen = False
        default_init_weights([self.conv_first, self.conv_last], 0.1)
        self.encoder = nn.Sequential(self.conv_first, self.down, self.body, self.conv_last)

    def freeze_encoder(self, current_iter='ITER_NOT_GIVEN'):
        if not self.enc_is_frozen:
            self.enc_is_frozen = True
            logger = get_root_logger()
            logger.info(f'Freeze encoder at {current_iter} iterations.')
            for param in self.encoder.parameters():
                param.requires_grad = False

    def freeze_decoder(self, current_iter='ITER_NOT_GIVEN'):
        if not self.dec_is_frozen:
            self.dec_is_frozen = True
            logger = get_root_logger()
            logger.info(f'Freeze decoder at {current_iter} iterations.')
            for param in self.decoder.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self, current_iter='ITER_NOT_GIVEN'):
        if self.enc_is_frozen:
            self.enc_is_frozen = False
            logger = get_root_logger()
            logger.info(f'Unfreeze encoder at {current_iter} iterations.')
            for param in self.encoder.parameters():
                param.requires_grad = True

    def unfreeze_decoder(self, current_iter='ITER_NOT_GIVEN'):
        if self.dec_is_frozen:
            self.dec_is_frozen = False
            logger = get_root_logger()
            logger.info(f'Unfreeze decoder at {current_iter} iterations.')
            for param in self.decoder.parameters():
                param.requires_grad = True

    def forward(self, x, return_bottleneck=False):
        bottleneck = self.encoder(x)
        x = self.decoder(bottleneck)
        if return_bottleneck:
            return (x, bottleneck)
        else:
            return x
