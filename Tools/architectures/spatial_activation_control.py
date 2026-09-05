"""Experimental local conditioning for the frozen sparse activation controller.

Replaces global crop statistics with a fixed neighborhood. This is an engineering
hypothesis about crop/full-frame consistency, not a demonstrated quality gain.
"""
import torch
from torch.nn import functional as F
from .activation_control import ActivationControl


class SpatialActivationControl(ActivationControl):
    def __init__(self, anchor, selection, rms, dynamic=True, window=9):
        if window != 9:
            raise ValueError('the initial locality probe fixes a nine-feature neighborhood')
        super().__init__(anchor, selection, rms, dynamic)
        self.window = window

    def coefficients(self, features):
        condition = torch.cat((features, features.abs()), 1)
        condition = F.avg_pool2d(condition, self.window, stride=1,
                                 padding=self.window // 2, count_include_pad=False)
        if not self.dynamic:
            condition = torch.zeros_like(condition)
        values = self.controller(condition.permute(0, 2, 3, 1)).tanh()
        return values.permute(0, 3, 1, 2).reshape(features.shape[0], 6, 8,
                                                features.shape[2], features.shape[3])

    def forward(self, x):
        core = self.anchor.core
        features = core.conv_1(self.anchor.unshuffle(x))
        coefficients = self.coefficients(features)
        current = features
        first = None
        for i, block in enumerate((core.block_1, core.block_2, core.block_3,
                                   core.block_4, core.block_5, core.block_6)):
            current, auxiliary, _ = block(current)
            scale = 1 + .5 * (coefficients[:, i, :4].permute(0, 2, 3, 1) @ self.masks[i])
            shift = .2 * (coefficients[:, i, 4:].permute(0, 2, 3, 1) @ self.masks[i]) * self.shift_units[i]
            current = current * scale.permute(0, 3, 1, 2).to(current.dtype) + shift.permute(0, 3, 1, 2).to(current.dtype)
            if i == 0:
                first = current
        current = core.conv_2(current)
        return core.upsampler(core.conv_cat(torch.cat((features, current, first, auxiliary), 1)))


class CoarseSpatialActivationControl(SpatialActivationControl):
    """Fixed stride-two control grid; interpolation changes the learned function."""
    def coefficients(self, features):
        condition = F.avg_pool2d(torch.cat((features, features.abs()), 1), self.window,
                                 stride=2, padding=self.window // 2, count_include_pad=False)
        if not self.dynamic:
            condition = torch.zeros_like(condition)
        values = self.controller(condition.permute(0,2,3,1)).tanh().permute(0,3,1,2)
        values = F.interpolate(values, size=features.shape[-2:], mode='bilinear', align_corners=False)
        return values.reshape(features.shape[0],6,8,features.shape[2],features.shape[3])
