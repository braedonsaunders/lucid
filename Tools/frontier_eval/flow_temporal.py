"""Reference-defined motion-compensated temporal error with visibility masks.

DIS flow is an evaluation approximation, not ground-truth motion. Outputs never
influence motion or visibility. Report coverage and reference-warp error beside
reconstruction residuals, and retain spatial detail metrics to expose blur.
"""
import cv2
import numpy as np


def warp(image, backward):
    height, width = backward.shape[:2]
    y, x = np.mgrid[:height, :width].astype(np.float32)
    mx, my = x+backward[..., 0], y+backward[..., 1]
    visible = (mx >= 0) & (my >= 0) & (mx < width-1) & (my < height-1)
    result = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return result, visible


def correspondence(previous, current, forward, backward):
    warped_forward, inside = warp(forward, backward)
    disagreement = ((backward+warped_forward)**2).sum(axis=2)
    energy = (backward**2+warped_forward**2).sum(axis=2)
    prior, _ = warp(previous, backward)
    reference_delta = current-prior
    visible = inside & (disagreement <= 0.01*energy+0.5) & (np.abs(reference_delta) < 32)
    return {'backward': backward, 'visible': visible,
            'moving': visible & ((backward**2).sum(axis=2) >= 1),
            'reference_delta': reference_delta}


def score_pair(previous, current, context):
    prior, _ = warp(previous, context['backward'])
    residual = np.abs((current-prior)-context['reference_delta'])
    visible, moving = context['visible'], context['moving']
    return {'flow_residual_l1': float(residual[visible].mean()) if visible.any() else None,
            'flow_motion_residual_l1': float(residual[moving].mean()) if moving.any() else None,
            'flow_reference_residual_l1': float(np.abs(context['reference_delta'])[visible].mean()) if visible.any() else None,
            'flow_nonoccluded_fraction': float(visible.mean()),
            'flow_motion_fraction': float(moving.mean())}


class ReferenceMotion:
    def __init__(self, references, max_width=640):
        if len(references) < 2 or max_width < 64:
            raise ValueError('two reference frames and flow width >=64 required')
        cv2.setNumThreads(2)
        cv2.ocl.setUseOpenCL(False)
        self.gray = [np.asarray(frame.convert('L'), dtype=np.float32) for frame in references]
        height, width = self.gray[0].shape
        if any(frame.shape != (height, width) for frame in self.gray):
            raise ValueError('reference size changes within sequence')
        ratio = min(1, max_width/width)
        fw, fh = max(16, round(width*ratio)), max(16, round(height*ratio))
        small = [cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA).round().astype(np.uint8) for frame in self.gray]
        estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.contexts = []
        for index in range(1, len(references)):
            forward = estimator.calc(small[index-1], small[index], None)
            backward = estimator.calc(small[index], small[index-1], None)
            def full(flow):
                result = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
                result[..., 0] *= width/fw
                result[..., 1] *= height/fh
                return result
            self.contexts.append(correspondence(self.gray[index-1], self.gray[index], full(forward), full(backward)))
        self.metadata = {'estimator': 'OpenCV DIS PRESET_MEDIUM, reference frames only',
            'opencv': cv2.__version__, 'flow_size': [fw, fh], 'scoring_size': [width, height],
            'visibility': 'in bounds; forward/backward squared disagreement <=0.01*flow energy+0.5; reference warp difference <32/255',
            'motion_threshold_pixels': 1.0, 'aggregation': 'equal mean of valid frame-pair means; coverage reported separately',
            'limitation': 'estimated flow and occlusion; low coverage or high reference warp error weakens evidence'}

    def score(self, outputs):
        gray = [np.asarray(frame.convert('L'), dtype=np.float32) for frame in outputs]
        if len(gray) != len(self.gray) or any(a.shape != b.shape for a, b in zip(gray, self.gray)):
            raise ValueError('output/reference sequence mismatch')
        rows = [score_pair(gray[i], gray[i+1], context) for i, context in enumerate(self.contexts)]
        result = {key: float(np.mean([row[key] for row in rows if row[key] is not None]))
                  if any(row[key] is not None for row in rows) else None for key in rows[0]}
        result['flow_valid_pair_fraction'] = sum(row['flow_residual_l1'] is not None for row in rows)/len(rows)
        return result
