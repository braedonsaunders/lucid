#!/usr/bin/env python3
"""Measures overlay alignment and sharpness gain.

Captures the same screen rect twice (enhancement on, then off), then finds the
vertical/horizontal shift that best matches the two images and reports the
high-frequency energy ratio. Offset 0 means the enhanced pixels land exactly on
the video box; a positive ratio means the enhanced image carries more detail.
"""
import json, subprocess, sys, time
import numpy as np
from PIL import Image

def control(payload):
    subprocess.run(["node", sys.argv[0].replace("align_check.py", "control.js"), json.dumps(payload)],
                   capture_output=True, text=True, timeout=10)

def shot(path, rect):
    subprocess.run(["screencapture", "-x", "-R", rect, path], check=True)
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)

def high_freq(a):
    return float(np.mean(np.abs(np.diff(a, axis=0))) + np.mean(np.abs(np.diff(a, axis=1))))

def best_shift(a, b, span=48):
    # b shifted by dy/dx to match a, scored on the overlapping interior
    h, w = a.shape
    best = (1e9, 0, 0)
    for dy in range(-span, span + 1, 2):
        for dx in (-2, 0, 2):
            ay0, by0 = max(0, dy), max(0, -dy)
            ax0, bx0 = max(0, dx), max(0, -dx)
            hh = min(h - ay0, h - by0); ww = min(w - ax0, w - bx0)
            if hh < h // 2 or ww < w // 2: continue
            d = float(np.mean(np.abs(a[ay0:ay0+hh, ax0:ax0+ww] - b[by0:by0+hh, bx0:bx0+ww])))
            if d < best[0]: best = (d, dy, dx)
    return best

rect = sys.argv[1]                       # "x,y,w,h" of the video box on screen
control({"enabled": True});  time.sleep(2.0); on  = shot("/tmp/lucid-on.png", rect)
control({"enabled": False}); time.sleep(1.5); off = shot("/tmp/lucid-off.png", rect)
control({"enabled": True})
diff, dy, dx = best_shift(off, on)
print(f"best match offset: dy={dy} dx={dx} px (0,0 = aligned), residual={diff:.2f}")
print(f"high-frequency energy: off={high_freq(off):.3f}  on={high_freq(on):.3f}  ratio={high_freq(on)/max(high_freq(off),1e-6):.3f}")
