#!/usr/bin/env python3
"""Record a warmed fixed-graph control before repeating a contended cost comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import coremltools as ct
import numpy as np
from PIL import Image

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--model',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
if args.out.exists():ap.error('fresh output required')
model=ct.models.MLModel(str(args.model),compute_units=ct.ComputeUnit.CPU_AND_GPU)
constraint=model.get_spec().description.input[0].type.imageType
rng=np.random.default_rng(20260904)
images=[Image.fromarray(rng.integers(0,256,(constraint.height,constraint.width,3),dtype=np.uint8)) for _ in range(4)]
samples=[]
for step in range(40):
    started=time.perf_counter()
    output=model.predict({'input':images[step%4]})['output']
    elapsed=(time.perf_counter()-started)*1000
    if step>=10:samples.append(elapsed)
report=dict(complete=True,mean_ms=float(np.mean(samples)),p95_ms=float(np.percentile(samples,95)),samples_ms=samples,
            model_files={str(p.relative_to(args.model)):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.model.rglob('*') if p.is_file()},
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Fixed baseline performance diagnostic; no candidate selection or quality admission')
args.out.write_text(json.dumps(report,indent=2)+'\n')
print({k:report[k] for k in ('mean_ms','p95_ms')})
