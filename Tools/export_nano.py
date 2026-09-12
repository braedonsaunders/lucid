#!/usr/bin/env python3
"""Export and verify the production Nano ladder from a trusted local checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import coremltools as ct
import numpy as np
from PIL import Image
import torch

from architectures.nano_trunk import NanoTrunk
from convert_span import ImageRange, selective_fp16
from package_models import digest

ROOT = Path(__file__).resolve().parents[1]
SIZES = [(256,144),(320,180),(432,240),(480,270),(640,360),(864,480),(1280,720)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    actual = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if actual != args.sha256:
        raise ValueError('checkpoint hash mismatch')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if any(state.get(k) != v for k, v in {'trunk':'nano','channels':48,'blocks':6,'scale':2,'trunk_norm':'none'}.items()):
        raise ValueError('expected production ch48/b6, 2x, no-BatchNorm Nano checkpoint')
    model = NanoTrunk(channels=48, blocks=6, scale=2, frames=1, norm='none').eval()
    model.load_state_dict(state['model'])
    model.switch_to_deploy()
    torch.set_num_threads(4)
    entries = []
    with tempfile.TemporaryDirectory(prefix='lucid-export-', dir=ROOT / '.build') as temp:
        staged = Path(temp)
        for width,height in SIZES:
            name=f'lucidnano_{width}x{height}'
            package=staged/(name+'.mlpackage')
            with torch.no_grad():
                traced=torch.jit.trace(ImageRange(model).eval(),torch.zeros(1,3,height,width))
            exported=ct.convert(traced,
                inputs=[ct.ImageType(name='input',shape=(1,3,height,width),color_layout=ct.colorlayout.RGB,scale=1/255)],
                outputs=[ct.ImageType(name='output',color_layout=ct.colorlayout.RGB)],
                convert_to='mlprogram',compute_precision=selective_fp16(),minimum_deployment_target=ct.target.macOS15)
            exported.user_defined_metadata.update({'lucid.checkpoint_sha256':actual,'lucid.model':'r120 domain-trained Nano ch48/b6','lucid.scale':'2'})
            exported.save(str(package))
            compiled=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_AND_GPU)
            rng=np.random.default_rng(120)
            pixels=rng.integers(0,256,(height,width,3),dtype=np.uint8)
            sample=Image.fromarray(pixels)
            got=np.asarray(compiled.predict({'input':sample})['output'].convert('RGB')).astype(np.float32)
            with torch.no_grad():
                expected=ImageRange(model)(torch.from_numpy(pixels.astype(np.float32)/255).permute(2,0,1)[None])[0].permute(1,2,0).numpy()
            error=float(np.abs(got-expected).mean())
            if got.shape!=(height*2,width*2,3) or got.std()<5 or error>2:
                raise ValueError(f'{name}: failed geometry/level/numerical check (MAE {error})')
            for _ in range(3):compiled.predict({'input':sample})
            timings=[]
            for _ in range(5):
                start=time.perf_counter();compiled.predict({'input':sample});timings.append((time.perf_counter()-start)*1000)
            entries.append({'name':name,'width':width,'height':height,'scale':2,'referenceMilliseconds':round(float(np.median(timings)),2),'sha256':digest(package),'torchMeanAbsoluteError':round(error,4)})
            print(f'PASS {name}: mean error {error:.3f}/255',flush=True)
        # Publish only after the complete ladder has passed. Preserve existing assets.
        backup=ROOT/'.build/model-backups'/str(time.time_ns())
        backup.mkdir(parents=True)
        shutil.copy2(ROOT/'Lucid/Resources/Models.json',backup/'Models.json')
        for entry in entries:
            name=entry['name']+'.mlpackage';target=ROOT/'Model'/name
            if target.exists():target.rename(backup/name)
            shutil.copytree(staged/name,target)
        manifest={'schema':1,'family':'lucidnano','release':'1.0.0','checkpoint_sha256':actual,
                  'promoted':'r120 domain-trained Nano ch48/b6, 2x, no BatchNorm. Promoted by owner after live browser review on 2026-09-12. Frozen 24-frame browser holdout: LPIPS 9.28% and DISTS 12.02% better than r114 control; 8-frame lab holdout regresses 2.37% and 4.46%. Promotion is an explicit visual preference decision, not a clean automated gate pass.',
                  'models':entries}
        (ROOT/'Lucid/Resources/Models.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Published verified production ladder',actual,flush=True)


if __name__=='__main__':main()
