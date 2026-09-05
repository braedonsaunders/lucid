import argparse
import hashlib
from pathlib import Path
import json
import numpy as np
import coremltools as ct
from PIL import Image
ap=argparse.ArgumentParser(description='Audit the observed GPU image-output conversion against tensor values, without source-quality selection')
ap.add_argument('--models',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
if args.out.exists():ap.error('fresh report required')
root=args.models
image=ct.models.MLModel(str(root/'image4x.mlpackage'),compute_units=ct.ComputeUnit.CPU_AND_GPU)
tensor=ct.models.MLModel(str(root/'tensor4x_fp32.mlpackage'),compute_units=ct.ComputeUnit.CPU_AND_GPU)
rng=np.random.default_rng(20260905)
inputs={'random':rng.integers(0,256,size=(360,640,3),dtype=np.uint8),
        'ramp':np.broadcast_to((np.arange(640)%256)[None,:,None],(360,640,3)).astype(np.uint8)}
report={}
for name,pixels in inputs.items():
 x=Image.fromarray(pixels)
 expected=np.array(image.predict({'input':x})['output'].convert('RGB')).transpose(2,0,1)
 values=tensor.predict({'input':x})['output'][0]
 if values.shape!=expected.shape:raise ValueError((values.shape,expected.shape))
 nearest=np.floor(values+.5).astype(np.uint8)
 mask=nearest!=expected
 report[name]={'dtype':str(values.dtype),'saturated_fraction':float(((values<=0)|(values>=255)).mean()),
    'nearest_mismatches':int(mask.sum()),'floor_mismatches':int((np.floor(values).astype(np.uint8)!=expected).sum()),
    'binary16_then_nearest_even_mismatches':int((np.rint(values.astype(np.float16).astype(np.float32)).astype(np.uint8)!=expected).sum()),
    'tensor_values_and_image_at_first_mismatches':[[float(values[tuple(i)]),int(expected[tuple(i)])] for i in np.argwhere(mask)[:24]],
    'signed_nearest_minus_image_mean':float((nearest.astype(np.int16)-expected.astype(np.int16)).mean())}
report['_provenance']={'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'export_sha256':hashlib.sha256((root/'export.json').read_bytes()).hexdigest(),'numpy':np.__version__,'coremltools':ct.__version__,'compute_units':'CPU_AND_GPU','scope':'Observed native image conversion on this host; not a universal documented conversion guarantee'}
args.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
