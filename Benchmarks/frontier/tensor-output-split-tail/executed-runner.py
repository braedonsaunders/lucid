from pathlib import Path
import hashlib, json, subprocess
root = Path(__file__).resolve().parents[1]
out = root / '.build/tensor-output-split-tail'
out.mkdir()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources = ['Tools/experiments/TensorOutputBench.swift', 'Lucid/Metal/CoreMLTensorImagePacker.swift',
           '.build/TensorOutputBenchSplitTail', '.build/run_tensor_split_tail.py']
receipt = {'complete': False, 'samples': 120, 'candidate': 'tensor4x_fp32',
           'source_sha256': {p: sha(root/p) for p in sources}, 'runs': []}
try:
    for width, height in [(256,144),(320,180),(432,240),(480,270),(640,360),(864,480)]:
        name = f'{width}x{height}'
        models = root / (f'.build/tensor-output-{height}' if height in [360,480]
                         else f'.build/tensor-output-small-ladder/{name}')
        export = json.loads((models/'export.json').read_text())
        for label in ['image4x','tensor4x_fp32']:
            for rel,h in export['models'][label].items(): assert sha(models/(label+'.mlpackage')/rel)==h
        result = out/(name+'-native.json')
        with (out/(name+'.log')).open('w') as log:
            subprocess.run([str(root/'.build/TensorOutputBenchSplitTail'),str(models),str(result),
                '120','tensor4x_fp32'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
        d=json.loads(result.read_text())
        assert d['complete'] and all(r['max_rgb']==0 and r['mean_rgb']==0 for r in d['accuracy']['tensor4x_fp32'])
        receipt['runs'].append({'input':[width,height], 'model_export_sha256':sha(models/'export.json'),
            'native_sha256':sha(result), 'native':str(result)})
        print(name, 'exact; timing complete',flush=True)
    receipt['complete']=True
finally:
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
