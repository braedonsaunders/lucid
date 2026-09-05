from pathlib import Path
import hashlib
import json
import subprocess

root = Path(__file__).resolve().parents[1]
out = root / '.build/tensor-output-small-ladder'
out.mkdir()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
report = {'complete': False, 'scope': 'Four deterministic RGB boundary probes per geometry; not source quality or browser delivery',
          'candidate': 'tensor4x_fp32', 'samples': 120, 'runs': [],
          'source_sha256': {p: sha(root / p) for p in [
              'Tools/experiments/export_tensor_output.py', 'Tools/experiments/TensorOutputBench.swift',
              'Lucid/Metal/CoreMLTensorImagePacker.swift', '.build/TensorOutputBenchLadder',
              '.build/run_tensor_ladder.py']}}
try:
    for width, height in [(256, 144), (320, 180), (432, 240), (480, 270)]:
        name = f'{width}x{height}'
        models = out / name
        with (out / (name + '-export.log')).open('w') as log:
            subprocess.run([str(root / '.venv-convert/bin/python'), 'Tools/experiments/export_tensor_output.py',
                '--checkpoint', 'Model/weights/span_ch32utc.pth', '--width', str(width), '--height', str(height),
                '--out', str(models)], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
        result = out / (name + '-native.json')
        with (out / (name + '-native.log')).open('w') as log:
            subprocess.run([str(root / '.build/TensorOutputBenchLadder'), str(models), str(result), '120',
                'tensor4x_fp32'], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
        report['runs'].append({'input': [width, height], 'export': str(models / 'export.json'),
            'export_sha256': sha(models / 'export.json'), 'native': str(result), 'native_sha256': sha(result)})
        print(name, 'complete', flush=True)
    report['complete'] = True
finally:
    (out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
