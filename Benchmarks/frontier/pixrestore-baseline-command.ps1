$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location 'C:\lucid\pixrestore-baseline-20260904'
$jobs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($jobs) { throw 'Existing Python job found; preserve it' }
if (Test-Path 'outputs-eager') { throw 'Baseline output already exists' }
New-Item -ItemType File 'owner.lock' -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH = 'C:\lucid\pixrestore-baseline-20260904\deps'
    # Windows lacks this release's Triton compiler. Keep CUDA/BF16 execution;
    # disable only torch.compile. These timings are not the paper's benchmark.
    $env:TORCHDYNAMO_DISABLE = '1'
    & 'C:\lucid\.venv\Scripts\python.exe' -u inference.py --checkpoint 'C:\lucid\frontier-teachers-20260904\pixrestore-s' --input input --output-dir outputs-eager --infer-steps 1 --cfg-scale 1 --seed 0 --upscale 2 --test-mode off --dinov2-repository 'C:\lucid\causal-frontier-20260904-dino\third_party\dinov2' --dinov2-checkpoint 'C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth'
    if ($LASTEXITCODE -ne 0) { throw 'Official PixRestore inference failed' }
} finally { Remove-Item 'owner.lock' }
