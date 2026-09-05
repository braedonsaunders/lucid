$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location 'C:\lucid\pixrestore-baseline-20260904'
$jobs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($jobs) { throw 'Existing Python job found; preserve it' }
if (Test-Path 'outputs-tiles') { throw 'Baseline output already exists' }
if (Test-Path 'input-tiles') { throw 'Tile input already exists' }
tar -xf tiles.tar
if ($LASTEXITCODE -ne 0) { throw 'Tile extraction failed' }
New-Item -ItemType File 'tiles-owner.lock' -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH = 'C:\lucid\pixrestore-baseline-20260904\deps'
    # Windows lacks this release's Triton compiler. Keep CUDA/BF16 execution;
    # disable only torch.compile. These timings are not the paper's benchmark.
    $env:TORCHDYNAMO_DISABLE = '1'
    $arguments = @('-u', 'inference.py', '--checkpoint', 'C:\lucid\frontier-teachers-20260904\pixrestore-s', '--input', 'input-tiles', '--output-dir', 'outputs-tiles', '--infer-steps', '1', '--cfg-scale', '1', '--seed', '0', '--upscale', '1', '--test-mode', 'off', '--dinov2-repository', 'C:\lucid\causal-frontier-20260904-dino\third_party\dinov2', '--dinov2-checkpoint', 'C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
    $job = Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput 'tiles-inference.log' -RedirectStandardError 'tiles-inference.err' -Wait -PassThru
    @{ exit_code = $job.ExitCode; finished_utc = (Get-Date).ToUniversalTime().ToString('o') } | ConvertTo-Json | Set-Content 'tiles-result.json'
    if ($job.ExitCode -ne 0) { throw 'Official PixRestore inference failed' }
} finally { Remove-Item 'tiles-owner.lock' }
