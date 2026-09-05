$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location 'C:\lucid\causal-capacity-20260904'
$jobs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($jobs) { throw 'Existing Python job found; preserve it' }
if ((Test-Path 'wide') -or (Test-Path 'control')) { throw 'Training output already exists' }
New-Item -ItemType File 'owner.lock' -ErrorAction Stop | Out-Null
try {
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
    $common = @('--architecture', 'causal_detail_v2', '--bank', 'C:\lucid\causal-stream-bank-20260904\bank', '--steps', '8000', '--blocks', '4', '--batch', '4', '--crop', '96', '--lr', '0.00002', '--seed', '20260912', '--device', 'cuda', '--precision', 'bf16', '--dino-weight', '0.03', '--dino-repository', 'C:\lucid\causal-frontier-20260904-dino\third_party\dinov2', '--dino-checkpoint', 'C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
    & 'C:\lucid\.venv\Scripts\python.exe' -u Tools/experiments/train_causal_detail.py @common --channels 64 --init widened-ch64.pth --out wide
    if ($LASTEXITCODE -ne 0) { throw 'Wider reconstruction training failed' }
    & 'C:\lucid\.venv\Scripts\python.exe' -u Tools/experiments/train_causal_detail.py @common --channels 32 --init 'C:\lucid\causal-frontier-20260904-dino\dino\step004000.pth' --out control
    if ($LASTEXITCODE -ne 0) { throw 'Matched capacity control failed' }
} finally { Remove-Item 'owner.lock' }
