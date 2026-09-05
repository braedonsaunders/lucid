$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location 'C:\lucid\causal-frontier-20260904-dino'
function Assert-Idle {
    $jobs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
    if ($jobs) { throw 'Existing Python job found; preserve it and defer this experiment' }
}
Assert-Idle
foreach ($output in @('smoke', 'dino', 'control')) {
    if (Test-Path $output) { throw "Training output already exists: $output" }
}
$common = @('--architecture', 'causal_detail_v2', '--bank', 'C:\lucid\causal-frontier-20260904-v1\bank',
    '--init', 'C:\lucid\causal-frontier-20260904-v2\causal\step020000.pth',
    '--channels', '32', '--blocks', '4', '--batch', '4', '--crop', '96', '--lr', '0.00002',
    '--seed', '20260908', '--device', 'cuda', '--precision', 'bf16')
$teacher = @('--dino-weight', '0.03', '--dino-size', '224', '--dino-repository', 'third_party/dinov2',
    '--dino-checkpoint', 'C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
New-Item -ItemType File -Path 'owner.lock' -ErrorAction Stop | Out-Null
try {
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
    & 'C:\lucid\.venv\Scripts\python.exe' -u Tools/experiments/train_causal_detail.py @common @teacher --out smoke --steps 2
    if ($LASTEXITCODE -ne 0) { throw 'CUDA DINO smoke failed' }
    Assert-Idle
    & 'C:\lucid\.venv\Scripts\python.exe' -u Tools/experiments/train_causal_detail.py @common @teacher --out dino --steps 4000
    if ($LASTEXITCODE -ne 0) { throw 'DINO fine-tune failed' }
    Assert-Idle
    & 'C:\lucid\.venv\Scripts\python.exe' -u Tools/experiments/train_causal_detail.py @common --out control --steps 4000
    if ($LASTEXITCODE -ne 0) { throw 'Matched fine-tune control failed' }
} finally { Remove-Item 'owner.lock' }
