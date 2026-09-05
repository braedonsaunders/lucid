$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location 'C:\lucid\causal-frontier-20260904-v2'
$jobs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($jobs) { throw 'Existing Python job found; preserve it and defer this experiment' }
if (Test-Path 'causal') { throw 'Training output already exists' }
New-Item -ItemType File -Path 'owner.lock' -ErrorAction Stop | Out-Null
try {
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
    & 'C:\lucid\.venv\Scripts\python.exe' -u Tools/experiments/train_causal_detail.py --architecture causal_detail_v2 --bank 'C:\lucid\causal-frontier-20260904-v1\bank' --out causal --steps 20000 --channels 32 --blocks 4 --batch 4 --crop 96 --device cuda --precision bf16
    if ($LASTEXITCODE -ne 0) { throw 'V2 causal training failed' }
} finally { Remove-Item 'owner.lock' }
