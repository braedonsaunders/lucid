$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# Reuse exactly the trainer used by the completed DINO/no-DINO pair.
Set-Location 'C:\lucid\causal-frontier-20260904-dino'
$jobs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($jobs) { throw 'Existing Python job found; preserve it' }
if (Test-Path 'stream-data') { throw 'Data experiment output already exists' }
New-Item -ItemType File -Path 'owner.lock' -ErrorAction Stop | Out-Null
try {
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
    & 'C:\lucid\.venv\Scripts\python.exe' -u Tools/experiments/train_causal_detail.py --architecture causal_detail_v2 --bank 'C:\lucid\causal-stream-bank-20260904\bank' --init 'C:\lucid\causal-frontier-20260904-v2\causal\step020000.pth' --out stream-data --steps 4000 --channels 32 --blocks 4 --batch 4 --crop 96 --lr 0.00002 --seed 20260908 --device cuda --precision bf16
    if ($LASTEXITCODE -ne 0) { throw 'Full-frame codec data experiment failed' }
} finally { Remove-Item 'owner.lock' }
