
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location 'C:\lucid\causal-frontier-20260904-v1'
$jobs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($jobs) { $jobs | Select-Object ProcessId, Name, CommandLine | Format-List; throw 'Existing Python job found; preserve it' }
if ((Test-Path 'causal') -or (Test-Path 'control')) { throw 'Training output already exists' }
New-Item -ItemType File -Path 'owner.lock' -ErrorAction Stop | Out-Null
try {
 nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
 & 'C:\lucid\.venv\Scripts\python.exe' -u Tools/experiments/train_causal_detail.py --bank bank --out causal --steps 20000 --channels 32 --blocks 4 --batch 4 --crop 96 --device cuda --precision bf16
 if ($LASTEXITCODE -ne 0) { throw 'Causal training failed' }
 $jobs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
 if ($jobs) { throw 'Another Python job appeared; defer matched control' }
 & 'C:\lucid\.venv\Scripts\python.exe' -u Tools/experiments/train_causal_detail.py --bank bank --out control --steps 20000 --channels 32 --blocks 4 --batch 4 --crop 96 --device cuda --precision bf16 --no-history
 if ($LASTEXITCODE -ne 0) { throw 'Control training failed' }
} finally { Remove-Item 'owner.lock' }
