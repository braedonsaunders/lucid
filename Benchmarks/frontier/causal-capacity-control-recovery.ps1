# Launched through Win32_Process.Create so an SSH disconnect cannot kill training.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location 'C:\lucid\causal-capacity-20260904'
$jobs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($jobs) { throw 'Existing Python job found; preserve it' }
if (Test-Path 'control-recovery') { throw 'Recovery output already exists' }
New-Item -ItemType File 'control-recovery.lock' -ErrorAction Stop | Out-Null
try {
    $arguments = @('-u', 'Tools/experiments/train_causal_detail.py', '--architecture', 'causal_detail_v2', '--bank', 'C:\lucid\causal-stream-bank-20260904\bank', '--steps', '8000', '--blocks', '4', '--batch', '4', '--crop', '96', '--lr', '0.00002', '--seed', '20260912', '--device', 'cuda', '--precision', 'bf16', '--dino-weight', '0.03', '--dino-repository', 'C:\lucid\causal-frontier-20260904-dino\third_party\dinov2', '--dino-checkpoint', 'C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth', '--channels', '32', '--init', 'C:\lucid\causal-frontier-20260904-dino\dino\step004000.pth', '--out', 'control-recovery')
    $job = Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput 'control-recovery.log' -RedirectStandardError 'control-recovery.err' -Wait -PassThru
    @{ exit_code = $job.ExitCode; finished_utc = (Get-Date).ToUniversalTime().ToString('o') } | ConvertTo-Json | Set-Content 'control-recovery-result.json'
} finally { Remove-Item 'control-recovery.lock' }
