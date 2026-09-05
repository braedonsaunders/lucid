$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location 'C:\lucid\presented-eval-20260904'
$jobs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($jobs) { throw 'Existing Python job found; preserve it' }
if ((Test-Path 'evaluation.json') -or (Test-Path 'owner.lock')) { throw 'Preserve prior evaluation' }
New-Item -ItemType File 'owner.lock' -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH = 'C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
    $env:PATH = 'C:\ffmpeg\bin;' + $env:PATH
    $arguments = @('-u', 'Tools/frontier_eval/evaluate_sequences.py', '--device', 'cuda', '--manifest', 'sequences.json', '--checkpoint', 'blend80', 'blend-0.8.pth', '--checkpoint', 'shipping', 'C:\lucid\presented-detail-20260904\shipping.pth', '--present-4x-at-2x', 'shipping', '--report', 'evaluation.json')
    $job = Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput 'evaluation.log' -RedirectStandardError 'evaluation.err' -Wait -PassThru
    @{ exit_code = $job.ExitCode; finished_utc = (Get-Date).ToUniversalTime().ToString('o') } | ConvertTo-Json | Set-Content 'evaluation-result.json'
    if ($job.ExitCode -ne 0) { throw 'Evaluation failed' }
} finally { Remove-Item 'owner.lock' }
