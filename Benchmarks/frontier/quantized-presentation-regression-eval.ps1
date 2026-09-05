$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location 'C:\lucid\quantized-presentation-regression-20260905'
$jobs = Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe')}
if ($jobs) { throw 'Existing Python job found; preserve it' }
if ((Test-Path 'evaluation.json') -or (Test-Path 'owner.lock')) { throw 'Preserve prior evaluation' }
New-Item -ItemType File 'owner.lock' -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH = 'C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
    $arguments = @('-u', 'Tools/frontier_eval/score_native_holdout.py', '--development-presentation', '--device', 'cuda', '--frames', 'frames', '--references', 'C:\lucid\quality-holdout-frozen-20260905\frames', '--report', 'evaluation.json')
    $job = Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput 'evaluation.log' -RedirectStandardError 'evaluation.err' -Wait -PassThru
    @{ exit_code = $job.ExitCode; finished_utc = (Get-Date).ToUniversalTime().ToString('o') } | ConvertTo-Json | Set-Content 'evaluation-result.json'
    if ($job.ExitCode -ne 0) { throw 'Evaluation failed' }
} finally { Remove-Item 'owner.lock' }
