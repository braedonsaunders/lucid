$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\learned-past-information-20260905-r1'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe')}){throw 'Preserve existing Python jobs'}
$lock=[IO.File]::Open((Join-Path (Get-Location) 'owner.lock'),'CreateNew','ReadWrite','None')
try {
 $env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
 $env:PYTHONPATH='C:\lucid\past-information-20260905-r2\deps;C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
 $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList @('-m','unittest','discover','-s','Tools/experiments','-p','test_*past_information.py','-v') -RedirectStandardOutput 'tests.log' -RedirectStandardError 'tests.err' -Wait -PassThru
 if($p.ExitCode -ne 0){throw 'Probe math/alignment tests failed'}
 $arguments=@('-u','Tools/experiments/probe_learned_past_information.py','--checkpoint','C:\lucid\presented-detail-20260904\shipping.pth','--device','cuda','--bank','C:\lucid\presented-diversity-20260905\data\bank','--shipping-cache','C:\lucid\presented-diversity-20260905\shipping-cache','--out','result')
 $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput 'probe.log' -RedirectStandardError 'probe.err' -Wait -PassThru
 @{exitCode=$p.ExitCode;finishedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content 'probe-exit.json'
 if($p.ExitCode -ne 0){throw 'Past-information probe failed'}
} finally {$lock.Dispose()}
