$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\activation-control-20260905'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'}){throw 'Preserve existing Python jobs'}
foreach($p in @('owner.lock','probe.json','result.json')){if(Test-Path $p){throw "Preserve existing $p"}}
New-Item -ItemType File owner.lock -ErrorAction Stop | Out-Null
try {
 $env:PYTHONPATH='C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
 $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList @('-u','Tools/experiments/probe_activation_control.py','--bank','C:\lucid\presented-diversity-20260905\data\bank','--init','C:\lucid\presented-detail-20260904\shipping.pth','--out','probe.json','--device','cuda') -RedirectStandardOutput 'probe.log' -RedirectStandardError 'probe.err' -Wait -PassThru
 @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content result.json
 if($job.ExitCode -ne 0){throw 'Activation probe failed'}
} finally {Remove-Item owner.lock}
