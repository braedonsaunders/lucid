$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\quality-breakthrough-20260908-r34'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe')}){throw 'Preserve existing Python jobs'}
$lock=[IO.File]::Open((Join-Path (Get-Location) 'owner.lock'),'CreateNew','ReadWrite','None')
function Run-Python($label,$arguments){
 $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
 @{exitCode=$p.ExitCode;finishedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content "$label-exit.json"
 if($p.ExitCode -ne 0){throw "$label failed"}
}
try {
 $env:PYTHONPATH='C:\lucid\quality-breakthrough-20260908-r34\Tools;C:\lucid\quality-breakthrough-20260908-r34\Tools\experiments;C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
 Run-Python 'confidence-unit-tests' @('-m','unittest','discover','-s','Tools/experiments','-p','test_confidence_stages.py')
 Run-Python 'confidence-train' @('-u','Tools/experiments/train_confidence_head.py','--bank','C:\lucid\stream-bank-v4','--init','C:\lucid\paired-ladder-20260905-r10\big_2k\step002000.pth','--out','confidence_2k','--steps','2000','--batch','8','--crop','96','--seed','20260914')
} finally {$lock.Dispose()}
