$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\quality-breakthrough-20260908-r32'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe')}){throw 'Preserve existing Python jobs'}
$lock=[IO.File]::Open((Join-Path (Get-Location) 'owner.lock'),'CreateNew','ReadWrite','None')
function Run-Python($label,$arguments){
 $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
 @{exitCode=$p.ExitCode;finishedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content "$label-exit.json"
 if($p.ExitCode -ne 0){throw "$label failed"}
}
try {
 $env:PYTHONPATH='C:\lucid\quality-breakthrough-20260908-r32\Tools;C:\lucid\quality-breakthrough-20260908-r32\Tools\experiments;C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
 Run-Python 'recurrent-unit-tests' @('-m','unittest','discover','-s','Tools/experiments','-p','test_recurrent_frame_feeding.py')
 Run-Python 'recurrent-train' @('-u','Tools/experiments/train_recurrent_span.py','--bank','C:\lucid\stream-bank-v4','--init','C:\lucid\paired-ladder-20260905-r10\big_2k\step002000.pth','--out','recurrent_2k','--batch','4','--crop','96','--frames','3','--seed','20260914','--steps','2000','--native-input-stages')
} finally {$lock.Dispose()}
