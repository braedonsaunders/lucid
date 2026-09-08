$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'D:\lucid-quality-breakthrough-r37'
$lock=[IO.File]::Open((Join-Path (Get-Location) 'bank-owner.lock'),'CreateNew','ReadWrite','None')
function Run-Python($label,$arguments){
 $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
 @{exitCode=$p.ExitCode;finishedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content "$label-exit.json"
 if($p.ExitCode -ne 0){throw "$label failed"}
}
try {
 if((Get-PSDrive D).Free -lt 100GB){throw '100 GiB expansion headroom required'}
 if(!(Test-Path reds-masters\master-receipts.json)){throw 'verified lossless masters required'}
 $env:PATH='C:\ffmpeg\bin;'+$env:PATH
 Run-Python 'reds-bank' @('-u','Tools/experiments/build_large_stream_bank.py','--sources','reds-masters/sources.json','--out','reds-bank','--windows-per-source','18','--patches-per-window','10','--seed','20260918')
 Run-Python 'expanded-mapped' @('-u','Tools/experiments/mmap_training_bank.py','--bank','C:\lucid\stream-bank-v4','--bank','reds-bank','--out','expanded-mapped')
} finally {$lock.Dispose()}
