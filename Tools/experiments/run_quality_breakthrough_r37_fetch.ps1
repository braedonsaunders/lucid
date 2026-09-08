$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'D:\lucid-quality-breakthrough-r37'
$lock=[IO.File]::Open((Join-Path (Get-Location) 'fetch-owner.lock'),'CreateNew','ReadWrite','None')
function Run-Python($label,$arguments){
 $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
 @{exitCode=$p.ExitCode;finishedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content "$label-exit.json"
 if($p.ExitCode -ne 0){throw "$label failed"}
}
try {
 if((Get-PSDrive D).Free -lt 100GB){throw '100 GiB corpus headroom required'}
 $env:PATH='C:\ffmpeg\bin;'+$env:PATH
 # CPU/network-only corpus work; may run alongside the owned r36 GPU experiment.
 Run-Python 'reds-fetch' @('-u','Tools/experiments/fetch_reds_subset.py','--index','reds-archive-index.json','--out','reds-frames','--sequences','60','--frames','100','--workers','4')
 Run-Python 'reds-masters' @('-u','Tools/experiments/materialize_reds_sources.py','--frames','reds-frames','--out','reds-masters')
} finally {$lock.Dispose()}
