$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location C:\lucid\fidesr-20260905-r1
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe')}){throw 'Preserve current GPU jobs'}
$prior=Get-Content C:\lucid\remote-access\jobs\paired-critic-r2-20260905\status.json -Raw | ConvertFrom-Json
if($prior.state -ne 'completed' -or $prior.exitCode -ne 0){throw 'Student and its evaluations must finish first'}
$lock=[IO.File]::Open((Join-Path (Get-Location) 'owner.lock'),'CreateNew','ReadWrite','None')
function Run-Python($label,$arguments){
 $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
 @{exitCode=$p.ExitCode;finishedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content "$label-exit.json"
 if($p.ExitCode -ne 0){throw "$label failed"}
}
try {
 $env:PYTHONPATH='C:\lucid\fidesr-20260905-r1\deps;C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
 $common=@('-u','Tools/frontier_eval/fidesr_baseline.py','--repository','repository','--assets','.','--sources','sources.json','--frames','C:\lucid\presented-frozen-eval-20260904\frames')
 Run-Python 'smoke' ($common+@('--smoke','--out','smoke'))
 Run-Python 'full' ($common+@('--out','full'))
 Run-Python 'score' @('-u','Tools/frontier_eval/score_spatial_frames.py','--device','cuda','--frames','C:\lucid\presented-frozen-eval-20260904\frames','--outputs','full','--label','fidesr','--report','score.json')
 Run-Python 'shipping' @('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames','C:\lucid\presented-frozen-eval-20260904\frames','--checkpoint','shipping','C:\lucid\presented-detail-20260904\shipping.pth','--present-4x-at-2x','shipping','--report','shipping.json')
 Run-Python 'gate' @('Tools/frontier_eval/gate_external_development.py','--candidate','score.json','--shipping','shipping.json','--outputs','full/outputs.json','--label','fidesr','--out','gate.json')
} finally {$lock.Dispose()}
