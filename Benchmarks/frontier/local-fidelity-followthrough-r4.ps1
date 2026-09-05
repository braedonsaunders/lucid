$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\local-fidelity-20260905-r4'
$trainingStatus='C:\lucid\remote-access\jobs\local-fidelity-train-r4-20260905\status.json'
$deadline=(Get-Date).AddHours(4)
do {
    if((Get-Date) -gt $deadline){throw 'Training supervision deadline exceeded; preserve job'}
    $state=(Get-Content $trainingStatus -Raw | ConvertFrom-Json).state
    if($state -eq 'failed'){throw 'Training failed; no evaluation or automatic retry'}
    if($state -ne 'completed'){Start-Sleep -Seconds 15}
} while($state -ne 'completed')
foreach($label in @('unconstrained','constrained')){
    if((Get-Content "$label-result.json" -Raw | ConvertFrom-Json).exit_code -ne 0){throw 'Training exit failed'}
    $done=Get-Content "$label/complete.json" -Raw | ConvertFrom-Json
    if($done.steps -ne 8000 -or !$done.frozen_anchor_unchanged){throw 'Incomplete or changed reconstruction'}
}
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe')}){throw 'Preserve existing Python jobs'}
if(Test-Path owner.lock){throw 'Existing GPU owner'}
New-Item -ItemType File owner.lock -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH='C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
    foreach($mode in @('development','bank')){
        $frames=if($mode -eq 'development'){'C:\lucid\presented-frozen-eval-20260904\frames'}else{'C:\lucid\anchored-detail-20260905\bank-validation-frames'}
        $report="Benchmarks/frontier/local-fidelity-$mode.json"
        if(Test-Path $report){throw 'Preserve existing evaluation'}
        $arguments=@('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames',$frames,'--checkpoint','constrained','constrained/step008000.pth','--checkpoint','unconstrained','unconstrained/step008000.pth','--checkpoint','shipping','C:\lucid\presented-detail-20260904\shipping.pth','--present-4x-at-2x','shipping','--report',$report)
        $evaluation=Start-Process -FilePath C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$mode-evaluation.log" -RedirectStandardError "$mode-evaluation.err" -Wait -PassThru
        if($evaluation.ExitCode -ne 0){exit $evaluation.ExitCode}
        $scoring=Start-Process -FilePath C:\lucid\.venv\Scripts\python.exe -ArgumentList @('Benchmarks/frontier/local-fidelity-score.py','--mode',$mode,'--checkpoints','.') -RedirectStandardOutput "$mode-gate.log" -RedirectStandardError "$mode-gate.err" -Wait -PassThru
        if($scoring.ExitCode -ne 0){exit $scoring.ExitCode}
    }
} finally {Remove-Item owner.lock}
exit 0
