$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\local-fidelity-20260905-r2'
$trainingStatus='C:\lucid\remote-access\jobs\local-fidelity-train-r2-20260905\status.json'
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
        & C:\lucid\.venv\Scripts\python.exe -u Tools/frontier_eval/score_checkpoint_frames.py --device cuda --frames $frames --checkpoint constrained constrained/step008000.pth --checkpoint unconstrained unconstrained/step008000.pth --checkpoint shipping C:\lucid\presented-detail-20260904\shipping.pth --present-4x-at-2x shipping --report $report
        if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
        & C:\lucid\.venv\Scripts\python.exe Benchmarks/frontier/local-fidelity-score.py --mode $mode --checkpoints .
        if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
    }
} finally {Remove-Item owner.lock}
exit 0
