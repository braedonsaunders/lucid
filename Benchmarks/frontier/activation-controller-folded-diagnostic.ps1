$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\activation-controller-20260905-r2'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe')}){throw 'Preserve existing Python jobs'}
foreach($file in @('owner.lock','folded-diagnostic.json')){if(Test-Path $file){throw "Preserve $file"}}
New-Item -ItemType File owner.lock -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH='C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
    & C:\lucid\.venv\Scripts\python.exe -u Tools/frontier_eval/score_checkpoint_frames.py --device cuda --frames C:\lucid\presented-frozen-eval-20260904\frames --checkpoint folded_initialization folded-initialization.pth --checkpoint shipping C:\lucid\presented-detail-20260904\shipping.pth --present-4x-at-2x shipping --report folded-diagnostic.json
    $result=$LASTEXITCODE
} finally {Remove-Item owner.lock}
exit $result
