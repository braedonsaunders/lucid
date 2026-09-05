$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\presented-diversity-eval-20260905'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'}) {throw 'Preserve existing Python jobs'}
foreach($path in @('calibrated','evaluation.json','evaluation-result.json','owner.lock')) {if(Test-Path $path){throw "Preserve existing $path"}}
$expected=@{
    'area-init.pth'='8dd69c03424003aa98585edcbbc6c659d882a94fa680efba7d784a28deba241a';
    'C:\lucid\presented-frozen-eval-20260904\frames\manifest.json'='aac663ad088fede87e1ede68cb46501603fc0f0b066b5ae6c07726b6555d923f';
    'C:\lucid\presented-frozen-eval-20260904\blend-0.8.pth'='98886f65c0877f3813f46c6c9895597a8969a6b9f027446ebad5b1d803959852'
}
foreach($path in $expected.Keys) {if((Get-FileHash $path -Algorithm SHA256).Hash -ne $expected[$path]){throw "Changed input: $path"}}
$result=Get-Content 'C:\lucid\presented-diversity-20260905\adversarial-result.json' -Raw | ConvertFrom-Json
if($result.exit_code -ne 0){throw 'Training did not succeed'}
New-Item -ItemType File 'owner.lock' -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH='C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
    & 'C:\lucid\.venv\Scripts\python.exe' Tools/experiments/interpolate_presented_detail.py --anchor area-init.pth --candidate 'C:\lucid\presented-diversity-20260905\adversarial\step008000.pth' --weights 0.8 --out calibrated
    if($LASTEXITCODE -ne 0){throw 'Fixed interpolation failed'}
    $arguments=@('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames','C:\lucid\presented-frozen-eval-20260904\frames','--checkpoint','expanded80','calibrated/blend-0.8.pth','--checkpoint','expanded_raw','C:\lucid\presented-diversity-20260905\adversarial\step008000.pth','--checkpoint','previous80','C:\lucid\presented-frozen-eval-20260904\blend-0.8.pth','--checkpoint','shipping','C:\lucid\presented-detail-20260904\shipping.pth','--present-4x-at-2x','shipping','--report','evaluation.json')
    $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput 'evaluation.log' -RedirectStandardError 'evaluation.err' -Wait -PassThru
    @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content 'evaluation-result.json'
    if($job.ExitCode -ne 0){throw 'Evaluation failed'}
} finally {Remove-Item 'owner.lock'}
