$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\anchored-detail-20260905'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'}) {throw 'Preserve existing Python jobs'}
foreach($path in @('evaluation.json','evaluation-result.json','owner.lock')){if(Test-Path $path){throw "Preserve existing $path"}}
$result=Get-Content 'anchored-result.json' -Raw | ConvertFrom-Json
if($result.exit_code -ne 0){throw 'Anchored training did not succeed'}
if((Get-FileHash 'C:\lucid\presented-frozen-eval-20260904\frames\manifest.json').Hash -ne 'aac663ad088fede87e1ede68cb46501603fc0f0b066b5ae6c07726b6555d923f'){throw 'Development frames changed'}
New-Item -ItemType File 'owner.lock' -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH='C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
    $arguments=@('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames','C:\lucid\presented-frozen-eval-20260904\frames','--checkpoint','anchored','anchored/step008000.pth','--checkpoint','coupled_raw','C:\lucid\presented-diversity-20260905\adversarial\step008000.pth','--checkpoint','coupled80','C:\lucid\presented-diversity-eval-20260905\calibrated\blend-0.8.pth','--checkpoint','folded','C:\lucid\presented-diversity-eval-20260905\area-init.pth','--checkpoint','shipping','C:\lucid\presented-detail-20260904\shipping.pth','--present-4x-at-2x','shipping','--report','evaluation.json')
    $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput 'evaluation.log' -RedirectStandardError 'evaluation.err' -Wait -PassThru
    @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content 'evaluation-result.json'
    if($job.ExitCode -ne 0){throw 'Anchored evaluation failed'}
} finally {Remove-Item 'owner.lock'}
