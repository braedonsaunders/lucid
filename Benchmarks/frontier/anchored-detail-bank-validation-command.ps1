$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\anchored-detail-20260905'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'}) {throw 'Preserve existing Python jobs'}
foreach($path in @('bank-validation.json','bank-validation-result.json','owner.lock')){if(Test-Path $path){throw "Preserve existing $path"}}
$result=Get-Content 'anchored-result.json' -Raw | ConvertFrom-Json
if($result.exit_code -ne 0){throw 'Anchored training did not succeed'}
if((Get-FileHash 'bank-validation-frames\manifest.json').Hash -ne 'eb8ad3792cc1999f4dc593494f959656be28b815ee21d3a9094c47b683f3a7fe'){throw 'Development frames changed'}
New-Item -ItemType File 'owner.lock' -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH='C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
    $arguments=@('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames','bank-validation-frames','--checkpoint','anchored','anchored/step008000.pth','--checkpoint','coupled_raw','C:\lucid\presented-diversity-20260905\adversarial\step008000.pth','--checkpoint','coupled80','C:\lucid\presented-diversity-eval-20260905\calibrated\blend-0.8.pth','--checkpoint','previous80','C:\lucid\presented-frozen-eval-20260904\blend-0.8.pth','--checkpoint','folded','C:\lucid\presented-diversity-eval-20260905\area-init.pth','--checkpoint','shipping','C:\lucid\presented-detail-20260904\shipping.pth','--present-4x-at-2x','shipping','--report','bank-validation.json')
    $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput 'bank-validation.log' -RedirectStandardError 'bank-validation.err' -Wait -PassThru
    @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content 'bank-validation-result.json'
    if($job.ExitCode -ne 0){throw 'Anchored evaluation failed'}
} finally {Remove-Item 'owner.lock'}
