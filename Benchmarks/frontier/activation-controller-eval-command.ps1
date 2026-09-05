param([ValidateSet('development','bank','coverage720')][string]$Mode='development')
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\activation-controller-20260905'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'}){throw 'Preserve existing Python jobs'}
foreach($path in @('owner.lock',"$Mode.json","$Mode-result.json")){if(Test-Path $path){throw "Preserve existing $path"}}
foreach($label in @('static','dynamic')){
 if((Get-Content "$label-result.json" -Raw | ConvertFrom-Json).exit_code -ne 0 -or (Get-Content "$label/complete.json" -Raw | ConvertFrom-Json).steps -ne 8000){throw 'Both complete training arms required'}
}
$frames=switch($Mode){'development'{'C:\lucid\presented-frozen-eval-20260904\frames'} 'bank'{'C:\lucid\anchored-detail-20260905\bank-validation-frames'} 'coverage720'{'C:\lucid\coverage-720-20260905\frames'}}
$expected=switch($Mode){'development'{'aac663ad088fede87e1ede68cb46501603fc0f0b066b5ae6c07726b6555d923f'} 'bank'{'eb8ad3792cc1999f4dc593494f959656be28b815ee21d3a9094c47b683f3a7fe'} 'coverage720'{'d2946f12954e6ba799dee2caf24216f96fb4c74904cf364a9549145aed4c491a'}}
if((Get-FileHash "$frames/manifest.json").Hash -ne $expected){throw 'Frozen manifest changed'}
New-Item -ItemType File owner.lock -ErrorAction Stop | Out-Null
try {
 $env:PYTHONPATH='C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
 $arguments=@('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames',$frames,'--checkpoint','dynamic','dynamic/step008000.pth','--checkpoint','static','static/step008000.pth','--checkpoint','shipping','C:\lucid\presented-detail-20260904\shipping.pth','--present-4x-at-2x','shipping','--report',"$Mode.json")
 if($Mode -eq 'coverage720'){$arguments+=@('--interpolation','lanczos','bicubic','bilinear')}
 $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -RedirectStandardOutput "$Mode.log" -RedirectStandardError "$Mode.err" -Wait -PassThru
 @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content "$Mode-result.json"
 if($job.ExitCode -ne 0){throw "$Mode evaluation failed"}
} finally {Remove-Item owner.lock}
