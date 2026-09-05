param([ValidateSet('development','bank')][string]$Mode='development')
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\anchored-lowpass-20260905'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'}){throw 'Preserve existing Python jobs'}
foreach($path in @('owner.lock',"$Mode.json","$Mode-result.json")){if(Test-Path $path){throw "Preserve existing $path"}}
if((Get-Content filtered-result.json -Raw | ConvertFrom-Json).exit_code -ne 0){throw 'Training did not succeed'}
if((Get-Content filtered/complete.json -Raw | ConvertFrom-Json).steps -ne 8000){throw 'Full schedule required'}
$frames=if($Mode -eq 'development'){'C:\lucid\presented-frozen-eval-20260904\frames'}else{'C:\lucid\anchored-detail-20260905\bank-validation-frames'}
$expected=if($Mode -eq 'development'){'aac663ad088fede87e1ede68cb46501603fc0f0b066b5ae6c07726b6555d923f'}else{'eb8ad3792cc1999f4dc593494f959656be28b815ee21d3a9094c47b683f3a7fe'}
if((Get-FileHash "$frames\manifest.json").Hash -ne $expected){throw 'Frozen frame manifest changed'}
New-Item -ItemType File owner.lock -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH='C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
    $arguments=@('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames',$frames,'--checkpoint','filtered','filtered/step008000.pth','--checkpoint','unfiltered','C:\lucid\anchored-detail-20260905\anchored\step008000.pth','--checkpoint','folded','C:\lucid\presented-diversity-eval-20260905\area-init.pth','--checkpoint','shipping','C:\lucid\presented-detail-20260904\shipping.pth','--present-4x-at-2x','shipping','--report',"$Mode.json")
    $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput "$Mode.log" -RedirectStandardError "$Mode.err" -Wait -PassThru
    @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content "$Mode-result.json"
    if($job.ExitCode -ne 0){throw "$Mode scoring failed"}
} finally {Remove-Item owner.lock}
