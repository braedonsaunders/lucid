$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\coverage-720-20260905'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'}){throw 'Preserve existing Python jobs'}
foreach($path in @('owner.lock','quality.json','quality-result.json')){if(Test-Path $path){throw "Preserve existing $path"}}
if((Get-FileHash frames/manifest.json).Hash -ne 'd2946f12954e6ba799dee2caf24216f96fb4c74904cf364a9549145aed4c491a'){throw 'Frame manifest changed'}
if((Get-FileHash folded.pth).Hash -ne '8dd69c03424003aa98585edcbbc6c659d882a94fa680efba7d784a28deba241a'){throw 'Candidate changed'}
if((Get-FileHash shipping.pth).Hash -ne 'fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65'){throw 'Shipping changed'}
New-Item -ItemType File owner.lock -ErrorAction Stop | Out-Null
try {
 $env:PYTHONPATH='C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
 $arguments=@('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames','frames','--checkpoint','folded','folded.pth','--checkpoint','shipping','shipping.pth','--present-4x-at-2x','shipping','--interpolation','lanczos','bicubic','bilinear','--report','quality.json')
 $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -RedirectStandardOutput quality.log -RedirectStandardError quality.err -Wait -PassThru
 @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content quality-result.json
 if($job.ExitCode -ne 0){throw 'Quality evaluation failed'}
} finally {Remove-Item owner.lock}
