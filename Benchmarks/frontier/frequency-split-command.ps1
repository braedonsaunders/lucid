$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\frequency-split-20260905'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'}){throw 'Preserve existing Python jobs'}
foreach($path in @('owner.lock','evaluation.json','result.json')){if(Test-Path $path){throw "Preserve existing $path"}}
if((Get-FileHash 'C:\lucid\presented-frozen-eval-20260904\frames\manifest.json').Hash -ne 'aac663ad088fede87e1ede68cb46501603fc0f0b066b5ae6c07726b6555d923f'){throw 'Frame manifest changed'}
New-Item -ItemType File owner.lock -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH='C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
    $arguments=@('-u','Tools/frontier_eval/frequency_split_probe.py','--device','cuda','--frames','C:\lucid\presented-frozen-eval-20260904\frames','--shipping','C:\lucid\presented-detail-20260904\shipping.pth','--candidate','C:\lucid\reference-detail-20260905\calibrated\blend-0.8.pth','--out','evaluation.json')
    $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput 'evaluation.log' -RedirectStandardError 'evaluation.err' -Wait -PassThru
    @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content 'result.json'
    if($job.ExitCode -ne 0){throw 'Spatial decomposition failed'}
} finally {Remove-Item owner.lock}
