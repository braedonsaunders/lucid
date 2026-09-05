$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\nanovsr-baseline-20260905'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'}){throw 'Preserve existing Python jobs'}
foreach($path in @('owner.lock','evaluation-with-ffmpeg.json','evaluation-with-ffmpeg-result.json')){if(Test-Path $path){throw "Preserve existing $path"}}
if((Get-FileHash 'C:\lucid\presented-detail-20260904\shipping.pth').Hash -ne 'fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65'){throw 'Shipping weights changed'}
New-Item -ItemType File owner.lock -ErrorAction Stop | Out-Null
try {
    $env:PATH='C:\ffmpeg\bin;' + $env:PATH
    Get-Command ffmpeg -ErrorAction Stop | Out-Null
    $env:PYTHONPATH='C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
    $arguments=@('-u','Tools/frontier_eval/nanovsr_baseline.py','--source','nanovsr.py','--weights','nanovsr_644k.pth','--source-sha256','a7eae614f0dbfb5d9532d312d2eebdae0c560fd34de7dae2384c3d1b14e10125','--weights-sha256','515eec20ff589986fc0e0da18b363962b931008ebd848d0c8fb7e256ce200cf7','--shipping','C:\lucid\presented-detail-20260904\shipping.pth','--manifest','C:\lucid\presented-eval-20260904\sequences.json','--device','cuda','--out','evaluation-with-ffmpeg.json')
    $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput 'evaluation-with-ffmpeg.log' -RedirectStandardError 'evaluation-with-ffmpeg.err' -Wait -PassThru
    @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content 'evaluation-with-ffmpeg-result.json'
    if($job.ExitCode -ne 0){throw 'NanoVSR comparison failed'}
} finally {Remove-Item owner.lock}
