$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\reds-training-subset-20260905'
if (Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }) { throw 'Preserve existing Python jobs' }
foreach($path in @('teacher-cache','teacher-result.json','owner.lock')) { if(Test-Path $path){throw "Preserve existing $path"} }
$script='C:\lucid\causal-pixrestore-distill-20260904\Tools\experiments\cache_pixrestore_teacher.py'
if ((Get-FileHash $script -Algorithm SHA256).Hash -ne '2bc7f6a1cd85c3bdd489a0e645d7d0ea31c443b875cdce26456c854b9504b088') { throw 'Teacher utility changed' }
if ((Get-FileHash 'bank\manifest.json' -Algorithm SHA256).Hash -ne 'ecc0e205f0a28ea0e23c4a98328479de2d30c5ab89ed7ed9e2d7103f576dabef') { throw 'Bank changed' }
New-Item -ItemType File 'owner.lock' -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH='C:\lucid\pixrestore-baseline-20260904\deps'
    $env:TORCHDYNAMO_DISABLE='1'
    $arguments=@('-u',$script,'--bank','bank','--out','teacher-cache','--repository','C:\lucid\pixrestore-baseline-20260904','--provenance','C:\lucid\causal-pixrestore-distill-20260904\teacher-provenance.json','--checkpoint','C:\lucid\frontier-teachers-20260904\pixrestore-s','--dino-repository','C:\lucid\causal-frontier-20260904-dino\third_party\dinov2','--dino-checkpoint','C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth','--seed','20260913')
    $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput 'teacher.log' -RedirectStandardError 'teacher.err' -Wait -PassThru
    @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content 'teacher-result.json'
    if($job.ExitCode -ne 0){throw 'Teacher target generation failed'}
} finally {Remove-Item 'owner.lock'}
