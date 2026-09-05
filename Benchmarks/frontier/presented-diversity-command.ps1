$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\presented-diversity-20260905'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'}) {throw 'Preserve existing Python jobs'}
foreach($path in @('adversarial','adversarial-result.json','owner.lock')) {if(Test-Path $path){throw "Preserve existing $path"}}
$expected=@{
    'Tools\experiments\train_presented_detail.py'='927e438b6e5ad16f1553455790ffffef8cd9e6f9b92c1de9116bee823187ae8d';
    'data\bank\manifest.json'='b11ccdaba5ed388ec615da4270058b6441369fb6325a6e5e6680472e98d6d2f0';
    'C:\lucid\presented-detail-20260904\shipping.pth'='fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65'
}
foreach($path in $expected.Keys) {if((Get-FileHash $path -Algorithm SHA256).Hash -ne $expected[$path]){throw "Changed input: $path"}}
New-Item -ItemType File 'owner.lock' -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH='C:\lucid\pixrestore-baseline-20260904\deps'
    $env:TORCHDYNAMO_DISABLE='1'
    $arguments=@('-u','Tools/experiments/train_presented_detail.py','--bank','data/bank','--pixrestore-cache','data/teacher-cache','--shipping-cache','shipping-cache','--init','C:\lucid\presented-detail-20260904\shipping.pth','--batch','4','--crop','96','--lr','0.00002','--seed','20260914','--teacher-mix','0.5','--steps','8000','--out','adversarial','--dino-gan-weight','0.005','--pixrestore-repository','C:\lucid\pixrestore-baseline-20260904','--dino-repository','C:\lucid\causal-frontier-20260904-dino\third_party\dinov2','--dino-checkpoint','C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
    $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput 'adversarial.log' -RedirectStandardError 'adversarial.err' -Wait -PassThru
    @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content 'adversarial-result.json'
    if($job.ExitCode -ne 0){throw 'Broader-data training failed'}
} finally {Remove-Item 'owner.lock'}
