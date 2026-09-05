param([ValidateSet('smoke','train')][string]$Mode='smoke')
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\anchored-detail-20260905'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'}) {throw 'Preserve existing Python jobs'}
if(Test-Path 'owner.lock'){throw 'Preserve existing owner lock'}
$expected=@{
    'Tools\experiments\train_presented_detail.py'='32c4cb10b23624ff79c5d45eb22628a76e891f00bc80e73aec175771c9447d46';
    'Tools\architectures\anchored_detail.py'='274ffeffb08d8131b3f701302249ed34081cf5029918e343465d5c9b9c1b25f0';
    'C:\lucid\presented-diversity-20260905\data\bank\manifest.json'='b11ccdaba5ed388ec615da4270058b6441369fb6325a6e5e6680472e98d6d2f0';
    'C:\lucid\presented-detail-20260904\shipping.pth'='fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65'
}
foreach($path in $expected.Keys){if((Get-FileHash $path -Algorithm SHA256).Hash -ne $expected[$path]){throw "Changed input: $path"}}
$labels=if($Mode -eq 'smoke'){@('coupled-smoke','anchored-smoke')}else{@('anchored')}
foreach($label in $labels){if((Test-Path $label) -or (Test-Path "$label-result.json")){throw "Preserve existing $label"}}
function Check-Smoke {
    $control=Get-Content 'coupled-smoke/experiment.json' -Raw | ConvertFrom-Json
    $candidate=Get-Content 'anchored-smoke/experiment.json' -Raw | ConvertFrom-Json
    foreach($key in @('discriminator_initial_sha256','first_batch_sha256')){
        if(!$control.$key -or ($control.$key -ne $candidate.$key)){throw "Mismatched smoke evidence: $key"}
    }
    foreach($label in @('coupled-smoke','anchored-smoke')){
        $result=Get-Content "$label-result.json" -Raw | ConvertFrom-Json
        if($result.exit_code -ne 0){throw "Failed smoke: $label"}
    }
}
if($Mode -eq 'train'){Check-Smoke}
New-Item -ItemType File 'owner.lock' -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH='C:\lucid\pixrestore-baseline-20260904\deps'
    $env:TORCHDYNAMO_DISABLE='1'
    $shared=@('-u','Tools/experiments/train_presented_detail.py','--bank','C:\lucid\presented-diversity-20260905\data\bank','--pixrestore-cache','C:\lucid\presented-diversity-20260905\data\teacher-cache','--shipping-cache','C:\lucid\presented-diversity-20260905\shipping-cache','--init','C:\lucid\presented-detail-20260904\shipping.pth','--batch','4','--crop','96','--lr','0.00002','--seed','20260914','--teacher-mix','0.5','--dino-gan-weight','0.005','--pixrestore-repository','C:\lucid\pixrestore-baseline-20260904','--dino-repository','C:\lucid\causal-frontier-20260904-dino\third_party\dinov2','--dino-checkpoint','C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
    foreach($label in $labels){
        $architecture=if($label -eq 'coupled-smoke'){'coupled'}else{'anchored_detail'}
        $steps=if($Mode -eq 'smoke'){'2'}else{'8000'}
        $arguments=$shared+@('--architecture',$architecture,'--steps',$steps,'--out',$label)
        $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
        @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content "$label-result.json"
        if($job.ExitCode -ne 0){throw "$label failed"}
    }
    if($Mode -eq 'smoke'){Check-Smoke}
} finally {Remove-Item 'owner.lock'}
