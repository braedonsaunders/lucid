param([ValidateSet('smoke','train')][string]$Mode='smoke')
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\anchored-lowpass-20260905'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'}) {throw 'Preserve existing Python jobs'}
if(Test-Path 'owner.lock'){throw 'Preserve existing owner lock'}
$expected=@{
    'Tools\experiments\train_presented_detail.py'='a24df29bb051eff4e2a0b6d5044435f6b791dbefb21fb50c6fb0fa07cd2d8b06';
    'Tools\architectures\anchored_detail.py'='7d518a670d5161329ee210040ddb2b181b247eb7db8947ec8627e0e6b590d5a8';
    'C:\lucid\presented-diversity-20260905\data\bank\manifest.json'='b11ccdaba5ed388ec615da4270058b6441369fb6325a6e5e6680472e98d6d2f0';
    'C:\lucid\presented-detail-20260904\shipping.pth'='fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65'
}
foreach($path in $expected.Keys){if((Get-FileHash $path -Algorithm SHA256).Hash -ne $expected[$path]){throw "Changed input: $path"}}
$labels=if($Mode -eq 'smoke'){@('unfiltered-smoke','filtered-smoke')}else{@('filtered')}
foreach($label in $labels){if((Test-Path $label) -or (Test-Path "$label-result.json")){throw "Preserve existing $label"}}
function Check-Smoke {
    $control=Get-Content 'unfiltered-smoke/experiment.json' -Raw | ConvertFrom-Json
    $candidate=Get-Content 'filtered-smoke/experiment.json' -Raw | ConvertFrom-Json
    foreach($key in @('discriminator_initial_sha256','first_batch_sha256')){
        if(!$control.$key -or ($control.$key -ne $candidate.$key)){throw "Mismatched smoke evidence: $key"}
    }
    foreach($label in @('unfiltered-smoke','filtered-smoke')){
        $result=Get-Content "$label-result.json" -Raw | ConvertFrom-Json
        if($result.exit_code -ne 0){throw "Failed smoke: $label"}
    }
}
if($Mode -eq 'train'){
    Check-Smoke
    $profile=Get-Content 'native-profile.json' -Raw | ConvertFrom-Json
    $row=@($profile.rows | Where-Object {$_.input_size -eq '1280x720' -and $_.compute_units -eq 'CPU_AND_GPU'})
    if(!$profile.complete -or !$profile.anchored_probe.residual_lowpass -or $row.Count -ne 1 -or $row[0].timings.anchored_detail_untrained.mean_ms -ge 30){throw 'Native feasibility gate not met'}
    if($profile.anchored_probe.code_sha256 -ne $expected['Tools\architectures\anchored_detail.py']){throw 'Native graph source differs'}
    $old=Get-Content 'C:\lucid\anchored-detail-20260905\anchored\experiment.json' -Raw | ConvertFrom-Json
    $smoke=Get-Content 'unfiltered-smoke/experiment.json' -Raw | ConvertFrom-Json
    foreach($key in @('first_batch_sha256','discriminator_initial_sha256','bank_sha256','checkpoint_sha256','teacher_manifest_sha256','shipping_manifest_sha256')){
        if(!$old.$key -or $old.$key -ne $smoke.$key){throw "Historical control mismatch: $key"}
    }
    foreach($key in @('batch','crop','lr','seed','teacher_mix','dino_gan_weight','detail_channels','detail_blocks')){
        if($old.args.$key -ne $smoke.args.$key){throw "Historical training setting changed: $key"}
    }
}
New-Item -ItemType File 'owner.lock' -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH='C:\lucid\pixrestore-baseline-20260904\deps'
    $env:TORCHDYNAMO_DISABLE='1'
    $shared=@('-u','Tools/experiments/train_presented_detail.py','--bank','C:\lucid\presented-diversity-20260905\data\bank','--pixrestore-cache','C:\lucid\presented-diversity-20260905\data\teacher-cache','--shipping-cache','C:\lucid\presented-diversity-20260905\shipping-cache','--init','C:\lucid\presented-detail-20260904\shipping.pth','--batch','4','--crop','96','--lr','0.00002','--seed','20260914','--teacher-mix','0.5','--dino-gan-weight','0.005','--pixrestore-repository','C:\lucid\pixrestore-baseline-20260904','--dino-repository','C:\lucid\causal-frontier-20260904-dino\third_party\dinov2','--dino-checkpoint','C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
    foreach($label in $labels){
        $architecture=if($label -eq 'unfiltered-smoke'){'anchored_detail'}else{'anchored_lowpass'}
        $steps=if($Mode -eq 'smoke'){'2'}else{'8000'}
        $arguments=$shared+@('--architecture',$architecture,'--steps',$steps,'--out',$label)
        $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
        @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content "$label-result.json"
        if($job.ExitCode -ne 0){throw "$label failed"}
    }
    if($Mode -eq 'smoke'){Check-Smoke}
} finally {Remove-Item 'owner.lock'}
