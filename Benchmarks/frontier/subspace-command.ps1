param([ValidateSet('smoke','train')][string]$Mode='smoke')
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\subspace-adapter-20260905'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'}){throw 'Preserve existing Python jobs'}
if(Test-Path owner.lock){throw 'Preserve owner lock'}
$expected=@{
 'Tools\experiments\train_presented_detail.py'='b27e7a8ccc07e4c448cb323647b617f2fc21f63539c759f714dab1cf52228e08';
 'Tools\architectures\subspace_adapter.py'='1098e70a08e118f31e009dff2711fed1d70bee2e85e8cd1283bf1ebd107c8f99';
 'Tools\eval_checkpoint.py'='57800a8080c961f3ee325f5f3b5eb7577404b526b3047f00fbe95491168c7f34';
 'native-profile.json'='2b56bcb21ed3fa62547c9822df0c10b318a5f120d1ec46f9873c6756c6145e01';
 'preflight.json'='785939994efaa6b29dc1f813ec96e2ab49980ddff4b12ebe8430620f82c64715';
 'C:\lucid\presented-diversity-20260905\data\bank\manifest.json'='b11ccdaba5ed388ec615da4270058b6441369fb6325a6e5e6680472e98d6d2f0';
 'C:\lucid\presented-detail-20260904\shipping.pth'='fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65'
}
foreach($path in $expected.Keys){if((Get-FileHash $path).Hash -ne $expected[$path]){throw "Changed input: $path"}}
$labels=if($Mode -eq 'smoke'){@('full-smoke','protected-smoke')}else{@('full','protected')}
foreach($label in $labels){if((Test-Path $label) -or (Test-Path "$label-result.json")){throw "Preserve existing $label"}}
function Check-Smoke {
 $control=Get-Content full-smoke/experiment.json -Raw | ConvertFrom-Json
 $candidate=Get-Content protected-smoke/experiment.json -Raw | ConvertFrom-Json
 foreach($key in @('first_batch_sha256','first_output_sha256','discriminator_initial_sha256','bank_sha256','checkpoint_sha256','teacher_manifest_sha256','shipping_manifest_sha256')){
  if(!$control.$key -or $control.$key -ne $candidate.$key){throw "Smoke mismatch: $key"}
 }
 foreach($label in @('full-smoke','protected-smoke')){if((Get-Content "$label-result.json" -Raw | ConvertFrom-Json).exit_code -ne 0){throw "Smoke failed: $label"}}
}
if($Mode -eq 'train'){
 Check-Smoke
 $profile=Get-Content native-profile.json -Raw | ConvertFrom-Json
 $row=@($profile.rows | Where-Object {$_.input_size -eq '1280x720' -and $_.compute_units -eq 'CPU_AND_GPU'})
 if(!$profile.complete -or $row.Count -ne 1 -or $row[0].timings.direct2x_trained.mean_ms -ge 30){throw 'Native feasibility failed'}
 $preflight=Get-Content preflight.json -Raw | ConvertFrom-Json
 if($profile.direct_checkpoint_sha256 -ne $preflight.checkpoint_sha256){throw 'Native preflight identity changed'}
}
New-Item -ItemType File owner.lock -ErrorAction Stop | Out-Null
try {
 $env:PYTHONPATH='C:\lucid\pixrestore-baseline-20260904\deps'
 $env:TORCHDYNAMO_DISABLE='1'
 $shared=@('-u','Tools/experiments/train_presented_detail.py','--bank','C:\lucid\presented-diversity-20260905\data\bank','--pixrestore-cache','C:\lucid\presented-diversity-20260905\data\teacher-cache','--shipping-cache','C:\lucid\presented-diversity-20260905\shipping-cache','--init','C:\lucid\presented-detail-20260904\shipping.pth','--batch','4','--crop','96','--lr','0.00002','--seed','20260914','--teacher-mix','0.5','--detail-target','reference','--dino-gan-weight','0.005','--pixrestore-repository','C:\lucid\pixrestore-baseline-20260904','--dino-repository','C:\lucid\causal-frontier-20260904-dino\third_party\dinov2','--dino-checkpoint','C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
 foreach($label in $labels){
  $architecture=if($label -like 'full*'){'subspace_full'}else{'subspace_protected'}
  $steps=if($Mode -eq 'smoke'){'2'}else{'8000'}
  $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList ($shared+@('--architecture',$architecture,'--steps',$steps,'--out',$label)) -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
  @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content "$label-result.json"
  if($job.ExitCode -ne 0){throw "$label failed"}
 }
 if($Mode -eq 'smoke'){Check-Smoke}
} finally {Remove-Item owner.lock}
