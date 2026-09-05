param([ValidateSet('smoke','train')][string]$Mode='smoke')
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\activation-controller-20260905-r2'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'}){throw 'Preserve existing Python jobs'}
if(Test-Path owner.lock){throw 'Preserve existing owner lock'}
if((Get-FileHash 'probe.json').Hash -ne '1fbfface8ca3189f1dcb8875f9373d3853e41b4541a0e4e6f4e932252dd63ef5' -or (Get-FileHash 'native-profile.json').Hash -ne 'dfc8dd4e025bcbe6faf06b1739ebbfe872620d7e36f8c8126539578bbd97c67f'){throw 'Changed frozen preflight'}
if(!(Get-Content native-profile.json -Raw | ConvertFrom-Json).complete){throw 'Native preflight incomplete'}
if($Mode -eq 'train'){
 foreach($label in @('static','dynamic')){if((Get-Content "$label-smoke-result.json" -Raw | ConvertFrom-Json).exit_code -ne 0){throw 'Smoke failed'}}
 $a=Get-Content 'static-smoke/experiment.json' -Raw | ConvertFrom-Json
 $b=Get-Content 'dynamic-smoke/experiment.json' -Raw | ConvertFrom-Json
 foreach($field in @('bank_sha256','checkpoint_sha256','teacher_manifest_sha256','shipping_manifest_sha256','probe_sha256','frozen_anchor_and_masks_sha256','controller_initial_sha256','discriminator_initial_sha256','first_batch_sha256','first_output_sha256')){if($a.$field -ne $b.$field){throw "Smoke mismatch: $field"}}
}
New-Item -ItemType File owner.lock -ErrorAction Stop | Out-Null
try {
 $env:PYTHONPATH='C:\lucid\pixrestore-baseline-20260904\deps'
 foreach($label in @('static','dynamic')){
  $output=if($Mode -eq 'smoke'){"$label-smoke"}else{$label}
  $steps=if($Mode -eq 'smoke'){2}else{8000}
  foreach($p in @($output,"$output-result.json","$output.log")){if(Test-Path $p){throw "Preserve existing $p"}}
  $arguments=@('-u','Tools/experiments/train_activation_control.py','--mode',$label,'--steps',$steps,'--out',$output,'--bank','C:\lucid\presented-diversity-20260905\data\bank','--shipping-cache','C:\lucid\presented-diversity-20260905\shipping-cache','--pixrestore-cache','C:\lucid\presented-diversity-20260905\data\teacher-cache','--init','C:\lucid\presented-detail-20260904\shipping.pth','--probe','probe.json','--pixrestore-repository','C:\lucid\pixrestore-baseline-20260904','--dino-repository','C:\lucid\causal-frontier-20260904-dino\third_party\dinov2','--dino-checkpoint','C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
  $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -RedirectStandardOutput "$output.log" -RedirectStandardError "$output.err" -Wait -PassThru
  @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content "$output-result.json"
  if($job.ExitCode -ne 0){throw "$output failed"}
 }
} finally {Remove-Item owner.lock}
