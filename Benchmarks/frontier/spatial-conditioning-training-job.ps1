$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\coarse-controller-20260905-r1'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe')}){throw 'Preserve existing Python jobs'}
$lock=[IO.File]::Open((Join-Path (Get-Location) 'owner.lock'),'CreateNew','ReadWrite','None')
try {
 $env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
 $env:PYTHONPATH='C:\lucid\pixrestore-baseline-20260904\deps'
 $common=@('--deterministic','--mode','dynamic','--coarse-spatial','--native-preflight','native-preflight.json','--local-fidelity-constraint','--bank','C:\lucid\presented-diversity-20260905\data\bank','--shipping-cache','C:\lucid\presented-diversity-20260905\shipping-cache','--pixrestore-cache','C:\lucid\presented-diversity-20260905\data\teacher-cache','--init','C:\lucid\presented-detail-20260904\shipping.pth','--probe','probe.json','--pixrestore-repository','C:\lucid\pixrestore-baseline-20260904','--dino-repository','C:\lucid\causal-frontier-20260904-dino\third_party\dinov2','--dino-checkpoint','C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
 $prior=Get-Content C:\lucid\local-fidelity-20260905-r4\constrained-smoke\experiment.json -Raw | ConvertFrom-Json
 foreach($label in @('smoke','repeat','trained')){
  if(Test-Path $label){throw "Preserve existing $label"}
  $steps=if($label -eq 'trained'){8000}else{2}
  $arguments=@('-u','Tools/experiments/train_activation_control.py','--out',$label,'--steps',"$steps")+$common
  $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
  @{exitCode=$p.ExitCode;finishedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content "$label-exit.json"
  if($p.ExitCode -ne 0){throw "$label failed"}
  $actual=Get-Content "$label/experiment.json" -Raw | ConvertFrom-Json
  foreach($field in @('bank_sha256','checkpoint_sha256','teacher_manifest_sha256','shipping_manifest_sha256','probe_sha256','frozen_anchor_and_masks_sha256','controller_initial_sha256','discriminator_initial_sha256','first_batch_sha256','first_output_sha256','first_fidelity_baseline_sha256')){
   if($actual.$field -ne $prior.$field){throw "Changed R4 matched identity: $field"}
  }
  if($label -eq 'repeat'){
   $a=Get-Content smoke/complete.json -Raw | ConvertFrom-Json
   $b=Get-Content repeat/complete.json -Raw | ConvertFrom-Json
   if($a.controller_final_sha256 -ne $b.controller_final_sha256){throw 'Coarse repeated CUDA smokes differ'}
  }
 }

 $done=Get-Content trained/complete.json -Raw | ConvertFrom-Json
 if($done.steps -ne 8000 -or !$done.frozen_anchor_unchanged){throw 'Training endpoint invalid'}
 New-Item -ItemType Directory evaluation -ErrorAction Stop | Out-Null
 $env:PYTHONPATH='C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
 foreach($mode in @('development','bank')){
  $frames=if($mode -eq 'development'){'C:\lucid\presented-frozen-eval-20260904\frames'}else{'C:\lucid\anchored-detail-20260905\bank-validation-frames'}
  $arguments=@('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames',$frames,'--checkpoint','coarse','trained/step008000.pth','--checkpoint','shipping','C:\lucid\presented-detail-20260904\shipping.pth','--present-4x-at-2x','shipping','--report',"evaluation/$mode.json")
  $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$mode-evaluation.log" -RedirectStandardError "$mode-evaluation.err" -Wait -PassThru
  if($p.ExitCode -ne 0){throw "$mode evaluation failed"}
  $arguments=@('Tools/frontier_eval/gate_controller_candidate.py','--mode',$mode,'--report',"evaluation/$mode.json",'--control',"C:\lucid\local-fidelity-20260905-r4\Benchmarks\frontier\local-fidelity-$mode.json",'--checkpoint','trained/step008000.pth','--out',"evaluation/$mode-gate.json")
  $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$mode-gate.log" -RedirectStandardError "$mode-gate.err" -Wait -PassThru
  if($p.ExitCode -ne 0){throw "$mode control verification/gating failed"}
 }

} finally {$lock.Dispose()}
