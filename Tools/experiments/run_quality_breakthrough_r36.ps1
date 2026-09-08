$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\quality-breakthrough-20260908-r36'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe')}){throw 'Preserve existing Python jobs'}
$lock=[IO.File]::Open((Join-Path (Get-Location) 'owner.lock'),'CreateNew','ReadWrite','None')
function Run-Python($label,$arguments){
 $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
 @{exitCode=$p.ExitCode;finishedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content "$label-exit.json"
 if($p.ExitCode -ne 0){throw "$label failed"}
}
try {
 $env:PATH='C:\ffmpeg\bin;'+$env:PATH
 $env:PYTHONPATH='C:\lucid\quality-breakthrough-20260908-r36\Tools;C:\lucid\quality-breakthrough-20260908-r36\Tools\experiments;C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
 $base=@('-u','Tools/experiments/train_presented_detail.py','--bank','C:\lucid\stream-bank-v4','--pixrestore-cache','none','--shipping-cache','none','--init','C:\lucid\paired-ladder-20260905-r10\big_2k\step002000.pth','--batch','4','--crop','96','--lr','0.00002','--seed','20260914','--teacher-mix','0','--intended','reference','--architecture','coupled','--detail-target','reference','--training-frames','3','--steps','1000','--paired-dino','--dino-gan-weight','0.0075','--pixrestore-repository','C:\lucid\pixrestore-baseline-20260904','--dino-repository','C:\lucid\causal-frontier-20260904-dino\third_party\dinov2','--dino-checkpoint','C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
 Run-Python 'fidelity-unit-tests' @('-m','unittest','discover','-s','Tools/experiments','-p','test_fidelity_losses.py')
 # Independent loss ablations; control is the matched r33 big_control_1k.
 foreach($arm in @('big_clean_lr_1k','big_aesop_1k','big_aesop_clean_lr_1k')){
  $arguments=$base+@('--out',$arm)
  if($arm -like '*aesop*'){$arguments+=@('--aesop-checkpoint','AE_RRDBdecoder_100K.pth')}
  if($arm -like '*clean_lr*'){$arguments+=@('--clean-lr-weight','1')}
  Run-Python "$arm-train" $arguments
  Run-Python "$arm-development" @('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames','C:\lucid\presented-frozen-eval-20260904\frames','--checkpoint',$arm,"$arm/step001000.pth",'--checkpoint','big2k','C:\lucid\paired-ladder-20260905-r10\big_2k\step002000.pth','--report',"$arm-development.json")
 }
} finally {$lock.Dispose()}
