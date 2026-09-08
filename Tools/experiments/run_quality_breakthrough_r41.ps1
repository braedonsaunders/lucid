$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\quality-breakthrough-20260908-r41'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe') -and $_.CommandLine -notlike '*mmap_training_bank.py*--bank C:\lucid\stream-bank-v4 --bank reds-bank --out I:\lucid-quality-breakthrough-r37\expanded-mapped*'}){throw 'Preserve existing Python jobs; only owned CPU bank mapping may overlap'}
$lock=[IO.File]::Open((Join-Path (Get-Location) 'owner.lock'),'CreateNew','ReadWrite','None')
function Run-Python($label,$arguments){
 $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
 @{exitCode=$p.ExitCode;finishedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content "$label-exit.json"
 if($p.ExitCode -ne 0){throw "$label failed"}
}
try {
 $env:PYTHONPATH='C:\lucid\quality-breakthrough-20260908-r41\Tools;C:\lucid\quality-breakthrough-20260908-r41\Tools\experiments;C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
 Run-Python 'recurrent-unit-tests' @('-m','unittest','discover','-s','Tools/experiments','-p','test_recurrent_frame_feeding.py')
 $base=@('-u','Tools/experiments/train_recurrent_span.py','--bank','C:\lucid\stream-bank-v4','--init','C:\lucid\paired-ladder-20260905-r10\big_2k\step002000.pth','--batch','4','--crop','96','--frames','3','--seed','20260914','--steps','2000','--native-input-stages','--joint','--dino-gan-weight','0.0075','--pixrestore-repository','C:\lucid\pixrestore-baseline-20260904','--dino-repository','C:\lucid\causal-frontier-20260904-dino\third_party\dinov2','--dino-checkpoint','C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
 foreach($arm in @('joint_control_2k','joint_recurrent_2k')){
  $arguments=$base+@('--out',$arm)
  if($arm -eq 'joint_control_2k'){$arguments+=@('--no-history')}
  Run-Python "$arm-train" $arguments
 }
} finally {$lock.Dispose()}
