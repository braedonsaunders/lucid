$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\quality-breakthrough-20260908-r44'
$unexpected=Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe')}
if($unexpected){throw 'Preserve existing Python jobs; completed data and control runs required'}
$lock=[IO.File]::Open((Join-Path (Get-Location) 'owner.lock'),'CreateNew','ReadWrite','None')
function Run-Python($label,$arguments){
 $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
 @{exitCode=$p.ExitCode;finishedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content "$label-exit.json"
 if($p.ExitCode -ne 0){throw "$label failed"}
}
try {
 $env:PYTHONPATH='C:\lucid\quality-breakthrough-20260908-r44\Tools;C:\lucid\quality-breakthrough-20260908-r44\Tools\experiments;C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
 $base=@('-u','Tools/experiments/train_presented_detail.py','--bank','I:\lucid-quality-breakthrough-r37\expanded-mapped','--pixrestore-cache','none','--shipping-cache','none','--crop','96','--seed','20260918','--teacher-mix','0','--intended','reference','--architecture','coupled','--detail-target','reference','--paired-dino','--dino-gan-weight','0.0075','--pixrestore-repository','C:\lucid\pixrestore-baseline-20260904','--dino-repository','C:\lucid\causal-frontier-20260904-dino\third_party\dinov2','--dino-checkpoint','C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
 Run-Python 'sampling-unit-tests' @('-m','unittest','discover','-s','Tools/experiments','-p','test_source_sampling.py')
 foreach($arm in @('balanced_base_critic_2k','balanced_shipping_critic_2k')){
  $initial=if($arm -eq 'balanced_base_critic_2k'){'C:\lucid\quality-breakthrough-20260908-r39\expanded_base_40k\step040000.pth'}else{'C:\lucid\paired-ladder-20260905-r10\big_2k\step002000.pth'}
  Run-Python "$arm-train" ($base+@('--source-balanced','--init',$initial,'--batch','4','--lr','0.00002','--steps','2000','--out',$arm))
  foreach($corpus in @('development','reds')){
   $frames=if($corpus -eq 'development'){'C:\lucid\presented-frozen-eval-20260904\frames'}else{'C:\lucid\quality-breakthrough-20260908-r39\reds-validation-frames'}
   Run-Python "$arm-$corpus" @('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames',$frames,'--checkpoint',$arm,"$arm/step002000.pth",'--checkpoint','big2k','C:\lucid\paired-ladder-20260905-r10\big_2k\step002000.pth','--report',"$arm-$corpus.json")
  }
 }
} finally {$lock.Dispose()}
