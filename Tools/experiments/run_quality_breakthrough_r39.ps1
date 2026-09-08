$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\quality-breakthrough-20260908-r39'
$unexpected=Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe')}
if($unexpected){throw 'Preserve existing Python jobs; completed data and control runs required'}
$lock=[IO.File]::Open((Join-Path (Get-Location) 'owner.lock'),'CreateNew','ReadWrite','None')
function Run-Python($label,$arguments){
 $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
 @{exitCode=$p.ExitCode;finishedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content "$label-exit.json"
 if($p.ExitCode -ne 0){throw "$label failed"}
}
try {
 $env:PYTHONPATH='C:\lucid\quality-breakthrough-20260908-r39\Tools;C:\lucid\quality-breakthrough-20260908-r39\Tools\experiments;C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
 $control=Get-Content I:\lucid-quality-breakthrough-r37\v4-mapped\manifest.json -Raw | ConvertFrom-Json
 $expanded=Get-Content I:\lucid-quality-breakthrough-r37\expanded-mapped\manifest.json -Raw | ConvertFrom-Json
 if(!$expanded.complete -or $expanded.storage -ne 'mmap-pairs-v1'){throw 'completed mapped expansion required'}
 $oldCount=@($control.sequences | Where-Object {$_.split -eq 'train'}).Count
 $newCount=@($expanded.sequences | Where-Object {$_.split -eq 'train'}).Count
 if($newCount -lt 10*$oldCount){throw '10x training-bank floor not met'}
 if(!(Test-Path C:\lucid\quality-breakthrough-20260908-r38b\control_critic_2k\complete.json)){throw 'matched control must finish first'}
 $base=@('-u','Tools/experiments/train_presented_detail.py','--pixrestore-cache','none','--shipping-cache','none','--crop','96','--seed','20260918','--teacher-mix','0','--intended','reference','--architecture','coupled','--detail-target','reference')
 # Only base-pretraining data changes; the final critic uses original v4 in both arms.
 Run-Python 'expanded_base_40k-train' ($base+@('--bank','I:\lucid-quality-breakthrough-r37\expanded-mapped','--init','C:\lucid\presented-detail-20260904\shipping.pth','--batch','16','--lr','0.0001','--steps','40000','--out','expanded_base_40k'))
 Run-Python 'expanded_base_40k-development' @('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames','C:\lucid\presented-frozen-eval-20260904\frames','--checkpoint','expanded_base_40k','expanded_base_40k/step040000.pth','--checkpoint','big2k','C:\lucid\paired-ladder-20260905-r10\big_2k\step002000.pth','--report','expanded_base_40k-development.json')
 $critic=@('--paired-dino','--dino-gan-weight','0.0075','--pixrestore-repository','C:\lucid\pixrestore-baseline-20260904','--dino-repository','C:\lucid\causal-frontier-20260904-dino\third_party\dinov2','--dino-checkpoint','C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
 Run-Python 'expanded_critic_2k-train' ($base+$critic+@('--bank','I:\lucid-quality-breakthrough-r37\v4-mapped','--init','expanded_base_40k/step040000.pth','--batch','4','--lr','0.00002','--steps','2000','--out','expanded_critic_2k'))
 Run-Python 'expanded_critic_2k-development' @('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames','C:\lucid\presented-frozen-eval-20260904\frames','--checkpoint','expanded_critic_2k','expanded_critic_2k/step002000.pth','--checkpoint','big2k','C:\lucid\paired-ladder-20260905-r10\big_2k\step002000.pth','--report','expanded_critic_2k-development.json')
} finally {$lock.Dispose()}
