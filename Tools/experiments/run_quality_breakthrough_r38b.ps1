$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\quality-breakthrough-20260908-r38b'
$unexpected=Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe') -and $_.CommandLine -notlike '*fetch_reds_subset.py*--sequences 60 --frames 100 --workers 4 --grouped-ranges*' -and $_.CommandLine -notlike '*materialize_reds_sources.py*--frames reds-frames --out reds-masters*'}
if($unexpected){throw 'Preserve other Python jobs; only the owned r37 CPU corpus jobs may overlap'}
$lock=[IO.File]::Open((Join-Path (Get-Location) 'owner.lock'),'CreateNew','ReadWrite','None')
function Run-Python($label,$arguments){
 $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
 @{exitCode=$p.ExitCode;finishedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content "$label-exit.json"
 if($p.ExitCode -ne 0){throw "$label failed"}
}
try {
 $env:PYTHONPATH='C:\lucid\quality-breakthrough-20260908-r38b\Tools;C:\lucid\quality-breakthrough-20260908-r38b\Tools\experiments;C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
 if((Get-PSDrive I).Free -lt 80GB){throw '80 GiB SSD headroom required'}
 Run-Python 'v4-ssd-mapped' @('-u','Tools/experiments/mmap_training_bank.py','--bank','C:\lucid\stream-bank-v4','--out','I:\lucid-quality-breakthrough-r37\v4-mapped')
 if((Get-FileHash I:\lucid-quality-breakthrough-r37\v4-mapped\manifest.json).Hash -ne (Get-FileHash D:\lucid-quality-breakthrough-r37\v4-mapped\manifest.json).Hash){throw 'storage relocation changed bank identity'}
 Run-Python 'mapped-unit-tests' @('-m','unittest','discover','-s','Tools/experiments','-p','test_mmap_training_bank.py')
 $base=@('-u','Tools/experiments/train_presented_detail.py','--bank','I:\lucid-quality-breakthrough-r37\v4-mapped','--pixrestore-cache','none','--shipping-cache','none','--crop','96','--seed','20260918','--teacher-mix','0','--intended','reference','--architecture','coupled','--detail-target','reference')
 # Matched base control: 40k reconstruction steps from the folded pretrained SPAN.
 Run-Python 'control_base_40k-train' ($base+@('--init','C:\lucid\presented-detail-20260904\shipping.pth','--batch','16','--lr','0.0001','--steps','40000','--out','control_base_40k'))
 Run-Python 'control_base_40k-development' @('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames','C:\lucid\presented-frozen-eval-20260904\frames','--checkpoint','control_base_40k','control_base_40k/step040000.pth','--checkpoint','big2k','C:\lucid\paired-ladder-20260905-r10\big_2k\step002000.pth','--report','control_base_40k-development.json')
 $critic=@('--paired-dino','--dino-gan-weight','0.0075','--pixrestore-repository','C:\lucid\pixrestore-baseline-20260904','--dino-repository','C:\lucid\causal-frontier-20260904-dino\third_party\dinov2','--dino-checkpoint','C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
 Run-Python 'control_critic_2k-train' ($base+$critic+@('--init','control_base_40k/step040000.pth','--batch','4','--lr','0.00002','--steps','2000','--out','control_critic_2k'))
 Run-Python 'control_critic_2k-development' @('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames','C:\lucid\presented-frozen-eval-20260904\frames','--checkpoint','control_critic_2k','control_critic_2k/step002000.pth','--checkpoint','big2k','C:\lucid\paired-ladder-20260905-r10\big_2k\step002000.pth','--report','control_critic_2k-development.json')
} finally {$lock.Dispose()}
