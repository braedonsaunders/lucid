$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\quality-breakthrough-20260908-r29'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe')}){throw 'Preserve existing Python jobs'}
$lock=[IO.File]::Open((Join-Path (Get-Location) 'owner.lock'),'CreateNew','ReadWrite','None')
function Run-Python($label,$arguments){
 $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
 @{exitCode=$p.ExitCode;finishedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content "$label-exit.json"
 if($p.ExitCode -ne 0){throw "$label failed"}
}
function Evaluate($n,$final,$ema){
 foreach($mode in @('development','bank')){
  $frames=if($mode -eq 'development'){'C:\lucid\presented-frozen-eval-20260904\frames'}else{'C:\lucid\anchored-detail-20260905\bank-validation-frames'}
  $evalArgs=@('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames',$frames,'--checkpoint',"${n}_raw",$final,'--checkpoint','shipping','C:\lucid\presented-detail-20260904\shipping.pth','--present-4x-at-2x','shipping','--report',"$n-$mode.json")
  if($ema){$evalArgs += @('--checkpoint',"${n}_ema",$ema)}
  Run-Python "$n-$mode" $evalArgs
 }
}
try {
 $env:PYTHONPATH='C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
 Run-Python 'ldl-unit-tests' @('-m','unittest','discover','-s','Tools/experiments','-p','test_ldl.py')
 Run-Python 'stage-unit-tests' @('-m','unittest','discover','-s','Tools/experiments','-p','test_native_stages.py')
 $base=@('-u','Tools/experiments/train_presented_detail.py','--bank','C:\lucid\stream-bank-v4','--pixrestore-cache','none','--shipping-cache','none','--init','C:\lucid\presented-detail-20260904\shipping.pth','--batch','4','--crop','96','--lr','0.00002','--seed','20260914','--teacher-mix','0','--intended','reference','--architecture','coupled','--detail-target','reference')
 $critic=@('--paired-dino','--dino-gan-weight','0.005','--pixrestore-repository','C:\lucid\pixrestore-baseline-20260904','--dino-repository','C:\lucid\causal-frontier-20260904-dino\third_party\dinov2','--dino-checkpoint','C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
 # Output-space loss ablation; all arms share 3-frame batches and the same v4 init.
 $c75=$critic -replace '^0.005$','0.0075'
 $v4=$base -replace 'C:\\lucid\\presented-detail-20260904\\shipping.pth','C:\lucid\paired-ladder-20260905-r17\v4_w0075_2k\step002000.pth'
 Run-Python 'output-stage-unit-tests' @('-m','unittest','discover','-s','Tools/experiments','-p','test_native_output_stages.py')
 Run-Python 'source_fp16_1k-train' ($v4+$c75+@('--training-frames','3','--native-input-stages','--steps','1000','--out','source_fp16_1k'))
 Evaluate 'source_fp16_1k' 'source_fp16_1k/step001000.pth' $null
 Run-Python 'output_stages_1k-train' ($v4+$c75+@('--training-frames','3','--native-output-stages','--steps','1000','--out','output_stages_1k'))
 Evaluate 'output_stages_1k' 'output_stages_1k/step001000.pth' $null
 Run-Python 'full_stages_1k-train' ($v4+$c75+@('--training-frames','3','--native-input-stages','--native-output-stages','--steps','1000','--out','full_stages_1k'))
 Evaluate 'full_stages_1k' 'full_stages_1k/step001000.pth' $null
} finally {$lock.Dispose()}
