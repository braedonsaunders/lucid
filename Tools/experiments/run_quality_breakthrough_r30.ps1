$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\quality-breakthrough-20260908-r30'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe')}){throw 'Preserve existing Python jobs'}
$lock=[IO.File]::Open((Join-Path (Get-Location) 'owner.lock'),'CreateNew','ReadWrite','None')
function Run-Python($label,$arguments){
 $p=Start-Process C:\lucid\.venv\Scripts\python.exe -ArgumentList $arguments -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
 @{exitCode=$p.ExitCode;finishedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content "$label-exit.json"
 if($p.ExitCode -ne 0){throw "$label failed"}
}
try {
 $env:PATH='C:\ffmpeg\bin;'+$env:PATH
 if(!(Test-Path 'C:\ffmpeg\bin\ffmpeg.exe')){throw 'Existing FFmpeg installation required'}
 $env:PYTHONPATH='C:\lucid\quality-breakthrough-20260908-r30\Tools;C:\lucid\quality-breakthrough-20260908-r30\Tools\experiments;C:\lucid\frontier-eval-deps-20260904;C:\lucid\pixrestore-baseline-20260904\deps'
 Run-Python 'cleaner-unit-tests' @('-m','unittest','discover','-s','Tools/experiments','-p','test_precleaner.py')
 $base=@('-u','Tools/experiments/train_precleaner.py','--bank','C:\lucid\stream-bank-v4','--clean-cache','C:\lucid\clean-lr-bank-v4-r30','--init','C:\lucid\paired-ladder-20260905-r10\big_2k\step002000.pth','--batch','8','--crop','96','--channels','16','--lr','0.001','--seed','20260914','--steps','2000')
 foreach($name in @('cleaner_raw_2k','cleaner_native_input_2k')){
  $arguments=$base+@('--out',$name)
  if($name -eq 'cleaner_native_input_2k'){$arguments+=@('--native-input-stages')}
  Run-Python "$name-train" $arguments
  Run-Python "$name-development" @('-u','Tools/frontier_eval/score_checkpoint_frames.py','--device','cuda','--frames','C:\lucid\presented-frozen-eval-20260904\frames','--checkpoint',$name,"$name/step002000.pth",'--checkpoint','big2k','C:\lucid\paired-ladder-20260905-r10\big_2k\step002000.pth','--report',"$name-development.json")
 }
} finally {$lock.Dispose()}
