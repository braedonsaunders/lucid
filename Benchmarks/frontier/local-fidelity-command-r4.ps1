$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\local-fidelity-20260905-r4'
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe')}){throw 'Preserve existing Python jobs'}
if(Test-Path owner.lock){throw 'Existing owner lock'}
New-Item -ItemType File owner.lock -ErrorAction Stop | Out-Null
try {
    $env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
    $env:PYTHONPATH='C:\lucid\pixrestore-baseline-20260904\deps'
    $preflight=Start-Process -FilePath C:\lucid\.venv\Scripts\python.exe -ArgumentList Tools/experiments/check_deterministic_resize_cuda.py -RedirectStandardOutput resize-preflight.log -RedirectStandardError resize-preflight.err -Wait -PassThru
    if($preflight.ExitCode -ne 0){throw 'Deterministic resize preflight failed'}
    $common=@('--deterministic','--mode','dynamic','--bank','C:\lucid\presented-diversity-20260905\data\bank','--shipping-cache','C:\lucid\presented-diversity-20260905\shipping-cache','--pixrestore-cache','C:\lucid\presented-diversity-20260905\data\teacher-cache','--init','C:\lucid\presented-detail-20260904\shipping.pth','--probe','probe.json','--pixrestore-repository','C:\lucid\pixrestore-baseline-20260904','--dino-repository','C:\lucid\causal-frontier-20260904-dino\third_party\dinov2','--dino-checkpoint','C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
    foreach($label in @('unconstrained-smoke','unconstrained-repeat','constrained-smoke','unconstrained','constrained')){
        if(Test-Path $label){throw "Preserve existing $label"}
        $steps=if($label -in @('unconstrained','constrained')){8000}else{2}
        $arguments=@('-u','Tools/experiments/train_activation_control.py','--out',$label,'--steps',"$steps")+$common
        if($label -in @('constrained-smoke','constrained')){$arguments+='--local-fidelity-constraint'}
        $job=Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
        @{exit_code=$job.ExitCode;finished_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content "$label-result.json"
        if($job.ExitCode -ne 0){throw "$label failed"}
        $prior=Get-Content 'unconstrained-smoke/experiment.json' -Raw | ConvertFrom-Json
        $actual=Get-Content "$label/experiment.json" -Raw | ConvertFrom-Json
        foreach($field in @('bank_sha256','checkpoint_sha256','teacher_manifest_sha256','shipping_manifest_sha256','probe_sha256','frozen_anchor_and_masks_sha256','controller_initial_sha256','discriminator_initial_sha256','first_batch_sha256','first_output_sha256')){
            if($actual.$field -ne $prior.$field){throw "Changed $label identity $field"}
        }
        if($label -eq 'unconstrained-repeat'){
            $expected=Get-Content 'unconstrained-smoke/complete.json' -Raw | ConvertFrom-Json
            $actual=Get-Content "$label/complete.json" -Raw | ConvertFrom-Json
            if($actual.controller_final_sha256 -ne $expected.controller_final_sha256){throw 'Unconstrained control update changed'}
        }
    }
} finally {Remove-Item owner.lock}
