$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location 'C:\lucid\causal-pixrestore-unfiltered-20260904'
$jobs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($jobs) { throw 'Existing Python job found; preserve it' }
foreach ($path in @('unfiltered', 'owner.lock')) {
    if (Test-Path $path) { throw "Preserve existing $path" }
}
New-Item -ItemType File 'owner.lock' -ErrorAction Stop | Out-Null
function Run-Python($label, $arguments) {
    $job = Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
    @{ exit_code = $job.ExitCode; finished_utc = (Get-Date).ToUniversalTime().ToString('o') } | ConvertTo-Json | Set-Content "$label-result.json"
    if ($job.ExitCode -ne 0) { throw "$label failed" }
}
try {
    $env:PYTHONPATH = 'C:\lucid\pixrestore-baseline-20260904\deps'
    $env:TORCHDYNAMO_DISABLE = '1'
    $shared = @('-u', 'Tools/experiments/train_causal_detail.py', '--architecture', 'causal_detail_v2', '--bank', 'C:\lucid\causal-pixrestore-distill-20260904\bank', '--steps', '4000', '--channels', '32', '--blocks', '4', '--batch', '4', '--crop', '96', '--lr', '0.0001', '--seed', '20260913', '--device', 'cuda', '--precision', 'bf16', '--dino-weight', '0.03', '--dino-repository', 'C:\lucid\causal-frontier-20260904-dino\third_party\dinov2', '--dino-checkpoint', 'C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth', '--init', 'C:\lucid\causal-capacity-20260904\control-recovery\step008000.pth', '--teacher-cache', 'C:\lucid\causal-pixrestore-distill-20260904\teacher-cache')
    Run-Python 'unfiltered' ($shared + @('--teacher-policy', 'unfiltered', '--teacher-weight', '3', '--out', 'unfiltered'))
} finally { Remove-Item 'owner.lock' }
