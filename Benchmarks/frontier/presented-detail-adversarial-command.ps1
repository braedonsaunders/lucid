param([ValidateSet('smoke', 'train')][string]$Mode = 'smoke')
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location 'C:\lucid\presented-adversarial-20260904'
$jobs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($jobs) { throw 'Existing Python job found; preserve it' }
$label = if ($Mode -eq 'smoke') { 'smoke' } else { 'adversarial' }
$steps = if ($Mode -eq 'smoke') { '2' } else { '8000' }
foreach ($path in @($label, 'owner.lock')) {
    if (Test-Path $path) { throw "Preserve existing $path" }
}
New-Item -ItemType File 'owner.lock' -ErrorAction Stop | Out-Null
try {
    $env:PYTHONPATH = 'C:\lucid\pixrestore-baseline-20260904\deps'
    $env:TORCHDYNAMO_DISABLE = '1'
    $arguments = @('-u', 'Tools/experiments/train_presented_detail.py', '--bank', 'C:\lucid\causal-pixrestore-distill-20260904\bank', '--pixrestore-cache', 'C:\lucid\causal-pixrestore-distill-20260904\teacher-cache', '--shipping-cache', 'C:\lucid\presented-detail-20260904\shipping-cache', '--init', 'C:\lucid\presented-detail-20260904\shipping.pth', '--batch', '4', '--crop', '96', '--lr', '0.00002', '--seed', '20260914', '--teacher-mix', '0.5', '--steps', $steps, '--out', $label, '--dino-gan-weight', '0.005', '--pixrestore-repository', 'C:\lucid\pixrestore-baseline-20260904', '--dino-repository', 'C:\lucid\causal-frontier-20260904-dino\third_party\dinov2', '--dino-checkpoint', 'C:\lucid\frontier-teachers-20260904\dinov2_vits14_pretrain.pth')
    $job = Start-Process -FilePath 'C:\lucid\.venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -RedirectStandardOutput "$label.log" -RedirectStandardError "$label.err" -Wait -PassThru
    @{ exit_code = $job.ExitCode; finished_utc = (Get-Date).ToUniversalTime().ToString('o') } | ConvertTo-Json | Set-Content "$label-result.json"
    if ($job.ExitCode -ne 0) { throw "$label failed" }
} finally { Remove-Item 'owner.lock' }
