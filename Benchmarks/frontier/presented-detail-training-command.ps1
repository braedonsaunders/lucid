$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location 'C:\lucid\presented-detail-20260904'
$jobs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($jobs) { throw 'Existing Python job found; preserve it' }
foreach ($path in @('smoke', 'candidate', 'control', 'owner.lock')) {
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
    $shared = @('-u', 'Tools/experiments/train_presented_detail.py', '--bank', 'C:\lucid\causal-pixrestore-distill-20260904\bank', '--pixrestore-cache', 'C:\lucid\causal-pixrestore-distill-20260904\teacher-cache', '--shipping-cache', 'shipping-cache', '--init', 'shipping.pth', '--batch', '4', '--crop', '96', '--lr', '0.00002', '--seed', '20260914')
    Run-Python 'smoke' ($shared + @('--steps', '1', '--teacher-mix', '0.5', '--out', 'smoke'))
    Run-Python 'candidate' ($shared + @('--steps', '8000', '--teacher-mix', '0.5', '--out', 'candidate'))
    Run-Python 'control' ($shared + @('--steps', '8000', '--teacher-mix', '0', '--out', 'control'))
} finally { Remove-Item 'owner.lock' }
