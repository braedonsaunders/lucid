# Launch a local PowerShell job independently of the initiating SSH session.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$')][string]$Name,
    [Parameter(Mandatory)][string]$ScriptPath,
    [string]$WorkingDirectory = 'C:\lucid',
    [string]$Root = 'C:\lucid\remote-access'
)
$ErrorActionPreference = 'Stop'
$script = (Resolve-Path -LiteralPath $ScriptPath).Path
$working = (Resolve-Path -LiteralPath $WorkingDirectory).Path
if ([IO.Path]::GetExtension($script) -ne '.ps1') { throw 'ScriptPath must be a PowerShell .ps1 file.' }
$jobDir = Join-Path (Join-Path $Root 'jobs') $Name
$taskName = "LucidJob-$Name"
if ((Test-Path -LiteralPath $jobDir) -or (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue)) {
    throw "Job already exists: $Name. Choose a new unique name."
}
$supervisor = Join-Path $Root 'Invoke-LucidJob.ps1'
if (!(Test-Path -LiteralPath $supervisor)) { throw "Missing supervisor: $supervisor" }
New-Item -ItemType Directory -Path $jobDir -Force | Out-Null
Copy-Item -LiteralPath $script -Destination (Join-Path $jobDir 'job.ps1')
@{name=$Name; workingDirectory=$working; submittedAt=[DateTime]::UtcNow.ToString('o'); sourceScript=$script} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $jobDir 'manifest.json') -Encoding UTF8
# S4U stores no password, runs without an interactive login, and has local-file access.
# Network resources and per-user encrypted files are intentionally unavailable.
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Highest
$args = '-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{0}" -JobDirectory "{1}"' -f $supervisor,$jobDir
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument $args -WorkingDirectory $working
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Description 'Lucid detached local job; explicit launch only; no automatic reboot replay.' | Out-Null
Start-ScheduledTask -TaskName $taskName
[pscustomobject]@{task=$taskName; directory=$jobDir; user=$user; status='submitted'} | ConvertTo-Json
