[CmdletBinding()]
param([Parameter(Mandatory)][string]$JobDirectory)
$ErrorActionPreference = 'Stop'
$manifest = Get-Content -LiteralPath (Join-Path $JobDirectory 'manifest.json') -Raw | ConvertFrom-Json
$statusPath = Join-Path $JobDirectory 'status.json'
$started = [DateTime]::UtcNow.ToString('o')
function Write-Status([string]$State, [int]$Code, [string]$Detail = '') {
    @{state=$State; exitCode=$Code; pid=$PID; startedAt=$started; updatedAt=[DateTime]::UtcNow.ToString('o'); detail=$Detail} |
        ConvertTo-Json | Set-Content -LiteralPath "$statusPath.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$statusPath.tmp" -Destination $statusPath -Force
}
$lock = $null
try {
    $lock = [IO.File]::Open((Join-Path $JobDirectory 'supervisor.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    Write-Status 'running' 0
    Set-Location -LiteralPath $manifest.workingDirectory
    $jobScript = Join-Path $JobDirectory 'job.ps1'
    # The fresh child process gives each job an isolated PowerShell scope and exit code.
    & "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $jobScript *> (Join-Path $JobDirectory 'output.log')
    $code = $LASTEXITCODE
    if ($code -ne 0) { Write-Status 'failed' $code } else { Write-Status 'completed' 0 }
    exit $code
} catch {
    if ($null -ne $lock) { Write-Status 'failed' 1 $_.Exception.Message }
    throw
} finally {
    if ($null -ne $lock) { $lock.Dispose() }
}
