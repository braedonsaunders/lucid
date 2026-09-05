# Boot-only recovery for the preexisting sshd service marked for deletion.
# Never stops a process/service; preserves the original OpenSSH config and host keys.
[CmdletBinding()]
param([switch]$InspectOnly, [Parameter(Mandatory)][ValidatePattern('^[0-9A-Fa-f]{64}$')][string]$ExpectedHostKeyHash)
$ErrorActionPreference = 'Stop'
$root = 'C:\lucid\remote-access'
$binary = "$env:SystemRoot\System32\OpenSSH\sshd.exe"
$config = "$env:ProgramData\ssh\sshd_config"
$hostKey = "$env:ProgramData\ssh\ssh_host_ed25519_key.pub"
if (!(Test-Path -LiteralPath $binary) -or !(Test-Path -LiteralPath $config)) { throw 'Original OpenSSH installation is missing.' }
if ((Get-FileHash -LiteralPath $hostKey -Algorithm SHA256).Hash -ne $ExpectedHostKeyHash) { throw 'SSH host identity changed; refusing unattended repair.' }
& $binary -t -f $config
if ($LASTEXITCODE -ne 0) { throw 'Original sshd configuration does not validate.' }
$service = Get-Service sshd -ErrorAction SilentlyContinue
if ($InspectOnly) {
    [pscustomobject]@{configurationValid=$true; identityMatches=$true; serviceState="$($service.Status)"; changesMade=$false} | ConvertTo-Json
    return
}
& {
    if (!$service) {
        New-Service -Name sshd -BinaryPathName $binary -DisplayName 'OpenSSH SSH Server' -StartupType Automatic | Out-Null
    }
    Set-Service -Name sshd -StartupType Automatic
    if ((Get-Service sshd).Status -eq 'Stopped') { Start-Service sshd }
    if ((Get-Service sshd).Status -ne 'Running') { throw 'sshd did not reach Running state.' }
    Get-Service sshd | Select Name,Status,StartType
}
