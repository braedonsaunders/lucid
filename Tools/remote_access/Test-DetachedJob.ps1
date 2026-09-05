# CPU-only proof; intended to finish after the initiating SSH connection closes.
$ErrorActionPreference = 'Stop'
$started = [DateTime]::UtcNow.ToString('o')
Start-Sleep -Seconds 20
[pscustomobject]@{startedAt=$started; completedAt=[DateTime]::UtcNow.ToString('o'); user=[Security.Principal.WindowsIdentity]::GetCurrent().Name; proof='survived-ssh-disconnect'; gpuUsed=$false} | ConvertTo-Json
exit 0
