# Install without executing: safe while an existing SSH session or training is running.
$ErrorActionPreference = 'Stop'
$root = 'C:\lucid\remote-access'
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
# Owner is already a local administrator; no ordinary-user write access is inherited.
icacls $root /inheritance:r /grant:r "${user}:(OI)(CI)F" '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Directory ACL hardening failed.' }
$allowedSids = @('S-1-5-18','S-1-5-32-544',[Security.Principal.WindowsIdentity]::GetCurrent().User.Value)
foreach ($entry in (Get-Acl $root).Access) {
    $sid = $entry.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
    if ($entry.AccessControlType -eq 'Allow' -and $sid -notin $allowedSids) {
        throw "Unexpected directory ACL principal: $sid. Review before installing a privileged task."
    }
}
$expectedHash = (Get-FileHash "$env:ProgramData\ssh\ssh_host_ed25519_key.pub" -Algorithm SHA256).Hash
$expectedHash | Set-Content (Join-Path $root 'sshd-host-public.sha256') -Encoding ASCII
& (Join-Path $root 'Ensure-LucidSshAtBoot.ps1') -InspectOnly -ExpectedHostKeyHash $expectedHash
# Embed the validated script and public-key digest in Task Scheduler's protected
# action. This prevents writable ancestor-directory replacement from changing
# what SYSTEM executes. Changes to the source require re-running this installer.
$source = Get-Content (Join-Path $root 'Ensure-LucidSshAtBoot.ps1') -Raw
$command = '& {' + $source + '} -ExpectedHostKeyHash ' + $expectedHash
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument ("-NoProfile -NonInteractive -EncodedCommand " + $encoded)
$principal = New-ScheduledTaskPrincipal -UserId SYSTEM -LogonType ServiceAccount -RunLevel Highest
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 3) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'Lucid-EnsureSshAtBoot' -Action $action -Principal $principal -Trigger $trigger -Settings $settings -Description 'Boot-only recovery of original sshd; verifies original host identity and config; never stops active sshd.' -Force | Select TaskName,State
