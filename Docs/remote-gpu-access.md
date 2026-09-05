# Remote GPU access

Status as of 2026-09-05: both Tailscale clients are installed and the Mac's native VPN menu is enabled. **Owner account enrollment is still required on both devices.** VPN SSH, off-LAN reachability, device-key expiry, and reboot recovery are not yet verified. Existing `ssh lucid-gpu` remains the LAN route to `192.168.68.85` as `bsaun` using `~/.ssh/lucid_gpu`.

## Installed configuration

- Mac: standalone Tailscale 1.102.3 at `/Applications/Tailscale.app`, bundle `io.tailscale.ipn.macsys`, enabled network system extension, native VPN profile `Tailscale` (`223ABF95-B90B-47DD-9ECF-96E63230169F`). System Settings → Menu Bar → VPN is enabled. System Settings → VPN visibly contains its native switch.
- PC: `C:\Program Files\Tailscale\tailscale.exe` 1.102.3; automatic, running Windows service. Preferences verified: hostname `lucid-gpu`, `ForceDaemon=true` (unattended), `WantRunning=true`, DNS/route acceptance false, no advertised routes or exit node.
- Remote tooling: `C:\lucid\remote-access`; owner, Administrators and SYSTEM have access. Existing training/data directories and LAN SSH alias were preserved.
- The Mac VPN connection can say “Connected” while Tailscale still says `NeedsLogin`. This means the native extension is running, not that remote access is ready.

The [standalone macOS variant](https://tailscale.com/docs/concepts/macos-variants) integrates with the native VPN system. A bare `tailscaled` install does not provide this UI. Windows [unattended mode](https://tailscale.com/docs/how-to/run-unattended) makes the node available without an interactive desktop login.

## Finish enrollment and verify the VPN route

1. Sign in to Tailscale on both devices using the same owner account. Short-lived enrollment links are supplied in the conversation, never committed here. If a link expires, run `Tailscale login --timeout=10s` using the executable paths above (PowerShell needs `&` before its quoted executable path). Do not use `--force-reauth` on a working remote connection.
2. In the [Machines console](https://login.tailscale.com/admin/machines), find `lucid-gpu` and choose its menu → **Disable Key Expiry**. Verify the resulting non-expiring device status. Tailscale's default for a new domain is 180 days; unattended mode alone does not remove expiry. Keep routine expiry on portable clients and reauthenticate the Mac when prompted. No reusable enrollment key or API token is needed. See [device key expiry](https://tailscale.com/docs/features/access-control/key-expiry).
3. While LAN access is available, from this repository run:

   ```sh
   python3 Tools/remote_access/configure_vpn_alias.py
   ssh lucid-gpu-vpn hostname
   ssh lucid-gpu-vpn nvidia-smi
   ```

   The script requires both clients to report `Running` and verifies VPN SSH against the already-trusted LAN host key. It restricts Tailscale TCP/22 to this Mac's exact IPv4/IPv6 addresses using Windows firewall rule `Lucid-Tailscale-SSH-OwnerOnly`, then retests SSH. It does not modify existing LAN firewall rules. The rule blocks other tailnet source addresses only on the actual Tailscale adapter; another owner device must be explicitly added before it can use SSH. Existing OpenSSH authentication remains required. Tailnet-wide sharing/policy has not been inspected or changed.

   Only after verification does the script back up `~/.ssh/config` and add `lucid-gpu-vpn` (stable Tailscale IP) and `lucid-gpu-lan` (existing LAN IP). `lucid-gpu` stays unchanged. `--check-only` verifies enrollment/host identity without changing aliases/firewall. Until enrollment is complete, no VPN alias or firewall rule is created.
4. Verify an SCP hash roundtrip over `lucid-gpu-vpn`, disconnect/reconnect Tailscale from the native menu, and retry SSH. Finally test from a genuinely different network, such as a phone hotspot. A VPN-IP test while on home Wi-Fi is not an off-LAN test.

Use the native VPN menu → Tailscale for normal connect/disconnect. CLI equivalents:

```sh
scutil --nc stop Tailscale
scutil --nc start Tailscale
/Applications/Tailscale.app/Contents/MacOS/Tailscale status
```

Only Tailscale was cycled during setup; LAN SSH still returned `DESKTOP-CR5L3VD`. Existing Rassaun VPN profiles were unchanged. No public router port forwarding, subnet router, or exit node was added.

## Jobs that survive SSH disconnection

Write a job wrapper on the PC. Every native command must have its exit code checked or propagated; Windows PowerShell does not automatically fail the wrapper when Python returns nonzero.

```powershell
$ErrorActionPreference = 'Stop'
Set-Location C:\lucid\my-experiment
& C:\lucid\.venv\Scripts\python.exe train.py --config experiment.json
exit $LASTEXITCODE
```

Launch it using a unique job name:

```sh
ssh lucid-gpu-vpn 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\lucid\remote-access\Start-LucidJob.ps1 -Name my-experiment-001 -ScriptPath C:\lucid\my-experiment\job.ps1 -WorkingDirectory C:\lucid\my-experiment'
```

Use `lucid-gpu` instead until VPN enrollment finishes. The launcher copies the wrapper into `C:\lucid\remote-access\jobs\<name>`, registers `LucidJob-<name>`, starts it and returns immediately. Windows Task Scheduler owns the supervisor, so closing SSH or sleeping the Mac does not end the job. It runs as `DESKTOP-CR5L3VD\bsaun` using S4U, stores no password, and does not require a desktop login. Local files are accessible; network shares and per-user encrypted files are not available under S4U.

Inspect `status.json` and `output.log` in the job directory. Status records `running`, `completed` or `failed`, exit code, PID and timestamps. Task Scheduler's `LastTaskResult` also contains the exit code. Duplicate job names are rejected. The supervisor lock applies only to that job: **the experiment must enforce exclusive GPU ownership itself**. Wrappers must wait for child processes rather than spawning untracked background work.

There is no execution timeout and no automatic reboot replay. SSH disconnection is survivable; PC reboot, shutdown or power loss stops computation. Resume a checkpoint explicitly with a new job name. After a crash, a stale `running` status can remain; check Task Scheduler/process state before resuming. Completed task entries can be unregistered without removing their files:

```powershell
Unregister-ScheduledTask -TaskName LucidJob-my-experiment-001 -Confirm:$false
```

## SSH startup recovery

The preexisting running `sshd` service is Manual and reports Windows error 1072, “marked for deletion,” when asked to change startup mode. It was not stopped or restarted. Its configuration validates with the original `C:\Windows\System32\OpenSSH\sshd.exe -t`, and its existing host keys were preserved.

`Lucid-EnsureSshAtBoot` is installed as a boot-triggered SYSTEM task. It embeds the recovery script and expected public-host-key digest inside Task Scheduler's protected action, so replacing a source file or its ancestor directory cannot change the SYSTEM code. `Install-BootRecovery.ps1` also reapplies the directory ACL. Reinstall the task after an intentional change to its source or SSH identity.

At boot, it validates the same host identity/configuration, recreates the original `sshd` service only if missing, sets automatic startup and starts it only if stopped. It never stops an existing service. Three retries are configured. It has **not been executed**, and recovery across a real reboot remains untested. Its read-only inspection passed. After a planned reboot, check `Get-Service sshd,Tailscale` and `Get-ScheduledTaskInfo Lucid-EnsureSshAtBoot` locally or through restored SSH. Do the first reboot test with local recovery available.

Rollback the recovery task with:

```powershell
Unregister-ScheduledTask -TaskName Lucid-EnsureSshAtBoot -Confirm:$false
```

This leaves the preexisting SSH service untouched. Its pending deletion must then be repaired manually before relying on reboot persistence. To undo a future VPN restriction, remove only `Lucid-Tailscale-SSH-OwnerOnly` using `Remove-NetFirewallRule -Name Lucid-Tailscale-SSH-OwnerOnly`; to undo aliases restore the timestamped SSH config backup or remove its marked Lucid block.

## Power and availability

The active custom power plan is `6fecc5ae-f350-48a5-b669-b472cb895ccf`. `powercfg /query` fails on this unnamed plan; a direct registry read confirms its AC standby timeout (`29f6c1db-86da-48c5-9fdb-f2b67b1f44da`) is `0`, meaning no idle sleep. No power setting was changed. Hibernate timing, firmware power recovery and behavior after outages are unverified. A powered-off, sleeping or disconnected PC cannot be reached through Tailscale. Wake-on-LAN from outside the home would need a separate always-on relay; none was configured. Check BIOS restore-on-AC-power behavior locally if outage recovery is required.

## Verification receipts

| Check | Actual result |
|---|---|
| macOS ZIP SHA256 | `8b43c7b864dd1d60301ae4d3dc5cfecbc9a8f22cafc4a36733cf862b571d5854`, matches vendor `.sha256` |
| macOS code signature | Valid; Tailscale team `W5364U7YZB`; extension activated/enabled |
| Windows MSI SHA256 | `03ac8183c6e3ce276e9b44281ebe7e4c02aef28a971034ca170c4b665df42dce`, matches vendor `.sha256` |
| Windows MSI signature | Valid Authenticode, Tailscale signer checked before installation |
| Native VPN | Toggle visibly present; menu-bar VPN checked; native disconnect/reconnect succeeded |
| LAN after disconnect | `ssh lucid-gpu hostname` → `DESKTOP-CR5L3VD` |
| Detached CPU success | `LucidJob-access-proof-20260905`; 14:30:54Z–14:31:14Z; exit 0 after launching SSH closed; Scheduler result 0 |
| Success output SHA256 | `400523b7a6b6dcf9269dd62d95382061561330c4073aed3865e3e458f07e997a` |
| Detached CPU failure | `LucidJob-access-failure-proof-20260905`; failed/exit 7; Scheduler result 7 |
| Failure status SHA256 | `c5f43b2d2e6526016686775e0380e9da9ebde98818be3d412a256d3ecfa71be0` |
| Actual detached CUDA job | Verified `LucidJob-subspace-development-20260905` completed exit 0 after initiating SSH closed; 14:34:18Z–14:35:16Z; RTX4080, 48 full-frame pairs |
| Boot recovery | Installed/Ready, SYSTEM + boot trigger, config/key inspection passed; reboot not tested |
| VPN SSH / SCP / off-LAN | Pending owner enrollment; not claimed as passed |
| Device expiry | Pending console inspection and disabling expiry for the trusted GPU node |

Installer downloads came from [Tailscale's stable package index](https://pkgs.tailscale.com/stable/). Installation used `/norestart` and `REBOOT=ReallySuppress`. Existing GPU training processes were preserved; the access setup's test jobs were CPU-only.
