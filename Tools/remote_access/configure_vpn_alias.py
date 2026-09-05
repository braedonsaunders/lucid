#!/usr/bin/env python3
"""Verify enrolled VPN access, then add a separate SSH alias; LAN stays unchanged."""
import argparse
import base64
import datetime
import ipaddress
import json
import pathlib
import shutil
import subprocess

TS = '/Applications/Tailscale.app/Contents/MacOS/Tailscale'

def run(args):
    return subprocess.run(args, check=True, text=True, capture_output=True, timeout=45).stdout

def powershell(script):
    encoded = base64.b64encode(script.encode('utf-16le')).decode()
    return run(['ssh', '-o', 'BatchMode=yes', 'lucid-gpu', 'powershell -NoProfile -EncodedCommand ' + encoded])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    local = json.loads(run([TS, 'status', '--json']))
    remote = json.loads(powershell("& 'C:\\Program Files\\Tailscale\\tailscale.exe' status --json"))
    if any(s['BackendState'] != 'Running' for s in [local, remote]):
        raise SystemExit('Both Mac and GPU PC must be enrolled and connected first.')
    vpn_ip = next(ip for ip in remote['TailscaleIPs'] if ipaddress.ip_address(ip).version == 4)
    cfg = dict(line.split(' ', 1) for line in run(['ssh', '-G', 'lucid-gpu']).splitlines() if ' ' in line)
    options = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias=' + cfg['hostname'],
               '-o', 'IdentitiesOnly=yes', '-i', cfg['identityfile'], '-p', cfg['port']]
    # This reuses the already trusted LAN host key, never accepts a new key blindly.
    run(['ssh', *options, cfg['user'] + '@' + vpn_ip, 'hostname'])
    print(json.dumps({'vpn_ip': vpn_ip, 'dns_name': remote['Self'].get('DNSName'),
                      'mac_ips': local['TailscaleIPs'], 'host_key_continuity': True}, indent=2))
    if args.check_only:
        return
    # Windows block rules override broad preexisting SSH allow rules, but only
    # on the actual Tailscale adapter. LAN SSH/firewall rules remain untouched.
    blocked = []
    for cidr in ['100.64.0.0/10', 'fd7a:115c:a1e0::/48']:
        space = ipaddress.ip_network(cidr)
        owner = [ipaddress.ip_address(ip) for ip in local['TailscaleIPs'] if ipaddress.ip_address(ip) in space]
        if len(owner) != 1:
            raise SystemExit('Expected exactly one owner address per Tailscale address family.')
        host = ipaddress.ip_network(str(owner[0]) + ('/32' if space.version == 4 else '/128'))
        blocked.extend(str(net) for net in space.address_exclude(host))
    ranges = ','.join("'" + cidr + "'" for cidr in blocked)
    firewall = f"""$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'
$adapter=(Get-NetIPAddress -IPAddress '{vpn_ip}' -ErrorAction Stop).InterfaceAlias
if (!$adapter -or @($adapter).Count -ne 1) {{ throw 'Expected one Tailscale adapter' }}
$rule=Get-NetFirewallRule -Name 'Lucid-Tailscale-SSH-OwnerOnly' -ErrorAction SilentlyContinue
if ($rule) {{
  $rule | Set-NetFirewallRule -Enabled True -InterfaceAlias $adapter -RemoteAddress @({ranges})
}} else {{
  New-NetFirewallRule -Name 'Lucid-Tailscale-SSH-OwnerOnly' -DisplayName 'Lucid: SSH only from owner Mac over Tailscale' -Direction Inbound -Action Block -Protocol TCP -LocalPort 22 -RemoteAddress @({ranges}) -InterfaceAlias $adapter -Profile Any | Out-Null
}}
"""
    powershell(firewall)
    run(['ssh', *options, cfg['user'] + '@' + vpn_ip, 'hostname'])
    print('Windows Tailscale SSH restricted to this Mac; LAN unchanged.')
    config = pathlib.Path.home() / '.ssh/config'
    old = config.read_text()
    begin, end = '# BEGIN LUCID REMOTE ACCESS', '# END LUCID REMOTE ACCESS'
    if begin in old:
        first, rest = old.split(begin, 1)
        _, last = rest.split(end, 1)
        old = first + last.lstrip('\n')
    elif 'Host lucid-gpu-vpn' in old or 'Host lucid-gpu-lan' in old:
        raise SystemExit('Unmanaged Lucid aliases already exist; inspect before changing.')
    block = f'''{begin}
Host lucid-gpu-vpn
    HostName {vpn_ip}
    User {cfg['user']}
    Port {cfg['port']}
    IdentityFile {cfg['identityfile']}
    IdentitiesOnly yes
    HostKeyAlias {cfg['hostname']}
    StrictHostKeyChecking yes
    ServerAliveInterval 30
    ServerAliveCountMax 6
Host lucid-gpu-lan
    HostName {cfg['hostname']}
    User {cfg['user']}
    Port {cfg['port']}
    IdentityFile {cfg['identityfile']}
    IdentitiesOnly yes
    HostKeyAlias {cfg['hostname']}
    StrictHostKeyChecking yes
    ServerAliveInterval 30
    ServerAliveCountMax 6
{end}

'''
    backup = config.with_name('config.lucid-backup-' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    shutil.copy2(config, backup)
    config.write_text(block + old)
    config.chmod(0o600)
    print('Verified alias installed; SSH config backup: ' + str(backup))

if __name__ == '__main__':
    main()
