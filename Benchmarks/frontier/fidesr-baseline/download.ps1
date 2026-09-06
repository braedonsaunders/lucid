$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location C:\lucid\fidesr-20260905-r1
$assets=Get-Content assets.json -Raw | ConvertFrom-Json
foreach($a in $assets){
 $destination=Join-Path (Get-Location) $a.destination
 if(Test-Path $destination){throw "Preserve existing destination $destination"}
 New-Item -ItemType Directory (Split-Path $destination) -Force | Out-Null
 & curl.exe --fail --location --silent --show-error --connect-timeout 20 --max-time 600 --output "$destination.partial" $a.url
 if($LASTEXITCODE -ne 0){throw "Download failed: $($a.path)"}
 if((Get-Item "$destination.partial").Length -ne $a.size){throw "Size differs: $($a.path)"}
 if($a.sha256 -and (Get-FileHash "$destination.partial" -Algorithm SHA256).Hash -ne $a.sha256){throw "Hash differs: $($a.path)"}
 Move-Item "$destination.partial" $destination
 Write-Output "verified $($a.destination)"
}
$assets | ForEach-Object { @{destination=$_.destination;sha256=(Get-FileHash $_.destination -Algorithm SHA256).Hash.ToLower();size=(Get-Item $_.destination).Length} } | ConvertTo-Json | Set-Content downloaded.json
