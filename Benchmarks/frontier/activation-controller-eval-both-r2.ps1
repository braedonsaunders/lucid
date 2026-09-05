$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Set-Location 'C:\lucid\activation-controller-20260905-r2'
foreach($mode in @('development','bank')) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File evaluate.ps1 -Mode $mode
    if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
}
exit 0
