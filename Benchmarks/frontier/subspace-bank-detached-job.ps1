$ErrorActionPreference='Stop'
& "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -File 'C:\lucid\subspace-adapter-20260905\evaluate.ps1' -Mode bank
exit $LASTEXITCODE
