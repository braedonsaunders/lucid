$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$env:PATH='C:\ffmpeg\bin;'+$env:PATH
Set-Location 'C:\lucid\reds-training-subset-20260905'
# CPU preparation only. Existing folders are preserved by each utility.
& 'C:\lucid\.venv\Scripts\python.exe' -u materialize_reds_sources.py --frames frames --out masters
if ($LASTEXITCODE -ne 0) { throw 'Master verification failed' }
& 'C:\lucid\.venv\Scripts\python.exe' -u build_stream_bank.py --sources masters/sources.json --out bank --patch 256 --frames 16 --widths 1280 --windows-per-source 2 --patches-per-window 4 --seed 20260916
if ($LASTEXITCODE -ne 0) { throw 'Bank construction failed' }
