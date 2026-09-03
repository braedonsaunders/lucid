#!/bin/zsh
set -euo pipefail

repo_dir="${0:A:h:h}"
server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$repo_dir"
python3 -m http.server 8765 --directory "$repo_dir" >/tmp/lucid-test-server.log 2>&1 &
server_pid=$!
sleep 1

xcodebuild \
  -project Lucid.xcodeproj \
  -scheme Lucid \
  -configuration Debug \
  -derivedDataPath .build/DerivedData \
  build \
  CODE_SIGNING_ALLOWED=NO \
  SDK_STAT_CACHE_ENABLE=NO \
  COMPILATION_CACHE_ENABLE_CACHING=NO

app_path="$repo_dir/.build/DerivedData/Build/Products/Debug/Lucid.app"

# A stable signing identity keeps the Screen Recording grant across rebuilds;
# ad-hoc signatures change every build and macOS asks again each time.
identity="$(security find-identity -v -p codesigning 2>/dev/null | grep -o '"Apple Development: [^"]*"' | head -1 | tr -d '"')"
if [[ -n "$identity" ]]; then
  codesign --force --deep --options runtime --timestamp=none --sign "$identity" "$app_path"
else
  codesign --force --deep --sign - "$app_path"
fi
open -a "Google Chrome" "http://127.0.0.1:8765/TestSite/lab.html"
"$app_path/Contents/MacOS/Lucid"
