#!/bin/bash
# Launch Lucid with the shipping model plus the candidate model families bundled,
# so the lab page's Model row can switch between them while a clip plays.
#
#   Tools/run-lab.sh
#
# Bundles .build/ref2k-ladder as lucid2k_<WxH> and .build/ref2k-raw-ladder as
# lucid2kraw_<WxH> into a copy of the Release app, re-signs it with the same
# stable identity run-poc.sh uses, and starts it on the shipping model.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
release_app="$repo_dir/.build/paired-release/Build/Products/Release/Lucid.app"
[[ -d "$release_app" ]] || { echo "Release app missing; build with: xcodebuild -project Lucid.xcodeproj -scheme Lucid -configuration Release -derivedDataPath .build/paired-release build CODE_SIGNING_ALLOWED=NO" >&2; exit 1; }
target="$repo_dir/.build/lab-app"
rm -rf "$target"; mkdir -p "$target"
cp -R "$release_app" "$target/Lucid.app"
add() { # stem source-dir
  local n=0
  for p in "$2"/direct2x_trained_*.mlpackage; do
    local size="${p##*direct2x_trained_}"; size="${size%.mlpackage}"
    cp -R "$p" "$target/Lucid.app/Contents/Resources/$1${size}.mlpackage"; n=$((n+1))
  done
  [[ "$n" -ge 6 ]] || { echo "expected six packages in $2, found $n" >&2; exit 1; }
}
add lucid2k_ "$repo_dir/.build/ref2k-ladder"
add lucid2kraw_ "$repo_dir/.build/ref2k-raw-ladder"
identity="$(security find-identity -v -p codesigning 2>/dev/null | grep -o '"Apple Development: [^"]*"' | head -1 | tr -d '"')"
if [[ -n "$identity" ]]; then codesign --force --deep --options runtime --timestamp=none --sign "$identity" "$target/Lucid.app"; else codesign --force --deep --sign - "$target/Lucid.app"; fi
cd "$repo_dir"
if ! lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
  python3 -m http.server 8765 --directory "$repo_dir" >/tmp/lucid-test-server.log 2>&1 &
  sleep 1
fi
echo "lab app: $target/Lucid.app  (models: shipping, lucid2k_, lucid2kraw_)"
open -a "Google Chrome" "http://127.0.0.1:8765/TestSite/lab.html"
exec "$target/Lucid.app/Contents/MacOS/Lucid"
