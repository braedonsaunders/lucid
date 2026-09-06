#!/bin/bash
# Launch Lucid with a candidate 2x model swapped in, without touching shipping assets.
#
#   Tools/run-candidate.sh [PACKAGES_DIR] [SHARPNESS]
#
# PACKAGES_DIR holds direct2x_trained_<W>x<H>.mlpackage for every ladder size
# (default: .build/ref4k-ladder, the 4,000-step reference-target blend).
# The Release app is copied to .build/candidate-app, the packages are added to
# its Resources, it is re-signed with the same stable identity run-poc.sh uses,
# and launched with LUCID_MODEL_STEM=direct2x_trained_ and a tuning file whose
# sharpness is SHARPNESS (default 0.2, the value used in the native holdout).
# Everything else is the shipping configuration. A/B against shipping by
# quitting and running Tools/run-poc.sh, or flip Enhancement in the lab page.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
packages="${1:-$repo_dir/.build/ref4k-ladder}"
sharpness="${2:-0.2}"
release_app="$repo_dir/.build/paired-release/Build/Products/Release/Lucid.app"
[[ -d "$release_app" ]] || { echo "Release app missing; build with: xcodebuild -project Lucid.xcodeproj -scheme Lucid -configuration Release -derivedDataPath .build/paired-release build CODE_SIGNING_ALLOWED=NO" >&2; exit 1; }
stem="direct2x_trained_"
count=$(ls -d "$packages"/${stem}*.mlpackage 2>/dev/null | wc -l | tr -d ' ')
[[ "$count" -ge 6 ]] || { echo "expected six ${stem}<WxH>.mlpackage in $packages, found $count" >&2; exit 1; }
target="$repo_dir/.build/candidate-app"
rm -rf "$target"; mkdir -p "$target"
cp -R "$release_app" "$target/Lucid.app"
for p in "$packages"/${stem}*.mlpackage; do cp -R "$p" "$target/Lucid.app/Contents/Resources/"; done
identity="$(security find-identity -v -p codesigning 2>/dev/null | grep -o '"Apple Development: [^"]*"' | head -1 | tr -d '"')"
if [[ -n "$identity" ]]; then codesign --force --deep --options runtime --timestamp=none --sign "$identity" "$target/Lucid.app"; else codesign --force --deep --sign - "$target/Lucid.app"; fi
python3 - "$repo_dir/Tools/tuning.json" "$target/tuning.json" "$sharpness" <<'PY'
import json, sys
t = json.load(open(sys.argv[1])); t['sharpness'] = float(sys.argv[3])
json.dump(t, open(sys.argv[2], 'w'), indent=2)
PY
cd "$repo_dir"
if ! lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
  python3 -m http.server 8765 --directory "$repo_dir" >/tmp/lucid-test-server.log 2>&1 &
  sleep 1
fi
echo "candidate app: $target/Lucid.app  stem=$stem  sharpness=$sharpness"
open -a "Google Chrome" "http://127.0.0.1:8765/TestSite/lab.html"
LUCID_MODEL_STEM="$stem" LUCID_TUNING="$target/tuning.json" exec "$target/Lucid.app/Contents/MacOS/Lucid"
