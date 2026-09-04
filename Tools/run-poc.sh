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
# The learned upscaler's models. Compiled here so the app does not pay for
# compilation on its first frame, and only the sizes it can actually use.
#
# The sizes are listed rather than globbed, and the stem is ch32u rather than
# ch28. Model/ also holds the ladders that were converted only to measure
# against, and a glob picked those up; worse, this script went on compiling ch28
# after the app moved to ch32u, so LearnedUpscaler.model() looked for a resource
# that was never installed and threw Failure.noModel on every session. There is
# no fallback upscaler by design, so the app came up, connected, reported
# healthy, and enhanced nothing.
#
# This list is LearnedUpscaler.variants, and Tools/release.sh carries the same
# one - keep all three in step.
resources="$app_path/Contents/Resources"
mkdir -p "$resources"
missing=0
for size in 256x144 320x180 432x240 480x270 640x360 864x480; do
  model="$repo_dir/Model/SPAN_x4_ch32u_$size.mlpackage"
  if [[ ! -d "$model" ]]; then
    echo "  missing $(basename "$model")"
    missing=1
    continue
  fi
  name="$(basename "${model%.mlpackage}")"
  if [[ ! -d "$resources/$name.mlmodelc" ]]; then
    xcrun coremlcompiler compile "$model" "$resources" >/dev/null 2>&1 || true
  fi
done
if (( missing )); then
  echo "⚠️  Some models are missing, so those sizes will not be enhanced at all."
  echo "    The six shipping packages are tracked in git (3 MB); a clean checkout has them."
fi

identity="$(security find-identity -v -p codesigning 2>/dev/null | grep -o '"Apple Development: [^"]*"' | head -1 | tr -d '"')"
if [[ -n "$identity" ]]; then
  codesign --force --options runtime --timestamp=none --sign "$identity" "$app_path"
else
  codesign --force --sign - "$app_path"
fi
open -a "Google Chrome" "http://127.0.0.1:8765/TestSite/lab.html"
"$app_path/Contents/MacOS/Lucid"
