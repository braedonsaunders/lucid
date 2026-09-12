#!/bin/zsh
# Build a verified distribution. --local explicitly permits development signing.
set -euo pipefail
version="${1:-}"
mode="${2:-}"
if [[ ! "$version" =~ '^[0-9]+\.[0-9]+\.[0-9]+$' ]] || [[ -n "$mode" && "$mode" != '--local' ]]; then
  echo 'usage: Tools/release.sh <major.minor.patch> [--local]' >&2
  exit 2
fi
repo="${0:A:h:h}"
cd "$repo"
build="$repo/.build/release"
app="$build/Build/Products/Release/Lucid.app"
identity="$(security find-identity -v -p codesigning | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' | head -1)"
suffix=''
if [[ "$mode" == '--local' ]]; then
  suffix='-local'
  if [[ -z "$identity" ]]; then
    identity="$(security find-identity -v -p codesigning | sed -n 's/.*"\(Apple Development: [^"]*\)".*/\1/p' | head -1)"
  fi
else
  [[ -n "$identity" ]] || { echo 'Production release blocked: Developer ID Application certificate is missing.' >&2; exit 1; }
  xcrun notarytool history --keychain-profile LUCID_NOTARY >/dev/null
fi
[[ -n "$identity" ]] || { echo 'No signing identity available.' >&2; exit 1; }
python3 Tools/sync_safari.py --check
python3 Tools/package_models.py
python3 - "$version" <<'PY'
import json,sys
from pathlib import Path
version=sys.argv[1]
assert json.loads(Path('BrowserExtension/manifest.json').read_text())['version']==version, 'Companion version differs'
assert json.loads(Path('Lucid/Resources/Models.json').read_text())['release']==version, 'Model release differs'
PY
mkdir -p "$repo/.build"
dmg="$repo/.build/Lucid-$version$suffix.dmg"
[[ ! -e "$dmg" ]] || { echo "Preserve existing release: $dmg already exists" >&2; exit 1; }
staging="$(mktemp -d "$repo/.build/release-stage.XXXXXX")"
echo "Building Lucid $version ($identity)"
xcodebuild -project Lucid.xcodeproj -scheme Lucid -configuration Release \
  -derivedDataPath "$build" build MARKETING_VERSION="$version" \
  CODE_SIGNING_ALLOWED=NO SDK_STAT_CACHE_ENABLE=NO COMPILATION_CACHE_ENABLE_CACHING=NO \
  >"$staging/build.log" 2>&1
[[ -d "$app" ]] || { echo 'Release app missing' >&2; exit 1; }
# Sign nested code from the inside out, then seal the complete application.
while IFS= read -r item; do
  codesign --force --options runtime --timestamp --sign "$identity" "$item"
done < <(python3 - "$app" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])/'Contents'
for p in sorted(root.rglob('*'),key=lambda p:len(p.parts),reverse=True):
 if p.suffix in {'.dylib','.framework','.appex','.xpc','.bundle'}: print(p)
PY
)
codesign --force --options runtime --timestamp --sign "$identity" "$app"
codesign --verify --deep --strict "$app"
if [[ "$mode" != '--local' ]]; then
  ditto -c -k --keepParent "$app" "$staging/notarize.zip"
  xcrun notarytool submit "$staging/notarize.zip" --keychain-profile LUCID_NOTARY --wait
  xcrun stapler staple "$app"
  xcrun stapler validate "$app"
  spctl --assess --type execute --verbose=2 "$app"
fi
payload="$staging/payload"
mkdir -p "$payload"
ditto "$app" "$payload/Lucid.app"
ditto "$repo/BrowserExtension" "$payload/BrowserExtension"
cp "$repo/LICENSE" "$repo/NOTICE" "$payload/"
ln -s /Applications "$payload/Applications"
cat > "$payload/Read me first.txt" <<'NOTE'
Lucid 1.0 — browser video enhancement for Apple silicon, macOS 26 or later.

1. Drag Lucid to Applications and open it.
2. Copy BrowserExtension to a permanent folder on your Mac before ejecting
   this disk image. Open chrome://extensions or edge://extensions, enable
   Developer mode, choose Load unpacked, and select that copied folder.
3. Reload your video tab and turn Lucid on in the menu bar.
4. Click Lucid’s icon in the Mac menu bar and enable Launch at login in
   the dropdown to start it when you sign in. Controls open only when you
   click Lucid’s menu bar icon.

Supports enlarged SDR video from 144p through 720p when the model fits your
Mac's frame budget. HDR, protected video, and 1080p sources are declined.
Decoded browser playback does not need Screen Recording permission.
Hold the comparison control to see the original; release to return to Lucid.
NOTE
if [[ "$mode" == '--local' ]]; then
  echo 'LOCAL BUILD: development signing only; this image is not notarized for public distribution.' > "$payload/LOCAL BUILD.txt"
fi
hdiutil create -volname "Lucid $version" -srcfolder "$payload" -format UDZO "$dmg"
codesign --force --timestamp --sign "$identity" "$dmg"
if [[ "$mode" != '--local' ]]; then
  xcrun notarytool submit "$dmg" --keychain-profile LUCID_NOTARY --wait
  xcrun stapler staple "$dmg"
  xcrun stapler validate "$dmg"
fi
hdiutil verify "$dmg"
(cd "${dmg:h}" && shasum -a 256 "${dmg:t}") > "$dmg.sha256"
python3 - "$dmg" "$app" "$version" "$mode" <<'PY'
import hashlib,json,plistlib,sys
from pathlib import Path
image,app,version,mode=sys.argv[1:]
p=Path(image)
info=plistlib.loads((Path(app)/'Contents/Info.plist').read_bytes())
receipt={'version':version,'build':info['CFBundleVersion'],'artifact':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
         'distribution':'local-development' if mode=='--local' else 'developer-id-notarized',
         'models':json.loads((Path(app)/'Contents/Resources/Models.json').read_text()),
         'compiled_models':json.loads((Path(app)/'Contents/Resources/ModelBuild.json').read_text())}
p.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
echo "Packaged: $dmg"
echo "Build log: $staging/build.log"
