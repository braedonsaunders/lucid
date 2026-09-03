#!/bin/zsh
#
# Builds a release of Lucid: a signed .app inside a .dmg, notarised and stapled
# when the credentials for it exist.
#
#   Tools/release.sh 0.3.0
#
# Signing is not decoration here. macOS ties the Screen Recording grant to the
# code signature, so an ad-hoc signature - which changes on every build - makes
# the system ask for permission again each time. A stable identity fixes that.
# Gatekeeper is a separate problem: an app downloaded from the internet also has
# to be signed with a Developer ID certificate and notarised by Apple, or people
# will be told it is damaged.
#
set -euo pipefail

version="${1:-}"
if [[ -z "$version" ]]; then
  echo "usage: Tools/release.sh <version>   e.g. Tools/release.sh 0.3.0" >&2
  exit 2
fi

repo="${0:A:h:h}"
cd "$repo"
build="$repo/.build/release"
app="$build/Build/Products/Release/Lucid.app"
staging="$repo/.build/dmg"
dmg="$repo/.build/Lucid-$version.dmg"

# ---- pick the strongest identity available --------------------------------
developer_id="$(security find-identity -v -p codesigning 2>/dev/null \
  | grep -o '"Developer ID Application: [^"]*"' | head -1 | tr -d '"' || true)"
development="$(security find-identity -v -p codesigning 2>/dev/null \
  | grep -o '"Apple Development: [^"]*"' | head -1 | tr -d '"' || true)"

if [[ -n "$developer_id" ]]; then
  identity="$developer_id"
  distributable=1
elif [[ -n "$development" ]]; then
  identity="$development"
  distributable=0
  echo "⚠️  No Developer ID certificate found; signing with a development identity."
  echo "    The permission grant will stick on this Mac, but anyone who downloads"
  echo "    the result will be told the app is damaged. See the notes at the end."
else
  echo "✗ No code signing identity at all. Add one in Xcode → Settings → Accounts." >&2
  exit 1
fi
echo "▸ signing as: $identity"

# ---- build ----------------------------------------------------------------
echo "▸ building Release"
xcodebuild -project Lucid.xcodeproj -scheme Lucid -configuration Release \
  -derivedDataPath "$build" build \
  CODE_SIGNING_ALLOWED=NO SDK_STAT_CACHE_ENABLE=NO COMPILATION_CACHE_ENABLE_CACHING=NO \
  >/dev/null
[[ -d "$app" ]] || { echo "✗ no app at $app" >&2; exit 1; }

# Record the version people will see.
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $version" "$app/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $version" "$app/Contents/Info.plist"

# The learned upscaler's models. Compiled here so the app does not pay for
# compilation on its first frame, and only the sizes it can actually use.
# The sizes are listed rather than globbed: Model/ also holds the ones that
# were converted to measure against, and shipping those would be dead weight.
# This list is LearnedUpscaler.variants - keep the two in step.
resources="$app/Contents/Resources"
mkdir -p "$resources"
for size in 256x144 320x180 432x240 480x270 640x360 864x480; do
  model="$repo/Model/SPAN_x4_ch32u_$size.mlpackage"
  [[ -d "$model" ]] || { echo "  missing $(basename "$model")"; continue; }
  name="$(basename "${model%.mlpackage}")"
  if [[ ! -d "$resources/$name.mlmodelc" ]]; then
    xcrun coremlcompiler compile "$model" "$resources" >/dev/null 2>&1 || true
  fi
done

# ---- sign -----------------------------------------------------------------
# A secure timestamp is required for notarisation and harmless without it.
echo "▸ signing"
codesign --force --deep --options runtime --timestamp \
  --sign "$identity" "$app"
codesign --verify --strict --verbose=2 "$app" 2>&1 | tail -2

# ---- package --------------------------------------------------------------
echo "▸ building disk image"
rm -rf "$staging" "$dmg"
mkdir -p "$staging"
cp -R "$app" "$staging/"
ln -s /Applications "$staging/Applications"
cat > "$staging/Read me first.txt" <<'NOTE'
Lucid enhances low-bitrate video in your browser.

1. Drag Lucid to Applications and open it.
2. Grant Screen Recording when asked. Lucid needs it to place the enhanced
   picture over the video; it is not used to record anything.
3. Load the browser companion: in Chrome open chrome://extensions, turn on
   Developer mode, choose "Load unpacked", and pick the BrowserExtension
   folder from the Lucid source.

Lucid lives in the menu bar and the Dock menu. There is an on/off switch and
four quality settings, and that is the whole interface.
NOTE

hdiutil create -volname "Lucid $version" -srcfolder "$staging" -ov -format UDZO "$dmg" >/dev/null
echo "▸ disk image: $dmg ($(du -h "$dmg" | cut -f1))"

# ---- notarise -------------------------------------------------------------
if [[ "$distributable" == "1" ]] && xcrun notarytool history --keychain-profile "LUCID_NOTARY" >/dev/null 2>&1; then
  echo "▸ notarising (this takes a few minutes)"
  xcrun notarytool submit "$dmg" --keychain-profile "LUCID_NOTARY" --wait
  xcrun stapler staple "$dmg"
  echo "▸ stapled; the image will open cleanly on any Mac"
elif [[ "$distributable" == "1" ]]; then
  cat <<'NOTE'

⚠️  Signed with a Developer ID but not notarised: no stored credentials.
    Create them once with:
      xcrun notarytool store-credentials LUCID_NOTARY \
        --apple-id <your Apple ID> --team-id <your team> --password <app-specific password>
    then run this script again.
NOTE
else
  cat <<'NOTE'

⚠️  This disk image is signed for local use only.
    To make one anyone can install you need a Developer ID Application
    certificate, which requires the paid Apple Developer Program:
      1. Enrol at developer.apple.com/programs
      2. Xcode → Settings → Accounts → Manage Certificates → + → Developer ID Application
      3. xcrun notarytool store-credentials LUCID_NOTARY --apple-id ... --team-id ... --password ...
      4. Tools/release.sh <version>
    Until then, anyone who downloads it must strip the quarantine flag by hand:
      xattr -dr com.apple.quarantine /Applications/Lucid.app
NOTE
fi

echo
echo "done: $dmg"
