# Developing Lucid

Requires Apple silicon, macOS 26+, and Xcode with the macOS 26 SDK.

## Build

```sh
xcodebuild -project Lucid.xcodeproj -scheme Lucid -configuration Release \
  -derivedDataPath .build/release build CODE_SIGNING_ALLOWED=NO
```

Open `.build/release/Build/Products/Release/Lucid.app`. For Chrome or Edge, open `chrome://extensions` or `edge://extensions`, enable Developer mode, select **Load unpacked**, and choose the repository’s `BrowserExtension` folder. Reload your video tab.

Every Xcode build checks source model hashes against `Lucid/Resources/Models.json`, compiles the seven models into the app, and removes other model packages. Changed models invalidate their compilation receipt. Missing or corrupt source models fail the build.

## Safari

Build the project under `SafariCompanion/Lucid Companion`, open Lucid Companion, and enable its extension in Safari Settings. Its resources are generated from the canonical browser implementation:

```sh
python3 Tools/sync_safari.py
python3 Tools/sync_safari.py --check
```

## Package a Mac release

```sh
Tools/release.sh 1.0.0
```

The normal release command requires a Developer ID Application certificate and the `LUCID_NOTARY` keychain profile. It signs and notarizes the app and DMG, then verifies signatures, notarization tickets, model packages, and the disk image. The image includes Lucid and the Chrome/Edge companion.

```sh
Tools/release.sh 1.0.0 --local
```

The `--local` option explicitly permits development signing and creates a `-local.dmg` plus checksum and JSON receipt. This build is not notarized. When sharing it, the download page must state that clearly; it must not be described as an Apple-notarized release. The script preserves existing release files rather than overwriting them.

## Verification

```sh
node --test Tools/stream-policy.test.js
python3 -m unittest discover -s Tools -p 'test_*.py'
zsh Tools/check-ladder.sh
xcodebuild -project Lucid.xcodeproj -scheme Lucid -configuration Debug \
  -derivedDataPath .build/tests test -only-testing:LucidTests CODE_SIGNING_ALLOWED=NO
```

The native suite exercises actual Metal kernels, moving grain, motion and scene
cuts, color conversion, malformed frame layouts, model predictions, settings
parity, and presentation metrics. GPU tests require a physical Apple silicon
Mac; hosted CI excludes that suite explicitly.

`Tools/browser_fixture.py` creates the local worker/iframe fixture. Launch a
separate test app with `LUCID_BRIDGE_PORT=47911 LUCID_TOKEN_PORT=47912
LUCID_EPHEMERAL=1` and serve the fixture over loopback HTTP. Ephemeral mode keeps
the test token and enable switch out of the normal app's persisted state.

## Model development

See [technical notes](TECHNICAL.md) for the production architecture, evaluation results, and export path. Older architecture experiments remain under `Tools/experiments` and `Tools/architectures`.
