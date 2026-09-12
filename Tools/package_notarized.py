#!/usr/bin/env python3
"""Package an Xcode-exported, notarized app without needing its cloud signing key."""
import argparse
import hashlib
import json
import plistlib
import subprocess
import tempfile
from pathlib import Path


def run(*args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path, help="Lucid.app exported by xcodebuild -exportNotarizedApp")
    args = parser.parse_args()
    app = args.app.resolve()
    repo = Path(__file__).resolve().parents[1]
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    if info["CFBundleIdentifier"] != "com.braedonsaunders.lucid":
        parser.error("Expected the Lucid app")
    version = info["CFBundleShortVersionString"]
    models = json.loads((app / "Contents/Resources/Models.json").read_text())
    companion = json.loads((repo / "BrowserExtension/manifest.json").read_text())
    if models["release"] != version or companion["version"] != version:
        parser.error("App, models, and browser companion versions must match")
    image = repo / ".build" / f"Lucid-{version}-Apple-Silicon.dmg"
    if image.exists():
        parser.error(f"Preserve existing release: {image}")

    run("codesign", "--verify", "--deep", "--strict", app)
    signature = run("codesign", "-dv", "--verbose=4", app, capture_output=True, text=True).stderr
    if "Authority=Developer ID Application:" not in signature:
        parser.error("The app must have a Developer ID Application signature")
    run("xcrun", "stapler", "validate", app)
    run("spctl", "--assess", "--type", "execute", "--verbose=2", app)
    run("syspolicy_check", "distribution", app)

    stage = Path(tempfile.mkdtemp(prefix="notarized-dmg-", dir=repo / ".build"))
    payload = stage / "payload"
    payload.mkdir()
    run("ditto", app, payload / "Lucid.app")
    run("ditto", repo / "BrowserExtension", payload / "BrowserExtension")
    for name in ["LICENSE", "NOTICE"]:
        run("ditto", repo / name, payload / name)
    (payload / "Applications").symlink_to("/Applications")
    (payload / "Read me first.txt").write_text(
        f"Lucid {version} — clearer browser video on your Mac.\n\n"
        "Requires Apple silicon and macOS 26 or later.\n\n"
        "1. Drag Lucid to Applications and open it.\n"
        "2. Copy BrowserExtension to a permanent folder before ejecting this image.\n"
        "   Open chrome://extensions or edge://extensions, enable Developer mode,\n"
        "   choose Load unpacked, and select that folder.\n"
        "3. Reload your video tab. Click Lucid’s aperture icon in the Mac menu bar\n"
        "   for picture controls and Launch at login.\n\n"
        "For Incognito, enable Allow in Incognito in the companion’s Details.\n"
        "After updating companion files, click Reload on the extension and refresh\n"
        "your video tabs.\n\n"
        "Supports enlarged, unprotected SDR streams from 144p through 720p within\n"
        "your Mac’s performance budget. HDR and protected video are not enhanced.\n"
        "The browser companion is not yet listed in extension stores.\n\n"
        "Lucid.app is Developer ID signed and notarized by Apple. Its notarization\n"
        "ticket is attached to the app. This DMG is a drag-to-Applications container.\n"
    )
    run("hdiutil", "create", "-volname", f"Lucid {version}", "-srcfolder", payload,
        "-format", "UDZO", image)
    run("hdiutil", "verify", image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    Path(str(image) + ".sha256").write_text(f"{digest}  {image.name}\n")
    receipt = {
        "version": version,
        "build": info["CFBundleVersion"],
        "artifact": image.name,
        "sha256": digest,
        "distribution": "developer-id-notarized-app",
        "app_notarized": True,
        "app_ticket_stapled": True,
        "container_separately_notarized": False,
        "signing_authorities": [line.removeprefix("Authority=") for line in signature.splitlines()
                                if line.startswith("Authority=")],
        "models": models,
    }
    image.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Packaged notarized Lucid app: {image}")


if __name__ == "__main__":
    main()
