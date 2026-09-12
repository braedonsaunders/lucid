<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/assets/lucid-logo-dark.svg">
    <img src=".github/assets/lucid-logo.svg" alt="Lucid — sharper browser video on Apple silicon" width="320">
  </picture>
</p>

<h2 align="center">Your Mac deserves better than blurry video.</h2>

<p align="center">
  Real-time AI video enhancement. Right in your browser.<br>
  Free. Open source. Powered by your Mac.
</p>

<p align="center">
  <a href="https://github.com/braedonsaunders/lucid/releases/download/v1.0.0/Lucid-1.0.0-Apple-Silicon.dmg"><img src="https://img.shields.io/badge/Download_for_Mac-v1.0.0-f5a623?style=for-the-badge&logo=apple&logoColor=white" alt="Download Lucid 1.0.0 for Mac"></a>
</p>

<p align="center">
  Apple silicon · macOS 26+ · Chrome &amp; Edge<br>
  <a href="https://github.com/braedonsaunders/lucid/releases/tag/v1.0.0">Release notes</a> ·
  <a href="#get-started">Install guide</a> ·
  <a href="LICENSE">MIT licensed</a>
</p>

---

A beautiful Retina display. A stream that looks like it’s made of blocks.

**Lucid gives low-quality browser video a second chance.** It uses AI to reconstruct detail at 2× resolution, soften compression artifacts, and make supported streams look clearer as you watch. Everything runs locally on Apple silicon.

Open a video. Let Lucid handle the picture.

### Small app. Better picture.

- **Enhances as you watch.** Works inside the video’s existing place on the page, including when you resize it or go fullscreen.
- **Uses the Mac you already have.** Automatically chooses a model size your Mac can keep up with.
- **Your video stays on your Mac.** Enhancement runs locally, with no cloud processing or account required.
- **Lives in your menu bar.** Click the aperture icon for picture presets, adjustments, and **Launch at login**.
- **See the difference yourself.** Hold the comparison control to see the original; release it to return to Lucid.
- **Free to use. Free to change.** MIT licensed, with the source and model packages included.

### Get started

**[Download the Mac installer →](https://github.com/braedonsaunders/lucid/releases/download/v1.0.0/Lucid-1.0.0-Apple-Silicon.dmg)**

The DMG includes **Lucid.app** and the **Chrome/Edge browser companion**. You don’t need Xcode or a source checkout.

1. **Install Lucid.** Open the DMG, drag Lucid to Applications, and launch it. Look for the aperture icon in your Mac’s top menu bar.
2. **Add the browser companion.** Copy the included `BrowserExtension` folder somewhere permanent. Open `chrome://extensions` or `edge://extensions`, enable **Developer mode**, select **Load unpacked**, and choose that folder.
3. **Play a video.** Reload your video tab. Lucid starts enhancing supported video automatically. Use the menu bar dropdown to adjust the picture or turn on **Launch at login**.

Keep the companion folder after installation; your browser loads it from there. The companion isn’t in the browser extension stores yet.

**Signed for your Mac.** Lucid is Developer ID signed and notarized by Apple. Its notarization ticket is included in the app.

### What works today?

| | Lucid 1.0 |
|---|---|
| Mac | Apple silicon, macOS 26 or later |
| Browsers in the installer | Chrome and Edge |
| Video | Enlarged, unprotected SDR streams from 144p to 720p, within your Mac’s performance budget |
| Controls | Mac menu bar dropdown, picture presets, live adjustments, original comparison, launch at login |
| Processing | Local on your Mac |

Results depend on the source. HDR, DRM-protected video, and 1080p-or-higher sources aren’t enhanced. Website compatibility varies. Safari integration is available to [build from source](Docs/DEVELOPMENT.md#safari).

### Built in the open

Curious about the model, measurements, or implementation? Read the [technical notes](Docs/TECHNICAL.md). Want to build or contribute? Start with the [developer guide](Docs/DEVELOPMENT.md).

Found a video that doesn’t work? [Open an issue](https://github.com/braedonsaunders/lucid/issues) with your Mac model, browser, and steps to reproduce it.

**Like what you see? Star Lucid and send it to someone with a Mac.**
