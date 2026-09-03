// Research probe: can a window from this process be z-ordered directly above another
// app's window (cross-process order(.above, relativeTo:)), and does it stay there when
// other apps come to the front? Prints WindowServer front-to-back order each step.
import AppKit
import ScreenCaptureKit

func zorder(mark: [CGWindowID: String]) -> String {
    let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []
    var out: [String] = []
    for w in list {
        let id = w[kCGWindowNumber as String] as? CGWindowID ?? 0
        let owner = w[kCGWindowOwnerName as String] as? String ?? "?"
        let layer = w[kCGWindowLayer as String] as? Int ?? 0
        let name = w[kCGWindowName as String] as? String ?? ""
        guard layer == 0 || mark[id] != nil else { continue }
        out.append("\(mark[id] ?? owner)\(name.isEmpty ? "" : "(\(name.prefix(18)))")[\(id)]L\(layer)")
    }
    return out.prefix(8).joined(separator: "  >  ")
}

@main
struct Main {
    static func main() async {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        guard let content = try? await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true),
              let win = content.windows.first(where: { ($0.title ?? "").hasPrefix("Lucid|") && ($0.owningApplication?.applicationName.lowercased().contains("chrome") ?? false) })
        else { print("no chrome test window"); exit(1) }
        let parts = (win.title ?? "").split(separator: "|")
        let crop = parts[1].split(separator: ",").compactMap { Double($0) }
        let mainH = CGDisplayBounds(CGMainDisplayID()).height
        let videoCG = CGRect(x: win.frame.origin.x + crop[0], y: win.frame.origin.y + crop[1], width: crop[2], height: crop[3])
        let videoAK = NSRect(x: videoCG.origin.x, y: mainH - videoCG.origin.y - videoCG.height, width: videoCG.width, height: videoCG.height)

        let overlay = await MainActor.run { () -> NSWindow in
            let w = NSWindow(contentRect: videoAK, styleMask: .borderless, backing: .buffered, defer: false)
            w.level = .normal
            w.backgroundColor = .systemBlue
            w.ignoresMouseEvents = true
            w.hasShadow = false
            w.alphaValue = 0.99
            w.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient, .ignoresCycle, .stationary]
            w.order(.above, relativeTo: Int(win.windowID))
            return w
        }
        let ovID = CGWindowID(await MainActor.run { overlay.windowNumber })
        let mark: [CGWindowID: String] = [ovID: "OVERLAY", win.windowID: "CHROME"]
        try? await Task.sleep(for: .seconds(1))
        print("1 after order(.above, chrome):  \(zorder(mark: mark))")

        // Bring Finder to front (its window may or may not intersect; we only care about order).
        NSWorkspace.shared.open(URL(fileURLWithPath: NSHomeDirectory() + "/Documents"))
        try? await Task.sleep(for: .seconds(2))
        print("2 after Finder activated:       \(zorder(mark: mark))")

        // Activate Chrome again.
        win.owningApplication.flatMap { NSRunningApplication(processIdentifier: $0.processID) }?.activate()
        try? await Task.sleep(for: .seconds(2))
        print("3 after Chrome re-activated:    \(zorder(mark: mark))")

        // Re-pin.
        await MainActor.run { overlay.order(.above, relativeTo: Int(win.windowID)) }
        try? await Task.sleep(for: .seconds(1))
        print("4 after re-pin:                 \(zorder(mark: mark))")

        // Close the Finder window we opened (best effort) and exit.
        let src = "tell application \"Finder\" to close (every window whose name is \"Documents\")"
        _ = NSAppleScript(source: src)?.executeAndReturnError(nil)
        await MainActor.run { overlay.orderOut(nil) }
        exit(0)
    }
}
