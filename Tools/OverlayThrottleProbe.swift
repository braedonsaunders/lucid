// Research probe: does a click-through overlay window placed over a Chrome/Safari
// <video> rectangle reduce the frame rate that ScreenCaptureKit window capture
// delivers for that browser window?  Not product code.
//
// Build: swiftc -O -parse-as-library Tools/OverlayThrottleProbe.swift -o .build/bin/overlay-probe
// Run:   .build/bin/overlay-probe [phaseSeconds]
import AppKit
import CoreMedia
import ScreenCaptureKit

final class Counter: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var complete = 0
    private var idle = 0
    private var other = 0
    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen else { return }
        let attachments = CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]]
        let statusRaw = attachments?.first?[.status] as? Int ?? -1
        lock.lock(); defer { lock.unlock() }
        switch SCFrameStatus(rawValue: statusRaw) {
        case .complete: complete += 1
        case .idle: idle += 1
        default: other += 1
        }
    }
    func stream(_ stream: SCStream, didStopWithError error: Error) { print("stream stopped: \(error)") }
    func reset() -> (Int, Int, Int) {
        lock.lock(); defer { lock.unlock() }
        let r = (complete, idle, other); complete = 0; idle = 0; other = 0; return r
    }
}

@MainActor
final class Probe {
    var overlay: NSWindow?

    func makeOverlay(frame: NSRect, alpha: CGFloat, color: NSColor, opaque: Bool? = nil, windowAlpha: CGFloat = 1) {
        overlay?.orderOut(nil)
        let w = NSWindow(contentRect: frame, styleMask: .borderless, backing: .buffered, defer: false)
        w.level = .floating
        w.isOpaque = opaque ?? (alpha >= 1)
        w.alphaValue = windowAlpha
        w.backgroundColor = color.withAlphaComponent(alpha)
        w.ignoresMouseEvents = true
        w.hasShadow = false
        w.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        w.orderFrontRegardless()
        overlay = w
    }
    func clearOverlay() { overlay?.orderOut(nil); overlay = nil }
}

@main
struct Main {
    static func main() async {
        let phaseSeconds = Double(CommandLine.arguments.dropFirst().first ?? "6") ?? 6
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)

        guard let content = try? await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true) else {
            print("ERROR: no shareable content (screen recording permission?)"); exit(1)
        }
        let wantApp = CommandLine.arguments.dropFirst(2).first?.lowercased()
        guard let win = content.windows.first(where: { ($0.title ?? "").hasPrefix("Lucid|") && (wantApp == nil || ($0.owningApplication?.applicationName.lowercased().contains(wantApp!) ?? false)) }) else {
            print("ERROR: no window with Lucid| title. Open the test page in Chrome/Safari."); exit(2)
        }
        let parts = (win.title ?? "").split(separator: "|")
        let crop = parts[1].split(separator: ",").compactMap { Double($0) }
        let size = parts[2].split(separator: "x").compactMap { Int($0) }
        let cropRect = CGRect(x: crop[0], y: crop[1], width: crop[2], height: crop[3])
        print("window: \(win.owningApplication?.applicationName ?? "?") [\(win.windowID)] frame=\(win.frame) crop=\(cropRect) intrinsic=\(size[0])x\(size[1])")

        // Screen-space rect of the video (CG top-left coords) -> AppKit bottom-left coords.
        let mainH = CGDisplayBounds(CGMainDisplayID()).height
        let videoCG = CGRect(x: win.frame.origin.x + cropRect.origin.x, y: win.frame.origin.y + cropRect.origin.y,
                             width: cropRect.width, height: cropRect.height)
        func appKit(_ r: CGRect) -> NSRect { NSRect(x: r.origin.x, y: mainH - r.origin.y - r.height, width: r.width, height: r.height) }
        let videoAK = appKit(videoCG)
        let windowAK = appKit(win.frame)

        let filter = SCContentFilter(desktopIndependentWindow: win)
        let cfg = SCStreamConfiguration()
        cfg.sourceRect = cropRect
        cfg.width = size[0]; cfg.height = size[1]
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: 120)
        cfg.showsCursor = false
        cfg.queueDepth = 5
        cfg.pixelFormat = kCVPixelFormatType_32BGRA
        let counter = Counter()
        let stream = SCStream(filter: filter, configuration: cfg, delegate: counter)
        do {
            try stream.addStreamOutput(counter, type: .screen, sampleHandlerQueue: DispatchQueue(label: "probe.frames"))
            try await stream.startCapture()
        } catch { print("ERROR: start capture: \(error)"); exit(3) }

        let probe = await Probe()
        struct Phase { let name: String; let setup: @MainActor (Probe) -> Void }
        let phases: [Phase] = [
            Phase(name: "A  no overlay") { $0.clearOverlay() },
            Phase(name: "B  opaque overlay over VIDEO rect only") { $0.makeOverlay(frame: videoAK, alpha: 1, color: .systemBlue) },
            Phase(name: "C  opaque overlay over WHOLE browser window") { $0.makeOverlay(frame: windowAK, alpha: 1, color: .systemPurple) },
            Phase(name: "D  near-transparent overlay over VIDEO rect") { $0.makeOverlay(frame: videoAK, alpha: 0.02, color: .black) },
            Phase(name: "E  opaque overlay over VIDEO rect, sharingType=none") { p in p.makeOverlay(frame: videoAK, alpha: 1, color: .systemGreen); p.overlay?.sharingType = .none },
            Phase(name: "F  whole window, isOpaque=false, opaque pixels") { $0.makeOverlay(frame: windowAK, alpha: 1, color: .systemOrange, opaque: false) },
            Phase(name: "G  whole window, isOpaque=true, alphaValue 0.99") { $0.makeOverlay(frame: windowAK, alpha: 1, color: .systemRed, opaque: true, windowAlpha: 0.99) },
            Phase(name: "H  whole window minus 1px top strip, opaque") { $0.makeOverlay(frame: NSRect(x: windowAK.origin.x, y: windowAK.origin.y, width: windowAK.width, height: windowAK.height - 1), alpha: 1, color: .systemTeal) },
            Phase(name: "I  no overlay again") { $0.clearOverlay() },
        ]
        print("phase seconds: \(phaseSeconds)")
        for phase in phases {
            await MainActor.run { phase.setup(probe) }
            try? await Task.sleep(for: .seconds(1.0))   // settle
            _ = counter.reset()
            try? await Task.sleep(for: .seconds(phaseSeconds))
            let (c, i, o) = counter.reset()
            print(String(format: "%-52@  complete %5.1f fps   idle %5.1f/s   other %d", phase.name as NSString, Double(c) / phaseSeconds, Double(i) / phaseSeconds, o))
        }
        await MainActor.run { probe.clearOverlay() }
        try? await stream.stopCapture()
        exit(0)
    }
}
