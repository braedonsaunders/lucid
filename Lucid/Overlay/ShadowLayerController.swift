//
//  ShadowLayerController.swift
//  Lucid
//
//  Keeps the shadow layer glued to the browser's video box.
//
//  Capture covers the video box plus a margin, so when the page scrolls the
//  enhanced frame still contains the pixels that are now under the box. Each
//  frame remembers the rect it was captured from; every tick the layer is
//  offset by the difference between that rect and where the video is now, so
//  the enhanced image stays locked to the page instead of sliding or being
//  hidden. The capture rect itself is only re-centred when the video drifts
//  past half the margin, which keeps stream reconfiguration rare.
//

import AppKit
import CoreGraphics
import Foundation

@MainActor
final class ShadowLayerController {
    let windowID: CGWindowID
    let pid: pid_t
    private let window: ShadowLayerWindow
    private let presenter: FramePresenter
    private var timer: Timer?
    private var report: BrowserVideoReport
    private(set) var isShowing = false
    private var everPinned = false
    private var forcedHidden = false

    /// Half-width of the captured border around the video box, in points.
    var margin = CGSize(width: 56, height: 56)
    /// Window-local capture rect (points, top-left origin), constant in size.
    private(set) var captureRect: CGRect = .zero
    private var videoRect: CGRect = .zero
    private var browserSize: CGSize = .zero

    var hasContent = false { didSet { if hasContent != oldValue { tick() } } }
    /// Fires when the stream should be pointed at a new rect.
    var onCaptureRectChanged: ((CGRect) -> Void)?
    /// Fires when the video box changed size, which invalidates the pipeline.
    var onResized: (() -> Void)?
    var onWindowLost: (() -> Void)?

    static let tickInterval: TimeInterval = 1.0 / 120.0

    init(window: ShadowLayerWindow, presenter: FramePresenter, windowID: CGWindowID, pid: pid_t, report: BrowserVideoReport) {
        self.window = window
        self.presenter = presenter
        self.windowID = windowID
        self.pid = pid
        self.report = report
    }

    func start() {
        guard timer == nil else { return }
        let timer = Timer(timeInterval: Self.tickInterval, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in self?.tick() }
        }
        timer.tolerance = 0.001
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
        tick()
    }

    func stop() {
        timer?.invalidate()
        timer = nil
        window.setVisible(false)
        window.orderOut(nil)
        isShowing = false
    }

    /// Screen rect of the video box right now, for the comparison harness.
    var videoScreenRect: CGRect {
        guard let browser = WindowSnapshot.capture().window(id: windowID), videoRect != .zero else { return .zero }
        return CGRect(x: browser.bounds.minX + videoRect.minX, y: browser.bounds.minY + videoRect.minY,
                      width: videoRect.width, height: videoRect.height)
    }

    func setCapturable(_ capturable: Bool) { window.setCapturable(capturable) }

    /// Holds the overlay hidden regardless of the usual rules, so the harness
    /// can photograph the page underneath without restarting anything.
    func setForcedHidden(_ hidden: Bool) {
        forcedHidden = hidden
        tick()
    }

    func update(report: BrowserVideoReport) {
        self.report = report
        tick()
    }

    /// Capture rect for a video box: the box plus a margin, kept at a constant
    /// size and slid (never shrunk) to stay inside the browser window.
    func captureRect(for box: CGRect, in windowSize: CGSize) -> CGRect {
        let mx = min(margin.width, max(0, (windowSize.width - box.width) / 2))
        let my = min(margin.height, max(0, (windowSize.height - box.height) / 2))
        let size = CGSize(width: box.width + mx * 2, height: box.height + my * 2)
        // Centre on the visible part of the box, so a video half off the top of
        // the viewport still has its visible half captured.
        let visible = box.intersection(CGRect(origin: .zero, size: windowSize))
        let anchor = visible.isNull ? box : visible
        var origin = CGPoint(x: anchor.midX - size.width / 2, y: anchor.midY - size.height / 2)
        origin.x = min(max(0, origin.x), max(0, windowSize.width - size.width))
        origin.y = min(max(0, origin.y), max(0, windowSize.height - size.height))
        return CGRect(origin: origin, size: size).integral
    }

    private func tick() {
        let snapshot = WindowSnapshot.capture()
        guard let browser = snapshot.window(id: windowID) else {
            hide()
            if WindowSnapshot.capture(onScreenOnly: false).window(id: windowID) == nil { onWindowLost?() }
            return
        }
        guard !forcedHidden else { hide(); return }
        guard let video = report.video, report.visible else { hide("report has no visible video"); return }

        let screen = WindowTracker.screen(for: browser.bounds) ?? NSScreen.main
        let backingScale = screen?.backingScaleFactor ?? 2
        let local = report.windowLocalRect(video.rect, backingScale: backingScale)
        guard local.width >= 2, local.height >= 2 else { hide("degenerate box"); return }

        // A resized box changes the decoded-pixels-per-point scale, so the
        // pipeline has to be rebuilt; the coordinator restarts the session.
        if videoRect != .zero,
           abs(local.width - videoRect.width) > 1 || abs(local.height - videoRect.height) > 1 {
            videoRect = local
            captureRect = captureRect(for: local, in: browser.bounds.size)
            onResized?()
        }
        videoRect = local
        browserSize = browser.bounds.size

        // Re-centre the capture only when the box drifts past half the margin.
        let desired = captureRect(for: local, in: browserSize)
        let drift = max(abs(desired.minX - captureRect.minX), abs(desired.minY - captureRect.minY))
        if captureRect == .zero || drift > max(4, min(margin.width, margin.height) / 2) {
            captureRect = desired
            onCaptureRectChanged?(desired)
        }

        var screenRect = CGRect(
            x: browser.bounds.minX + local.minX, y: browser.bounds.minY + local.minY,
            width: local.width, height: local.height
        )
        screenRect = alignToDevicePixels(screenRect, scale: backingScale)

        guard hasContent else { hide("no frames yet"); return }
        // The display layer keeps its last image forever. If the pipeline has
        // gone quiet - the clip looped, the page stopped sending, the tab
        // changed - showing that image over a playing video is worse than
        // showing nothing, so step aside until frames resume.
        let sinceFrame = ContinuousClock.now - presenter.lastPresented
        if sinceFrame > .milliseconds(400) {
            hide("no frame for \(Int(sinceFrame.milliseconds)) ms")
            return
        }
        if let visible = screen?.frame, !visible.intersects(WindowTracker.appKitRect(fromCG: screenRect)) {
            hide("video box off screen")
            return
        }
        // A menu or dialog the browser opens over the video wins. The browser's
        // own toolbar is a separate window too and overlaps the top of the
        // content area by a hair, so only a window covering a real part of the
        // video counts: otherwise scrolling a video up under the toolbar would
        // switch the enhancement off, which is exactly the kind of seam this is
        // supposed to avoid.
        if let browserIndex = snapshot.index(of: windowID) {
            let ourID = CGWindowID(window.windowNumber)
            let videoArea = max(screenRect.width * screenRect.height, 1)
            for entry in snapshot.windows[..<browserIndex] where entry.id != ourID {
                guard entry.pid == pid, entry.alpha > 0 else { continue }
                let overlap = entry.bounds.intersection(screenRect)
                guard !overlap.isNull else { continue }
                if (overlap.width * overlap.height) / videoArea > 0.25 {
                    hide("browser window [\(entry.id)] covers \(Int(overlap.width))x\(Int(overlap.height)) of the video")
                    return
                }
            }
        }

        // Place the enhanced frame where the pixels it holds are now. The
        // captured region can only cover the part of the video inside the
        // viewport; anywhere it does not reach, the window is transparent and
        // the browser's own video shows through, so a video scrolling off the
        // edge degrades smoothly instead of popping.
        let displayed = presenter.rectOnScreen()
        let imageRect = displayed == .zero ? captureRect : displayed
        let covered = imageRect.intersection(local)
        guard !covered.isNull, covered.width * covered.height > local.width * local.height * 0.05 else {
            hide("frame covers \(Int(covered.width))x\(Int(covered.height)) of \(Int(local.width))x\(Int(local.height))")
            return
        }

        // The overlay only ever covers the part of the video inside the
        // browser's content window: a video scrolled half out of the viewport
        // must not paint over the toolbar above it.
        let visibleScreen = alignToDevicePixels(screenRect.intersection(browser.bounds), scale: backingScale)
        guard !visibleScreen.isNull, visibleScreen.width >= 2, visibleScreen.height >= 2 else {
            hide("visible part \(Int(visibleScreen.width))x\(Int(visibleScreen.height)) in window \(Int(browser.bounds.width))x\(Int(browser.bounds.height))")
            return
        }
        let visibleLocal = CGRect(
            x: visibleScreen.minX - browser.bounds.minX, y: visibleScreen.minY - browser.bounds.minY,
            width: visibleScreen.width, height: visibleScreen.height
        )
        let frame = WindowTracker.appKitRect(fromCG: visibleScreen)
        if window.frame != frame {
            window.setFrame(frame, display: false)
        }
        window.updateBackingScale(backingScale)
        window.setImageFrame(NSRect(
            x: imageRect.minX - visibleLocal.minX,
            y: visibleLocal.height - (imageRect.minY - visibleLocal.minY) - imageRect.height,
            width: imageRect.width, height: imageRect.height
        ))
        applyCutouts(video: video, backingScale: backingScale, localVideoRect: visibleLocal)

        // Keep the layer directly above the browser window.
        var needsPin = !everPinned
        if let browserIndex = snapshot.index(of: windowID) {
            let ourID = CGWindowID(window.windowNumber)
            if let ourIndex = snapshot.index(of: ourID) {
                if ourIndex > browserIndex {
                    needsPin = true
                } else if snapshot.windows[(ourIndex + 1)..<browserIndex].contains(where: { $0.pid != pid }) {
                    needsPin = true
                }
            } else {
                needsPin = true
            }
        }
        if needsPin {
            window.order(.above, relativeTo: Int(windowID))
            everPinned = true
        }
        window.setVisible(true)
        isShowing = true
    }

    private var lastHideReason = ""

    private func hide(_ reason: String = "") {
        if AppCoordinator.debugLogging, !reason.isEmpty, reason != lastHideReason {
            print("   🙈 hidden: \(reason)")
            lastHideReason = reason
        }
        guard isShowing else { return }
        window.setVisible(false)
        isShowing = false
    }

    private func applyCutouts(video: BrowserVideoReport.Video, backingScale: CGFloat, localVideoRect: CGRect) {
        let z = report.cssToPoints(backingScale: backingScale)
        var rects: [NSRect] = []
        for cutout in report.cutouts {
            let local = report.windowLocalRect(cutout, backingScale: backingScale)
            let rect = NSRect(
                x: local.minX - localVideoRect.minX,
                y: localVideoRect.height - (local.minY - localVideoRect.minY) - local.height,
                width: local.width, height: local.height
            )
            rects.append(rect.insetBy(dx: -1, dy: -1))
        }
        window.setCutouts(rects, cornerRadius: CGFloat(video.radius) * z)
    }

    private func alignToDevicePixels(_ rect: CGRect, scale: CGFloat) -> CGRect {
        let minX = (rect.minX * scale).rounded() / scale
        let minY = (rect.minY * scale).rounded() / scale
        let maxX = (rect.maxX * scale).rounded() / scale
        let maxY = (rect.maxY * scale).rounded() / scale
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }
}
