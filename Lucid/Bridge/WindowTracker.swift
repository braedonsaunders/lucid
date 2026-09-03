//
//  WindowTracker.swift
//  Lucid
//
//  Cheap snapshots of the on-screen window list (front to back) used to find
//  the browser window a report belongs to, to position the shadow layer, and
//  to keep it correctly stacked.
//

import AppKit
import CoreGraphics
import Foundation

struct WindowInfo: Sendable, Equatable {
    let id: CGWindowID
    let pid: pid_t
    let ownerName: String
    let title: String
    let layer: Int
    let alpha: CGFloat
    /// Screen bounds, CG coordinates (top-left origin of the main display).
    let bounds: CGRect
}

struct WindowSnapshot: Sendable {
    /// Front to back.
    let windows: [WindowInfo]

    func window(id: CGWindowID) -> WindowInfo? {
        windows.first { $0.id == id }
    }

    func index(of id: CGWindowID) -> Int? {
        windows.firstIndex { $0.id == id }
    }

    static func capture(onScreenOnly: Bool = true) -> WindowSnapshot {
        let options: CGWindowListOption = onScreenOnly ? [.optionOnScreenOnly] : [.optionAll]
        let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] ?? []
        var result: [WindowInfo] = []
        result.reserveCapacity(list.count)
        for entry in list {
            guard let id = entry[kCGWindowNumber as String] as? CGWindowID,
                  let pid = entry[kCGWindowOwnerPID as String] as? pid_t,
                  let boundsDict = entry[kCGWindowBounds as String] as? [String: CGFloat]
            else { continue }
            let bounds = CGRect(
                x: boundsDict["X"] ?? 0,
                y: boundsDict["Y"] ?? 0,
                width: boundsDict["Width"] ?? 0,
                height: boundsDict["Height"] ?? 0
            )
            result.append(WindowInfo(
                id: id,
                pid: pid,
                ownerName: entry[kCGWindowOwnerName as String] as? String ?? "",
                title: entry[kCGWindowName as String] as? String ?? "",
                layer: entry[kCGWindowLayer as String] as? Int ?? 0,
                alpha: entry[kCGWindowAlpha as String] as? CGFloat ?? 1,
                bounds: bounds
            ))
        }
        return WindowSnapshot(windows: result)
    }
}

enum WindowTracker {
    /// How close a window frame must be (in points of summed edge error) to
    /// count as the reporting browser window when titles cannot decide.
    static let frameMatchTolerance: CGFloat = 48
    /// Minimum on-screen window size considered a browser window.
    static let minimumBrowserSize = CGSize(width: 200, height: 150)

    /// Finds the browser window a report describes.
    ///
    /// Title equality wins; the window frame reported by the page breaks ties.
    /// Frame fallback uses a *relative* tolerance (a fraction of the reported
    /// window size) instead of an absolute point count, so small and
    /// large windows match with the same strictness.
    static func matchWindow(for report: BrowserVideoReport, in snapshot: WindowSnapshot) -> WindowInfo? {
        let candidates = candidateWindows(for: report, in: snapshot)
        guard !candidates.isEmpty else { return nil }

        let reported = CGRect(x: report.screenX, y: report.screenY, width: report.outerWidth, height: report.outerHeight)

        let titled = candidates.filter { !report.title.isEmpty && $0.title == report.title }
        if titled.count == 1 { return titled[0] }
        if titled.count > 1 { return titled.min { frameDistance($0, reported) < frameDistance($1, reported) } }

        // Chrome and Safari suffix the tab title in the window title; accept prefix matches.
        let prefixed = candidates.filter { !report.title.isEmpty && $0.title.hasPrefix(report.title) }
        if prefixed.count == 1 { return prefixed[0] }

        if let closest = candidates.min(by: { frameDistance($0, reported) < frameDistance($1, reported) }),
           isFrameMatch(closest, reported: reported) {
            return closest
        }
        return nil
    }

    /// Browser-owned, normal-layer, plausibly sized windows, front to back.
    static func candidateWindows(for report: BrowserVideoReport, in snapshot: WindowSnapshot) -> [WindowInfo] {
        let browser = report.browser.lowercased()
        return snapshot.windows.filter { window in
            guard window.layer == 0,
                  window.bounds.width > minimumBrowserSize.width,
                  window.bounds.height > minimumBrowserSize.height else { return false }
            let owner = window.ownerName.lowercased()
            switch browser {
            case "chrome": return owner.contains("chrome") || owner.contains("chromium")
            case "edge": return owner.contains("edge")
            case "safari": return owner.contains("safari")
            case "brave": return owner.contains("brave")
            case "arc": return owner.contains("arc")
            default: return owner.contains("chrome") || owner.contains("safari") || owner.contains("edge")
            }
        }
    }

    /// Summed absolute edge error between a window and the reported frame.
    static func frameDistance(_ window: WindowInfo, _ reported: CGRect) -> CGFloat {
        abs(window.bounds.minX - reported.minX) + abs(window.bounds.minY - reported.minY)
            + abs(window.bounds.width - reported.width) + abs(window.bounds.height - reported.height)
    }

    /// Relative frame gate: the summed edge error must be small compared to
    /// the reported window's own size, with the legacy 48pt cap kept as an
    /// absolute ceiling so huge windows do not match loosely.
    static func isFrameMatch(_ window: WindowInfo, reported: CGRect) -> Bool {
        let distance = frameDistance(window, reported)
        let sizeScale = max(1, (reported.width + reported.height) / 2)
        return distance < min(frameMatchTolerance, sizeScale * 0.05)
    }

    /// Screen the window mostly lives on, for its backing scale.
    static func screen(for bounds: CGRect) -> NSScreen? {
        let appKit = appKitRect(fromCG: bounds)
        return NSScreen.screens.max { a, b in
            a.frame.intersection(appKit).area < b.frame.intersection(appKit).area
        } ?? NSScreen.main
    }

    /// CG (top-left origin, main display) → AppKit (bottom-left origin).
    static func appKitRect(fromCG rect: CGRect) -> NSRect {
        let mainHeight = NSScreen.screens.first?.frame.height ?? NSScreen.main?.frame.height ?? 0
        return NSRect(x: rect.minX, y: mainHeight - rect.maxY, width: rect.width, height: rect.height)
    }
}

private extension CGRect {
    var area: CGFloat { isNull ? 0 : width * height }
}
