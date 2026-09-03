//
//  BrowserVideoReport.swift
//  Lucid
//
//  Geometry and state of the video a browser companion is tracking. The browser
//  extension (or a page in direct mode) sends this as JSON over the local
//  WebSocket bridge. Everything is in CSS pixels relative to the viewport unless
//  noted; the native side converts to window points using `dpr` and the display's
//  backing scale.
//

import CoreGraphics
import Foundation

struct BrowserVideoReport: Codable, Sendable, Equatable {
    struct Rect: Codable, Sendable, Equatable {
        var x: Double
        var y: Double
        var w: Double
        var h: Double

        var cgRect: CGRect { CGRect(x: x, y: y, width: w, height: h) }
    }

    struct Video: Codable, Sendable, Equatable {
        /// Content box of the rendered video after `object-fit`, viewport CSS px.
        var rect: Rect
        /// Intrinsic decoded size (`videoWidth` × `videoHeight`).
        var iw: Int
        var ih: Int
        var paused: Bool
        var ended: Bool
        var fullscreen: Bool
        var pip: Bool
        /// CSS border radius in CSS px (corner rounding to mirror), 0 when none.
        var radius: Double
    }

    enum Kind: String, Codable, Sendable {
        case hello
        case video
        case gone
    }

    var type: Kind
    var browser: String
    /// Stable id for one tab/document; the same id updates the same session.
    var session: String
    var title: String
    var url: String?
    var visible: Bool
    /// Browser window position/size in screen points (top-left origin).
    var screenX: Double
    var screenY: Double
    var outerWidth: Double
    var outerHeight: Double
    /// Viewport size in CSS px.
    var innerWidth: Double
    var innerHeight: Double
    /// `window.devicePixelRatio` = page zoom × display backing scale.
    var dpr: Double
    var video: Video?
    /// True while the page is scrolling, resizing, or the video box is animating.
    /// True when the page is streaming decoded frames over the bridge.
    var frames: Bool?
    /// True when the page is drawing the enhanced frames itself.
    var draws: Bool?
    var moving: Bool
    /// True while the pointer is over the video and page controls may be visible.
    var hover: Bool
    /// Page elements drawn over the video that must stay live (control bars,
    /// captions). Viewport CSS px.
    var cutouts: [Rect]
    var ts: Double

    /// CSS px → window points. Page zoom makes CSS px differ from points.
    func cssToPoints(backingScale: CGFloat) -> CGFloat {
        guard dpr > 0, backingScale > 0 else { return 1 }
        return CGFloat(dpr) / backingScale
    }

    /// Origin of the viewport inside the browser window (points, top-left origin).
    /// Assumes browser chrome above and symmetric borders, which holds for Chrome,
    /// Edge and Safari without a sidebar.
    func viewportOrigin(backingScale: CGFloat) -> CGPoint {
        let z = cssToPoints(backingScale: backingScale)
        let chromeLeft = max(0, (CGFloat(outerWidth) - CGFloat(innerWidth) * z) / 2)
        let chromeTop = max(0, CGFloat(outerHeight) - CGFloat(innerHeight) * z)
        return CGPoint(x: chromeLeft, y: chromeTop)
    }

    /// Converts a viewport CSS rect to a window-local rect in points.
    func windowLocalRect(_ rect: Rect, backingScale: CGFloat) -> CGRect {
        let z = cssToPoints(backingScale: backingScale)
        let origin = viewportOrigin(backingScale: backingScale)
        return CGRect(
            x: origin.x + CGFloat(rect.x) * z,
            y: origin.y + CGFloat(rect.y) * z,
            width: CGFloat(rect.w) * z,
            height: CGFloat(rect.h) * z
        )
    }
}
