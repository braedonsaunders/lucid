//
//  LucidTests.swift
//  LucidTests
//
//  Unit tests for the pure session/policy/layout logic. Everything here is
//  synchronous geometry and arithmetic: no capture, no Metal, no VideoToolbox,
//  so these run on any Mac without permissions or hardware.
//

import CoreGraphics
import Testing

@testable import Lucid

private func report(
    session: String = "s1",
    browser: String = "chrome",
    title: String = "Clip",
    screenX: Double = 100, screenY: Double = 100,
    outerWidth: Double = 1400, outerHeight: Double = 900,
    innerWidth: Double = 1400, innerHeight: Double = 800, dpr: Double = 2,
    iw: Int = 640, ih: Int = 360, rectX: Double = 10, rectY: Double = 10,
    rectW: Double = 640, rectH: Double = 360
) -> BrowserVideoReport {
    let rect = BrowserVideoReport.Rect(x: rectX, y: rectY, w: rectW, h: rectH)
    return BrowserVideoReport(
        type: .video, browser: browser, session: session, title: title, url: nil,
        visible: true, screenX: screenX, screenY: screenY,
        outerWidth: outerWidth, outerHeight: outerHeight,
        innerWidth: innerWidth, innerHeight: innerHeight, dpr: dpr,
        video: .init(rect: rect, iw: iw, ih: ih, paused: false, ended: false, fullscreen: false, pip: false, radius: 0),
        frames: true, draws: true, moving: false, hover: false, cutouts: [], ts: 0
    )
}

private func window(
    id: CGWindowID = 7, pid: pid_t = 123, owner: String = "Google Chrome",
    title: String = "Clip", layer: Int = 0, bounds: CGRect = .init(x: 100, y: 100, width: 1400, height: 900)
) -> WindowInfo {
    WindowInfo(id: id, pid: pid, ownerName: owner, title: title, layer: layer, alpha: 1, bounds: bounds)
}

struct SessionPolicyTests {
    @Test @MainActor func enhanceableNeedsUpscaleHeadroom() {
        // 640px-wide source in a 640 CSS px box at dpr 2 = 2x physical: enhance.
        #expect(AppCoordinator.isEnhanceable(report()) == true)
        // Same pixels at dpr 1: physical == decoded, no headroom: skip.
        #expect(AppCoordinator.isEnhanceable(report(dpr: 1)) == false)
        // Just under the 1.15x start threshold: skip; the hysteresis overload keeps it.
        let edge = report(dpr: 1, iw: 640, ih: 360, rectX: 0, rectY: 0, rectW: 700, rectH: 400)
        #expect(AppCoordinator.isEnhanceable(edge) == false)
        #expect(AppCoordinator.isEnhanceable(edge, keeping: true) == true)
    }

    @Test @MainActor func enhanceableRejectsWrongSizesAndStates() {
        #expect(AppCoordinator.isEnhanceable(report(iw: 100, ih: 56)) == false)   // below 128x72
        // The ceiling is the learned upscaler's table, not a round number:
        // 640x360 is the largest size it carries inside the frame budget, so
        // 854x480 and up are left alone rather than handed to a weaker scaler.
        #expect(AppCoordinator.isEnhanceable(report(iw: 3840, ih: 2160)) == false)
        let dvd = report(dpr: 2, iw: 854, ih: 480, rectX: 0, rectY: 0, rectW: 1040, rectH: 585)
        #expect(AppCoordinator.isEnhanceable(dvd) == false)
        // 144p is the case that needs the most help, so it has to be inside the
        // window rather than rejected for being small.
        let tiny = report(dpr: 2, iw: 256, ih: 144, rectX: 0, rectY: 0, rectW: 1040, rectH: 585)
        #expect(AppCoordinator.isEnhanceable(tiny) == true)
        #expect(AppCoordinator.isEnhanceable(report(rectX: 0, rectY: 0, rectW: 100, rectH: 60)) == false) // tiny box
        var pip = report(); pip.video?.pip = true
        #expect(AppCoordinator.isEnhanceable(pip) == false)
        var ended = report(); ended.video?.ended = true
        #expect(AppCoordinator.isEnhanceable(ended) == false)
        var hidden = report(); hidden.visible = false
        #expect(AppCoordinator.isEnhanceable(hidden) == false)
        // The YouTube case from ship-prep: 426x240 at ratio ~4.9 must enhance.
        // 426 is not a multiple of 16, so this is the size that has to reach
        // the 432-wide model - the alignment that costs 23.0ms unaligned and
        // 13.6ms aligned.
        #expect(LearnedUpscaler.variant(width: 426, height: 240)?.width == 432)
        let yt = report(dpr: 2, iw: 426, ih: 240, rectX: 0, rectY: 0, rectW: 1040, rectH: 585)
        #expect(AppCoordinator.isEnhanceable(yt) == true)
    }

    @Test func stageCountPicksNearestPowerOfTwo() {
        // 640-wide source in a 1280-physical box: exactly 2x, one pass.
        #expect(EnhancementSession.stageCount(for: report()) == 1)
        // 426-wide source in a 2080-physical box: stretch ~4.9, nearest is 4x.
        #expect(EnhancementSession.stageCount(for: report(dpr: 2, iw: 426, ih: 240, rectX: 0, rectY: 0, rectW: 1040, rectH: 585)) == 2)
        // Stretch 5.5x renders at 4x, not 8x: overshooting is waste, not quality.
        #expect(EnhancementSession.stageCount(for: report(dpr: 2, iw: 400, ih: 225, rectX: 0, rectY: 0, rectW: 1100, rectH: 619)) == 2)
        // Stretch 7x still rounds up to 8x: the boundary sits at ~5.66x.
        #expect(EnhancementSession.stageCount(for: report(dpr: 2, iw: 400, ih: 225, rectX: 0, rectY: 0, rectW: 1400, rectH: 788)) == 3)
        // Below 1x clamps to one pass; missing video never crashes the factory choice.
        #expect(EnhancementSession.stageCount(for: report(dpr: 1, iw: 640, ih: 360)) == 1)
        var gone = report(); gone.video = nil
        #expect(EnhancementSession.stageCount(for: gone) == 1)
    }

    @Test func deliveryWidthIsClampedToTheBox() {
        #expect(EnhancementSession.deliveryWidth(for: report()) == 1280) // 640*2 physical
        #expect(EnhancementSession.deliveryWidth(for: report(dpr: 1)) == 640) // floor
        var huge = report(dpr: 2, rectX: 0, rectY: 0, rectW: 3000, rectH: 1600)
        #expect(EnhancementSession.deliveryWidth(for: huge) == 3840) // ceiling, not 6000
        var gone = report(); gone.video = nil
        #expect(EnhancementSession.deliveryWidth(for: gone) == 1280)
    }
}

struct GeometryTests {
    @Test func cssToPointsDividesZoomFromBackingScale() {
        let r = report(dpr: 2)
        #expect(r.cssToPoints(backingScale: 2) == 1)
        #expect(r.cssToPoints(backingScale: 1) == 2)
        #expect(r.cssToPoints(backingScale: 0) == 1) // degenerate guard
    }

    @Test func windowLocalRectAddsViewportOrigin() {
        // No browser chrome: outer == inner*dpr/scale, so origin is zero-ish
        // and the rect passes through scaled by dpr/backing.
        let r = report(outerWidth: 1400, outerHeight: 800, innerWidth: 1400, innerHeight: 800, dpr: 2)
        let local = r.windowLocalRect(.init(x: 10, y: 20, w: 100, h: 50), backingScale: 2)
        #expect(abs(local.minX - 10) < 0.001)
        #expect(abs(local.minY - 20) < 0.001)
        #expect(abs(local.width - 100) < 0.001)
    }
}

struct WindowMatchingTests {
    @Test func titleEqualityWinsOverFrame() {
        let snapshot = WindowSnapshot(windows: [
            window(id: 1, title: "Other", bounds: .init(x: 100, y: 100, width: 1400, height: 900)),
            window(id: 2, title: "Clip", bounds: .init(x: 5000, y: 5000, width: 1400, height: 900)),
        ])
        #expect(WindowTracker.matchWindow(for: report(), in: snapshot)?.id == 2)
    }

    @Test func duplicateTitlesBreakTiesByFrame() {
        let snapshot = WindowSnapshot(windows: [
            window(id: 1, title: "Clip", bounds: .init(x: 5000, y: 5000, width: 1400, height: 900)),
            window(id: 2, title: "Clip", bounds: .init(x: 100, y: 100, width: 1400, height: 900)),
        ])
        #expect(WindowTracker.matchWindow(for: report(), in: snapshot)?.id == 2)
    }

    @Test func frameFallbackRejectsFarWindows() {
        let snapshot = WindowSnapshot(windows: [
            window(id: 9, title: "Something else", bounds: .init(x: 5000, y: 5000, width: 1400, height: 900)),
        ])
        #expect(WindowTracker.matchWindow(for: report(), in: snapshot) == nil)
    }

    @Test func nonBrowserOwnersNeverMatch() {
        let snapshot = WindowSnapshot(windows: [
            window(id: 9, owner: "Finder", title: "Clip", bounds: .init(x: 100, y: 100, width: 1400, height: 900)),
        ])
        #expect(WindowTracker.matchWindow(for: report(), in: snapshot) == nil)
    }
}

struct TileLayoutTests {
    @Test func singleTileWhenItFits() {
        let layout = TiledVideoToolboxUpscaler.supportedLayout(width: 640, height: 360)
        #expect(layout?.tileCount == 1)
        #expect(layout?.overlap == 0)
    }

    @Test func oddDimensionsHaveNoLayout() {
        #expect(TiledVideoToolboxUpscaler.supportedLayout(width: 641, height: 360) == nil)
        #expect(TiledVideoToolboxUpscaler.supportedLayout(width: 0, height: 0) == nil)
    }

    @Test func largeFramesTileWithEvenCores() {
        // 1920x1080 exceeds the 960px tile edge, so it must split; every core
        // stays even for 420v chroma.
        let layout = TiledVideoToolboxUpscaler.supportedLayout(width: 1920, height: 1080)
        let count = layout?.tileCount ?? 0
        #expect(count > 1)
        #expect((layout?.tileWidth ?? 1) % 2 == 0)
        #expect((layout?.tileHeight ?? 1) % 2 == 0)
        #expect((layout?.tileWidth ?? 9999) <= TiledVideoToolboxUpscaler.maximumTileEdge)
    }
}

struct CutoutMergeTests {
    /// Port of the content.js merge loop: overlapping rects union, disjoint
    /// rects survive, output capped at 16. Kept in sync with
    /// BrowserExtension/content.js computeCutouts by hand; if that loop
    /// changes, this test names the file to update alongside it.
    func merge(_ rects: [BrowserVideoReport.Rect]) -> [BrowserVideoReport.Rect] {
        var out = rects
        var merged = true
        while merged, out.count > 1 {
            merged = false
            outer: for a in 0..<out.count {
                for b in (a + 1)..<out.count {
                    let A = out[a], B = out[b]
                    if A.x < B.x + B.w, B.x < A.x + A.w, A.y < B.y + B.h, B.y < A.y + A.h {
                        let x = min(A.x, B.x), y = min(A.y, B.y)
                        out[a] = .init(x: x, y: y, w: max(A.x + A.w, B.x + B.w) - x, h: max(A.y + A.h, B.y + B.h) - y)
                        out.remove(at: b); merged = true; break outer
                    }
                }
            }
        }
        return Array(out.prefix(16))
    }

    @Test func overlappingRectsUnion() {
        let merged = merge([.init(x: 0, y: 0, w: 10, h: 10), .init(x: 5, y: 5, w: 10, h: 10)])
        #expect(merged.count == 1)
        #expect(merged[0].w == 15)
        #expect(merged[0].h == 15)
    }

    @Test func disjointRectsSurviveAndCapHolds() {
        let many = (0..<20).map { BrowserVideoReport.Rect(x: Double($0 * 100), y: 0, w: 10, h: 10) }
        #expect(merge(many).count == 16)
    }
}
