//
//  ComparisonShooter.swift
//  Lucid
//
//  Captures the same video frame once per engine, from the composited screen,
//  so every comparison is of what the eye actually sees rather than of an
//  offline approximation of the pipeline.
//
//  The page pauses the video first, so all shots are the identical frame; only
//  the engine differs between them.
//

import AppKit
import CoreImage
import Foundation
import ScreenCaptureKit

@MainActor
final class ComparisonShooter {
    struct Shot: Codable, Sendable {
        var engine: String
        var file: String
        var fine: Double
        var coarse: Double
    }

    private let context = CIContext()

    /// Runs every engine over one frozen frame and writes a PNG each.
    /// Returns what it wrote, with a sharpness reading per shot.
    func run(
        box: CGRect,
        browserWindowID: CGWindowID,
        overlayWindowID: @escaping () -> CGWindowID,
        directory: URL,
        engines: [EngineKind],
        setEngine: @escaping (EngineKind) async -> Void,
        setOverlayVisible: @escaping (Bool) -> Void,
        settle: Duration = .milliseconds(1200)
    ) async -> [Shot] {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var shots: [Shot] = []

        // The browser's own rendering first. The session keeps running and the
        // overlay is simply left out of the composite, so nothing restarts.
        setOverlayVisible(false)
        try? await Task.sleep(for: .milliseconds(400))
        if let shot = await capture(box: box, browserWindowID: browserWindowID, overlayWindowID: 0,
                                    name: "0-browser", engine: "browser", directory: directory) {
            shots.append(shot)
        }

        setOverlayVisible(true)
        for (index, engine) in engines.enumerated() {
            await setEngine(engine)
            try? await Task.sleep(for: settle)
            if let shot = await capture(
                box: box, browserWindowID: browserWindowID, overlayWindowID: overlayWindowID(),
                name: "\(index + 1)-\(engine.rawValue)", engine: engine.rawValue, directory: directory
            ) {
                shots.append(shot)
            }
        }

        let summary = shots.map { "\($0.engine) fine \(String(format: "%.3f", $0.fine)) coarse \(String(format: "%.3f", $0.coarse))" }
        print("   📸 \(directory.lastPathComponent): " + summary.joined(separator: " · "))
        if let data = try? JSONEncoder().encode(shots) {
            try? data.write(to: directory.appendingPathComponent("shots.json"))
        }
        return shots
    }

    /// Composites only the browser window and Lucid's overlay, cropped to the
    /// video box. Anything else sitting on top of them is ignored, so the shot
    /// is of the video as the page renders it plus whatever Lucid draws over
    /// it, and nothing else.
    private func capture(
        box: CGRect, browserWindowID: CGWindowID, overlayWindowID: CGWindowID,
        name: String, engine: String, directory: URL
    ) async -> Shot? {
        // Off-screen too: the browser may not be frontmost while shooting.
        guard let content = try? await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false),
              let display = content.displays.first(where: { CGDisplayBounds($0.displayID).intersects(box) })
                ?? content.displays.first
        else { print("   ⚠️ shot \(name): no shareable content"); return nil }
        var windows = content.windows.filter { $0.windowID == browserWindowID }
        if overlayWindowID != 0 {
            if let overlay = content.windows.first(where: { $0.windowID == overlayWindowID }) {
                windows.append(overlay)
            } else {
                print("   ⚠️ shot \(name): overlay window \(overlayWindowID) not on screen")
            }
        }
        guard !windows.isEmpty else { print("   ⚠️ shot \(name): browser window \(browserWindowID) not found"); return nil }

        let scale = NSScreen.screens.first { $0.frame.intersects(WindowTracker.appKitRect(fromCG: box)) }?.backingScaleFactor ?? 2
        let configuration = SCStreamConfiguration()
        let bounds = CGDisplayBounds(display.displayID)
        configuration.sourceRect = CGRect(
            x: box.minX - bounds.minX, y: box.minY - bounds.minY, width: box.width, height: box.height
        )
        configuration.width = Int(box.width * scale)
        configuration.height = Int(box.height * scale)
        configuration.pixelFormat = kCVPixelFormatType_32BGRA
        configuration.showsCursor = false
        configuration.captureResolution = .best

        let image: CGImage
        do {
            image = try await SCScreenshotManager.captureImage(
                contentFilter: SCContentFilter(display: display, including: windows),
                configuration: configuration
            )
        } catch {
            print("   ⚠️ shot \(name): capture failed: \(error.localizedDescription)")
            return nil
        }

        let url = directory.appendingPathComponent("\(name).png")
        try? context.writePNGRepresentation(
            of: CIImage(cgImage: image), to: url,
            format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!
        )
        let (fine, coarse) = Self.bands(image)
        return Shot(engine: engine, file: url.lastPathComponent, fine: fine, coarse: coarse)
    }

    /// Mean absolute difference from a 1-pixel and a 2.5-pixel blur: a rough
    /// split into the band that carries fine detail and the band that carries
    /// coarse structure, which is where blockiness shows up.
    private static func bands(_ image: CGImage) -> (Double, Double) {
        let width = image.width, height = image.height
        var gray = [Float](repeating: 0, count: width * height)
        guard let space = CGColorSpace(name: CGColorSpace.linearGray),
              let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: width, space: space, bitmapInfo: 0)
        else { return (0, 0) }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let data = ctx.data else { return (0, 0) }
        let bytes = data.bindMemory(to: UInt8.self, capacity: width * height)
        for i in 0..<(width * height) { gray[i] = Float(bytes[i]) }

        func blur(_ source: [Float], radius: Int) -> [Float] {
            var out = source
            var temp = source
            let weight = 1.0 / Float(radius * 2 + 1)
            for y in 0..<height {
                for x in 0..<width {
                    var sum: Float = 0
                    for d in -radius...radius { sum += source[y * width + min(max(x + d, 0), width - 1)] }
                    temp[y * width + x] = sum * weight
                }
            }
            for y in 0..<height {
                for x in 0..<width {
                    var sum: Float = 0
                    for d in -radius...radius { sum += temp[min(max(y + d, 0), height - 1) * width + x] }
                    out[y * width + x] = sum * weight
                }
            }
            return out
        }
        let near = blur(gray, radius: 1)
        let far = blur(gray, radius: 3)
        var fine: Double = 0, coarse: Double = 0
        for i in 0..<(width * height) {
            fine += Double(abs(gray[i] - near[i]))
            coarse += Double(abs(near[i] - far[i]))
        }
        let count = Double(width * height)
        return (fine / count, coarse / count)
    }
}
