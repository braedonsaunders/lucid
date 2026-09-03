//
//  TiledVideoToolboxUpscaler.swift
//  Lucid
//
//  Runs Apple's 2× low-latency scaler on a full-resolution frame without ever
//  shrinking the source. Frames larger than the per-session input limit are
//  split into the smallest grid whose tiles (with context overlap) the
//  hardware accepts. Tiles are cropped and composited with Metal blits in the
//  session's native 420v format: no conversions, no Core Image, no clears.
//
//  Not thread-safe by itself; owned and driven by `EnhancementPipeline`.
//

import CoreMedia
import CoreVideo
import Foundation
@preconcurrency import VideoToolbox

@available(macOS 26.0, *)
final class TiledVideoToolboxUpscaler {
    struct Layout: Sendable, Equatable {
        let columns: Int
        let rows: Int
        let tileWidth: Int
        let tileHeight: Int
        let overlap: Int
        var tileCount: Int { columns * rows }
    }

    private struct Tile {
        let index: Int
        let coreRect: CGRect
        let inputRect: CGRect
        let session: SuperResolutionSession
    }

    /// CVPixelBuffer is not Sendable; tiles are handed to exactly one session each.
    private struct BufferBox: @unchecked Sendable {
        let buffer: CVPixelBuffer
    }

    enum Failure: Error {
        case noLayout
        case outputPool
        case sourceFormat
    }

    static let maximumTileEdge = 960
    static let contextOverlap = 32

    let width: Int
    let height: Int
    let layout: Layout
    private let compositor: MetalTileCompositor
    private var tiles: [Tile] = []
    private let outputPool: CVPixelBufferPool
    /// Optional second 2× stage fed with this stage's output (4× total), used
    /// when the video is displayed at 3× or more of its decoded size.
    var nextStage: TiledVideoToolboxUpscaler?

    var outputWidth: Int { nextStage?.outputWidth ?? width * scale }
    var outputHeight: Int { nextStage?.outputHeight ?? height * scale }
    var totalTileCount: Int { layout.tileCount + (nextStage?.totalTileCount ?? 0) }
    var stageCount: Int { 1 + (nextStage?.stageCount ?? 0) }

    /// The factor each tile session runs at. Two chained 2x passes and one 4x
    /// pass reach the same size by different routes and do not look the same.
    let scale: Int

    init(width: Int, height: Int, compositor: MetalTileCompositor, preferredScale: Int = 2) throws {
        guard let layout = Self.supportedLayout(width: width, height: height) else { throw Failure.noLayout }
        self.width = width
        self.height = height
        self.layout = layout
        self.compositor = compositor
        self.scale = preferredScale

        var tiles: [Tile] = []
        for (index, rects) in Self.makeTileRects(layout: layout, width: width, height: height).enumerated() {
            let session = try SuperResolutionSession(
                inputWidth: Int(rects.input.width), inputHeight: Int(rects.input.height),
                preferredScale: Float(preferredScale)
            )
            tiles.append(Tile(index: index, coreRect: rects.core, inputRect: rects.input, session: session))
        }
        self.tiles = tiles

        let format = tiles.first?.session.inputPixelFormat ?? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        var pool: CVPixelBufferPool?
        let attributes: [String: Any] = [
            kCVPixelBufferWidthKey as String: width * preferredScale,
            kCVPixelBufferHeightKey as String: height * preferredScale,
            kCVPixelBufferPixelFormatTypeKey as String: format,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        let poolAttributes: [String: Any] = [kCVPixelBufferPoolMinimumBufferCountKey as String: 4]
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttributes as CFDictionary, attributes as CFDictionary, &pool) == kCVReturnSuccess,
              let pool
        else { throw Failure.outputPool }
        outputPool = pool
        print("   🧩 Native tiled SR: \(width)x\(height) as \(layout.columns)x\(layout.rows) cores with \(layout.overlap)px context overlap")
    }

    /// Pixel format the source frames must arrive in (what the sessions accept).
    var inputPixelFormat: OSType {
        tiles.first?.session.inputPixelFormat ?? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
    }

    static func supportedLayout(width: Int, height: Int) -> Layout? {
        guard width > 0, height > 0, width % 2 == 0, height % 2 == 0 else { return nil }
        guard VTLowLatencySuperResolutionScalerConfiguration.isSupported else { return nil }

        for tileCount in 1...64 {
            for rows in 1...tileCount where tileCount % rows == 0 {
                let columns = tileCount / rows
                guard width % columns == 0, height % rows == 0 else { continue }
                let tileWidth = width / columns
                let tileHeight = height / rows
                // 420v chroma needs even tile edges.
                guard tileWidth % 2 == 0, tileHeight % 2 == 0 else { continue }
                let candidate = Layout(
                    columns: columns, rows: rows, tileWidth: tileWidth, tileHeight: tileHeight,
                    overlap: tileCount == 1 ? 0 : contextOverlap
                )
                let inputs = makeTileRects(layout: candidate, width: width, height: height).map(\.input)
                let accepted = inputs.allSatisfy { rect in
                    let w = Int(rect.width), h = Int(rect.height)
                    guard w <= maximumTileEdge, h <= maximumTileEdge else { return false }
                    return VTLowLatencySuperResolutionScalerConfiguration
                        .supportedScaleFactors(frameWidth: w, frameHeight: h)
                        .contains { abs($0 - 2) < 0.01 }
                }
                if accepted { return candidate }
            }
        }
        return nil
    }

    func upscale(_ source: CVPixelBuffer, pts: CMTime) async throws -> CVPixelBuffer {
        guard CVPixelBufferGetPixelFormatType(source) == inputPixelFormat,
              CVPixelBufferGetWidth(source) == width, CVPixelBufferGetHeight(source) == height
        else { throw Failure.sourceFormat }

        // 1. Crop every tile's context rect into a session-owned input buffer.
        var inputs: [CVPixelBuffer] = []
        inputs.reserveCapacity(tiles.count)
        for tile in tiles {
            inputs.append(try await tile.session.makeInputBuffer())
        }
        try compositor.perform(tiles.map { tile in
            MetalTileCompositor.Copy(source: source, sourceRect: tile.inputRect, destination: inputs[tile.index], destinationOrigin: .zero)
        })
        if let attachments = CVBufferCopyAttachments(source, .shouldPropagate) {
            for input in inputs { CVBufferSetAttachments(input, attachments, .shouldPropagate) }
        }

        // 2. Run all sessions concurrently.
        var results = [CVPixelBuffer?](repeating: nil, count: tiles.count)
        try await withThrowingTaskGroup(of: (Int, BufferBox).self) { group in
            for tile in tiles {
                let input = BufferBox(buffer: inputs[tile.index])
                let session = tile.session
                let index = tile.index
                group.addTask {
                    (index, BufferBox(buffer: try await session.process(input.buffer, pts: pts)))
                }
            }
            for try await (index, output) in group {
                results[index] = output.buffer
            }
        }

        // 3. Composite only each tile's core (drops model edge artifacts) into the output.
        var outputBuffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, outputPool, &outputBuffer) == kCVReturnSuccess,
              let output = outputBuffer
        else { throw Failure.outputPool }

        var copies: [MetalTileCompositor.Copy] = []
        for tile in tiles {
            guard let result = results[tile.index] else { continue }
            let factor = CGFloat(scale)
            let cropInTile = CGRect(
                x: (tile.coreRect.minX - tile.inputRect.minX) * factor,
                y: (tile.coreRect.minY - tile.inputRect.minY) * factor,
                width: tile.coreRect.width * factor,
                height: tile.coreRect.height * factor
            )
            copies.append(MetalTileCompositor.Copy(
                source: result,
                sourceRect: cropInTile,
                destination: output,
                destinationOrigin: CGPoint(x: tile.coreRect.minX * factor, y: tile.coreRect.minY * factor)
            ))
        }
        try compositor.perform(copies)

        // Carry colour metadata so the display layer converts YUV correctly.
        // VideoToolbox hands back pool buffers with no colour description, and
        // an undescribed 420v buffer gets guessed at, which desaturates it.
        if let attachments = CVBufferCopyAttachments(source, .shouldPropagate) {
            CVBufferSetAttachments(output, attachments, .shouldPropagate)
        }
        Self.ensureColorDescription(output)
        if let nextStage {
            return try await nextStage.upscale(output, pts: pts)
        }
        return output
    }

    func end() async {
        for tile in tiles { await tile.session.end() }
        await nextStage?.end()
    }

    /// Fills in any missing colour tags with the Rec. 709 video-range set that
    /// ScreenCaptureKit is asked to produce.
    /// Whether 4:2:0 chroma is treated as left-sited. Off falls back to what an
    /// untagged buffer gets, which is centre siting.
    nonisolated(unsafe) static var chromaSitingLeft = true

    static func ensureColorDescription(_ buffer: CVPixelBuffer) {
        let defaults: [(CFString, CFString)] = [
            (kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2),
            (kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2),
            (kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2),
            // H.264 and VP9 4:2:0 site chroma on the left luma column, not in the
            // centre of the quad. Untagged buffers are read as centre-sited, which
            // shifts colour half a luma pixel - two whole pixels once 4x scaled.
            (kCVImageBufferChromaLocationTopFieldKey,
             chromaSitingLeft ? kCVImageBufferChromaLocation_Left : kCVImageBufferChromaLocation_Center),
            (kCVImageBufferChromaLocationBottomFieldKey,
             chromaSitingLeft ? kCVImageBufferChromaLocation_Left : kCVImageBufferChromaLocation_Center),
        ]
        for (key, value) in defaults where CVBufferCopyAttachment(buffer, key, nil) == nil {
            CVBufferSetAttachment(buffer, key, value, .shouldPropagate)
        }
    }

    private static func makeTileRects(layout: Layout, width: Int, height: Int) -> [(core: CGRect, input: CGRect)] {
        var result: [(CGRect, CGRect)] = []
        let frame = CGRect(x: 0, y: 0, width: width, height: height)
        for row in 0..<layout.rows {
            for column in 0..<layout.columns {
                let core = CGRect(
                    x: column * layout.tileWidth, y: row * layout.tileHeight,
                    width: layout.tileWidth, height: layout.tileHeight
                )
                let input = core.insetBy(dx: -CGFloat(layout.overlap), dy: -CGFloat(layout.overlap)).intersection(frame)
                result.append((core, input))
            }
        }
        return result
    }
}
