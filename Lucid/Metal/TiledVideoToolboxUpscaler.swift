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

    /// Fills in any missing Rec. 709 tags, and always writes chroma siting
    /// from `chromaSitingLeft`. Whether 4:2:0 chroma is treated as left-sited
    /// (H.264 / VP9) or centre-sited (the untagged default).
    nonisolated(unsafe) static var chromaSitingLeft = true

    static func ensureColorDescription(_ buffer: CVPixelBuffer) {
        let defaults: [(CFString, CFString)] = [
            (kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2),
            (kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2),
            (kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2),
        ]
        for (key, value) in defaults where CVBufferCopyAttachment(buffer, key, nil) == nil {
            CVBufferSetAttachment(buffer, key, value, .shouldPropagate)
        }
        // Siting is a control, not a default. Fill-if-missing made the toggle
        // inert: AVAssetReader and recycled pool buffers already carry a
        // chroma location. Always write these two. That is not enough on its
        // own — VT samples 4:2:0 from the source, so LearnedUpscaler tags the
        // source before the 420→RGB convert. Tagging only this buffer after
        // the transfer leaves the attachment correct and the pixels wrong.
        let siting = chromaSitingLeft
            ? kCVImageBufferChromaLocation_Left
            : kCVImageBufferChromaLocation_Center
        CVBufferSetAttachment(buffer, kCVImageBufferChromaLocationTopFieldKey, siting, .shouldPropagate)
        CVBufferSetAttachment(buffer, kCVImageBufferChromaLocationBottomFieldKey, siting, .shouldPropagate)
    }

    /// Tag plus, when the flag is centre, a real chroma resample.
    ///
    /// `VTPixelTransferSession` and `CIImage` do not honour the chroma
    /// location attachment. Tagging the source made `--pipeline-ms` print
    /// Left vs Center and left `--bench` writing byte-identical PNGs — the
    /// sticker moved, the samples did not. AVAssetReader buffers are also
    /// often read-only, so this copies to a writable buffer first.
    ///
    /// Left is the H.264 / VP9 site and is a no-op on the planes. Centre
    /// shifts chroma half a luma pixel (0.25 of a chroma texel), which is
    /// the whole difference the control is supposed to be.
    static func prepareSource(_ source: CVPixelBuffer) -> CVPixelBuffer {
        let prepared = writableCopy(source) ?? source
        ensureColorDescription(prepared)
        if !chromaSitingLeft { shiftChromaToCenter(prepared) }
        return prepared
    }

    private static func writableCopy(_ source: CVPixelBuffer) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        let format = CVPixelBufferGetPixelFormatType(source)
        let attributes: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        var copy: CVPixelBuffer?
        guard CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, format,
            attributes as CFDictionary, &copy
        ) == kCVReturnSuccess, let copy else { return nil }
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(copy, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(copy, [])
        }
        let planes = max(1, CVPixelBufferGetPlaneCount(source))
        for plane in 0..<planes {
            let src = (planes > 1
                ? CVPixelBufferGetBaseAddressOfPlane(source, plane)
                : CVPixelBufferGetBaseAddress(source))
            let dst = (planes > 1
                ? CVPixelBufferGetBaseAddressOfPlane(copy, plane)
                : CVPixelBufferGetBaseAddress(copy))
            guard let src, let dst else { continue }
            let srcStride = planes > 1
                ? CVPixelBufferGetBytesPerRowOfPlane(source, plane)
                : CVPixelBufferGetBytesPerRow(source)
            let dstStride = planes > 1
                ? CVPixelBufferGetBytesPerRowOfPlane(copy, plane)
                : CVPixelBufferGetBytesPerRow(copy)
            let rows = planes > 1
                ? CVPixelBufferGetHeightOfPlane(source, plane)
                : CVPixelBufferGetHeight(source)
            let bytes = min(srcStride, dstStride)
            for row in 0..<rows {
                memcpy(dst.advanced(by: row * dstStride), src.advanced(by: row * srcStride), bytes)
            }
        }
        if let attachments = CVBufferCopyAttachments(source, .shouldPropagate) {
            CVBufferSetAttachments(copy, attachments, .shouldPropagate)
        }
        return copy
    }

    /// 0.5 luma pixel = 0.25 of a 4:2:0 chroma texel.
    private static func shiftChromaToCenter(_ buffer: CVPixelBuffer) {
        let planes = CVPixelBufferGetPlaneCount(buffer)
        guard planes >= 2 else { return }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) else { return }
        let width = CVPixelBufferGetWidthOfPlane(buffer, 1)
        let height = CVPixelBufferGetHeightOfPlane(buffer, 1)
        let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        let bytesPerPixel = 2
        let src = base.assumingMemoryBound(to: UInt8.self)
        var scratch = [UInt8](repeating: 0, count: stride * height)
        scratch.withUnsafeMutableBufferPointer { dst in
            for y in 0..<height {
                let fy = min(Float(height - 1), Float(y) + 0.25)
                let y0 = Int(fy)
                let y1 = min(height - 1, y0 + 1)
                let wy = fy - Float(y0)
                for x in 0..<width {
                    let fx = min(Float(width - 1), Float(x) + 0.25)
                    let x0 = Int(fx)
                    let x1 = min(width - 1, x0 + 1)
                    let wx = fx - Float(x0)
                    for c in 0..<bytesPerPixel {
                        let p00 = Float(src[y0 * stride + x0 * bytesPerPixel + c])
                        let p10 = Float(src[y0 * stride + x1 * bytesPerPixel + c])
                        let p01 = Float(src[y1 * stride + x0 * bytesPerPixel + c])
                        let p11 = Float(src[y1 * stride + x1 * bytesPerPixel + c])
                        let value = p00 * (1 - wx) * (1 - wy) + p10 * wx * (1 - wy)
                            + p01 * (1 - wx) * wy + p11 * wx * wy
                        dst[y * stride + x * bytesPerPixel + c] = UInt8(max(0, min(255, value.rounded())))
                    }
                }
            }
            memcpy(base, dst.baseAddress, stride * height)
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
