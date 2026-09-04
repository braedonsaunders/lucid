//
//  DecodedFrameSource.swift
//  Lucid
//
//  Turns decoded frames sent by the browser into pixel buffers.
//
//  This is the closest equivalent to what NVIDIA's video super resolution
//  receives: the frame as the decoder produced it, at its own resolution,
//  before the page stretches it to fit the player. Reading pixels back off the
//  screen instead would mean working on an image the browser has already
//  upscaled and resampled, and no amount of processing recovers what that
//  throws away.
//

import CoreMedia
import CoreVideo
import Foundation

final class DecodedFrameSource: @unchecked Sendable {
    private let lock = NSLock()
    private var pool: CVPixelBufferPool?
    private var poolSize = (width: 0, height: 0, format: OSType(0))
    private var continuation: AsyncStream<CapturedFrame>.Continuation?
    private(set) var lastSize = CGSize.zero
    private(set) var frameCount = 0
    /// Where this frame belongs in the browser window, in points. A decoded
    /// frame is the whole video, so it maps onto the video box exactly; without
    /// this the overlay would place it as if it covered only its own pixel
    /// count in the window's top-left corner.
    private var contentRect = CGRect.zero

    func setContentRect(_ rect: CGRect) {
        lock.lock(); contentRect = rect; lock.unlock()
    }

    /// Newest-frame-wins: a late frame is worse than a dropped one.
    func stream() -> AsyncStream<CapturedFrame> {
        AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            lock.lock(); self.continuation = continuation; lock.unlock()
        }
    }

    func finish() {
        lock.lock(); continuation?.finish(); continuation = nil; lock.unlock()
    }

    /// Accepts one frame from the bridge. Formats follow the WebCodecs names.
    func accept(_ frame: DecodedFrame) {
        let width = frame.header.w, height = frame.header.h
        guard width >= 16, height >= 16 else { return }
        let format: OSType
        switch frame.header.format.uppercased() {
        case "I420", "I420A": format = kCVPixelFormatType_420YpCbCr8Planar
        case "NV12": format = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        case "RGBA", "RGBX": format = kCVPixelFormatType_32BGRA
        case "BGRA", "BGRX": format = kCVPixelFormatType_32BGRA
        default: return
        }

        guard let buffer = makeBuffer(width: width, height: height, format: format) else { return }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        frame.payload.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            let planeCount = max(1, CVPixelBufferGetPlaneCount(buffer))
            for plane in 0..<planeCount {
                guard plane < frame.header.planes.count else { break }
                let source = frame.header.planes[plane]
                // A corrupt plane header must not read outside the payload:
                // clamp the copy to what is actually there, row by row.
                guard source.offset >= 0, source.stride > 0,
                      source.offset < raw.count else { continue }
                let destination = CVPixelBufferIsPlanar(buffer)
                    ? CVPixelBufferGetBaseAddressOfPlane(buffer, plane)
                    : CVPixelBufferGetBaseAddress(buffer)
                guard let destination else { continue }
                let destinationStride = CVPixelBufferIsPlanar(buffer)
                    ? CVPixelBufferGetBytesPerRowOfPlane(buffer, plane)
                    : CVPixelBufferGetBytesPerRow(buffer)
                let rows = CVPixelBufferIsPlanar(buffer)
                    ? CVPixelBufferGetHeightOfPlane(buffer, plane)
                    : CVPixelBufferGetHeight(buffer)
                let copyBytes = min(source.stride, destinationStride)
                for row in 0..<rows {
                    let (rowOffset, rowOverflow) = source.offset.addingReportingOverflow(row * source.stride)
                    guard !rowOverflow, rowOffset >= 0 else { break }
                    let (endOffset, endOverflow) = rowOffset.addingReportingOverflow(copyBytes)
                    guard !endOverflow, endOffset <= raw.count else { break }
                    let from = base.advanced(by: rowOffset)
                    memcpy(destination.advanced(by: row * destinationStride), from, copyBytes)
                }
            }
        }

        if format == kCVPixelFormatType_32BGRA {
            swizzleRGBAToBGRA(buffer)
        }
        let prepared = TiledVideoToolboxUpscaler.prepareSource(buffer)

        lock.lock()
        let box = contentRect
        lock.unlock()
        let captured = CapturedFrame(
            surface: unsafeBitCast(CVPixelBufferGetIOSurface(prepared)!.takeUnretainedValue(), to: IOSurface.self),
            pixelBuffer: prepared,
            presentationTimestamp: CMTime(value: CMTimeValue(frame.header.ts), timescale: 1_000_000),
            contentRect: CGRect(x: 0, y: 0, width: width, height: height),
            contentScale: 1, scaleFactor: 1,
            sourceRect: box.width > 1 ? box : CGRect(x: 0, y: 0, width: width, height: height)
        )
        lock.lock()
        lastSize = CGSize(width: width, height: height)
        frameCount += 1
        let sink = continuation
        lock.unlock()
        sink?.yield(captured)
    }

    /// The canvas fallback hands over RGBA; Core Video wants BGRA.
    private func swizzleRGBAToBGRA(_ buffer: CVPixelBuffer) {
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let height = CVPixelBufferGetHeight(buffer), width = CVPixelBufferGetWidth(buffer)
        let bytes = base.bindMemory(to: UInt8.self, capacity: stride * height)
        for row in 0..<height {
            var index = row * stride
            for _ in 0..<width {
                let r = bytes[index]
                bytes[index] = bytes[index + 2]
                bytes[index + 2] = r
                index += 4
            }
        }
    }

    /// Upper bound on a decoded frame the bridge will accept. The bridge
    /// carries 4K RGBA at ~33MB; anything far beyond that is corrupt, not
    /// a video the pipeline could process anyway.
    static let maximumFramePixels = 4096 * 2304

    private func makeBuffer(width: Int, height: Int, format: OSType) -> CVPixelBuffer? {
        guard width * height <= Self.maximumFramePixels else { return nil }
        lock.lock()
        if pool == nil || poolSize != (width, height, format) {
            let attributes: [String: Any] = [
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferPixelFormatTypeKey as String: format,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
            var created: CVPixelBufferPool?
            CVPixelBufferPoolCreate(kCFAllocatorDefault,
                                    [kCVPixelBufferPoolMinimumBufferCountKey as String: 4] as CFDictionary,
                                    attributes as CFDictionary, &created)
            pool = created
            poolSize = (width, height, format)
        }
        let current = pool
        lock.unlock()
        guard let current else { return nil }
        var buffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, current, &buffer) == kCVReturnSuccess else { return nil }
        return buffer
    }
}
