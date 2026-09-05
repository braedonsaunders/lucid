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
import VideoToolbox

final class DecodedFrameSource: @unchecked Sendable {
    private let lock = NSLock()
    private var transfer: VTPixelTransferSession?
    private var normalizedPool: CVPixelBufferPool?
    private var normalizedSize = CGSize.zero
    private var pool: CVPixelBufferPool?
    private var poolSize = (width: 0, height: 0, format: OSType(0))
    private var continuation: AsyncStream<CapturedFrame>.Continuation?
    private var lastDimensions = CGSize.zero
    private var acceptedCount = 0
    var lastSize: CGSize { lock.lock(); defer { lock.unlock() }; return lastDimensions }
    var frameCount: Int { lock.lock(); defer { lock.unlock() }; return acceptedCount }
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
        guard Self.validLayout(frame.header, payloadCount: frame.payload.count),
              frame.header.colorSpace?.isHDR != true else { return }
        let format: OSType
        switch frame.header.format.uppercased() {
        case "I420", "I420A": format = frame.header.colorSpace?.fullRange == true
            ? kCVPixelFormatType_420YpCbCr8PlanarFullRange : kCVPixelFormatType_420YpCbCr8Planar
        case "NV12": format = frame.header.colorSpace?.fullRange == true
            ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        case "RGBA", "RGBX": format = kCVPixelFormatType_32BGRA
        case "BGRA", "BGRX": format = kCVPixelFormatType_32BGRA
        default: return
        }

        guard let buffer = makeBuffer(width: width, height: height, format: format) else { return }
        CVPixelBufferLockBaseAddress(buffer, [])


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
                let planeWidth = CVPixelBufferIsPlanar(buffer) ? CVPixelBufferGetWidthOfPlane(buffer, plane) : width
                let bytesPerPixel = !CVPixelBufferIsPlanar(buffer) ? 4 : (plane == 1 && CVPixelBufferGetPlaneCount(buffer) == 2 ? 2 : 1)
                let copyBytes = min(planeWidth * bytesPerPixel, destinationStride)
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

        if ["RGBA", "RGBX"].contains(frame.header.format.uppercased()) {
            swizzleRGBAToBGRA(buffer)
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        (frame.header.colorSpace ?? .rec709).apply(to: buffer)
        guard let normalized = normalize(buffer) else { return }
        let prepared = TiledVideoToolboxUpscaler.prepareSource(normalized)

        lock.lock()
        let box = contentRect
        lock.unlock()
        var captured = CapturedFrame(
            surface: unsafeBitCast(CVPixelBufferGetIOSurface(prepared)!.takeUnretainedValue(), to: IOSurface.self),
            pixelBuffer: prepared,
            presentationTimestamp: CMTime(value: CMTimeValue(frame.header.ts), timescale: 1_000_000),
            contentRect: CGRect(x: 0, y: 0, width: width, height: height),
            contentScale: 1, scaleFactor: 1,
            sourceRect: box.width > 1 ? box : CGRect(x: 0, y: 0, width: width, height: height)
        )
        captured.fromBrowser = true
        captured.sequence = frame.header.seq
        captured.captureTimeMilliseconds = frame.header.captureTime ?? 0
        lock.lock()
        lastDimensions = CGSize(width: width, height: height)
        acceptedCount += 1
        let sink = continuation
        lock.unlock()
        sink?.yield(captured)
    }

    /// Reject incomplete and overflowing plane layouts before touching pooled memory.
    static func validLayout(_ h: DecodedFrame.Header, payloadCount: Int) -> Bool {
        guard h.w >= 16, h.h >= 16, h.w <= 4096, h.h <= 2304,
              h.ts.isFinite, h.ts >= 0, h.ts < Double(Int64.max) else { return false }
        let cw = (h.w + 1) / 2, ch = (h.h + 1) / 2
        let dimensions: [(Int, Int)]
        switch h.format.uppercased() {
        case "NV12": dimensions = [(h.w, h.h), (cw * 2, ch)]
        case "I420", "I420A": dimensions = [(h.w, h.h), (cw, ch), (cw, ch)]
        case "RGBA", "RGBX", "BGRA", "BGRX": dimensions = [(h.w * 4, h.h)]
        default: return false
        }
        guard h.planes.count >= dimensions.count else { return false }
        for (plane, (bytes, rows)) in zip(h.planes, dimensions) {
            guard plane.offset >= 0, plane.stride >= bytes else { return false }
            let (rowOffset, overflow) = (rows - 1).multipliedReportingOverflow(by: plane.stride)
            let (end, overflow2) = plane.offset.addingReportingOverflow(rowOffset)
            guard !overflow, !overflow2, end <= payloadCount, bytes <= payloadCount - end else { return false }
        }
        return true
    }

    /// All enhancement shaders have one explicit contract: video-range Rec.709 NV12.
    private func normalize(_ source: CVPixelBuffer) -> CVPixelBuffer? {
        let w = CVPixelBufferGetWidth(source), h = CVPixelBufferGetHeight(source)
        let color = VideoColorInfo.read(from: source)
        if CVPixelBufferGetPixelFormatType(source) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
           color.matrix == "bt709", color.primaries == "bt709", color.transfer == "bt709" { return source }
        if transfer == nil {
            VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transfer)
            if let transfer {
                VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DestinationColorPrimaries, value: kCVImageBufferColorPrimaries_ITU_R_709_2)
                VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DestinationTransferFunction, value: kCVImageBufferTransferFunction_ITU_R_709_2)
                VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DestinationYCbCrMatrix, value: kCVImageBufferYCbCrMatrix_ITU_R_709_2)
            }
        }
        guard let transfer else { return nil }
        if normalizedPool == nil || normalizedSize != CGSize(width: w, height: h) {
            normalizedPool = nil
            let attrs: [String: Any] = [kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:], kCVPixelBufferMetalCompatibilityKey as String: true]
            CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &normalizedPool)
            normalizedSize = CGSize(width: w, height: h)
        }
        guard let pool = normalizedPool else { return nil }
        var output: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &output) == kCVReturnSuccess, let output,
              VTPixelTransferSessionTransferImage(transfer, from: source, to: output) == noErr else { return nil }
        VideoColorInfo.rec709.apply(to: output)
        return output
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
