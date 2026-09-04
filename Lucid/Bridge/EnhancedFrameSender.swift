//
//  EnhancedFrameSender.swift
//  Lucid
//
//  Sends the enhanced frame back to the page so the page can draw it.
//
//  A separate overlay window can never be locked to the page: it trails the
//  scroll by a frame, it has to guess where the viewport clips, and it sits on
//  top of the video's own controls. NVIDIA's video super resolution has none of
//  those problems because the enhanced frame replaces the decoded one inside
//  the browser's compositor, so it is the video rather than something drawn
//  over it. We cannot reach into the compositor, but we can hand the frame back
//  to the page and let the page composite it, which puts it under the same
//  scroll, the same clip and the same stacking as any other element.
//
//  The payload is NV12 (4:2:0), not RGBA. The frame entered as 420v, the
//  model consumed 420v, and every chroma sample in an RGBA buffer above
//  one-in-four is interpolation we invented and then paid 4 bytes a pixel
//  to transmit. NV12 is 1.5 bytes a pixel: 3456x1920 goes from 25.31 MB
//  to 9.49 MB, which is the size that loopback already delivered without
//  drops. Alpha was a constant nobody read.
//

import CoreVideo
import Foundation
import VideoToolbox

final class EnhancedFrameSender: @unchecked Sendable {
    /// Physical width of the video box, from the page. Frames within
    /// `sendWholeFactor` of this are sent at reconstructed size; only a
    /// much larger frame is scaled down to the box.
    var maximumWidth = 1280 {
        didSet { if maximumWidth != oldValue { lock.lock(); pool = nil; lock.unlock() } }
    }

    private let lock = NSLock()
    private var transfer: VTPixelTransferSession?
    private var pool: CVPixelBufferPool?
    private var poolSize = (width: 0, height: 0)
    private var scratch = [UInt8]()
    /// Send the reconstructed frame whole when it is at most this times the
    /// box. Same-size NV12 is a plane copy (0.54 ms at 3456x1920); VT 420→420
    /// downscale costs 3–4 ms and throws away the detail the model just made.
    /// The page composites into the video rect anyway, so the browser's scaler
    /// is free. 1.5 leaves 3456→2560 unscaled (1.35x) and still scales
    /// 3456→1920 (1.80x).
    static let sendWholeFactor = 1.5

    init() {
        var session: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session)
        if let session {
            VTSessionSetProperty(session, key: kVTPixelTransferPropertyKey_ScalingMode,
                                 value: kVTScalingMode_Trim)
            transfer = session
        }
    }

    /// Packs NV12 at the on-screen size. Returns nil when nothing needs sending.
    func packet(for buffer: CVPixelBuffer, sequence: Int, session: String) -> Data? {
        lock.lock(); defer { lock.unlock() }

        let sourceWidth = CVPixelBufferGetWidth(buffer)
        let sourceHeight = CVPixelBufferGetHeight(buffer)
        guard sourceWidth > 0, sourceHeight > 0 else { return nil }
        let (width, height) = Self.sendSize(
            sourceWidth: sourceWidth, sourceHeight: sourceHeight, boxWidth: maximumWidth
        )

        let nv12: CVPixelBuffer
        if width == sourceWidth, height == sourceHeight, Self.is420(buffer) {
            nv12 = buffer
        } else {
            guard let transfer else { return nil }
            if pool == nil || poolSize != (width, height) {
                let attributes: [String: Any] = [
                    kCVPixelBufferWidthKey as String: width,
                    kCVPixelBufferHeightKey as String: height,
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                    kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
                    kCVPixelBufferMetalCompatibilityKey as String: true,
                ]
                var created: CVPixelBufferPool?
                guard CVPixelBufferPoolCreate(
                    kCFAllocatorDefault,
                    [kCVPixelBufferPoolMinimumBufferCountKey as String: 3] as CFDictionary,
                    attributes as CFDictionary, &created
                ) == kCVReturnSuccess else { return nil }
                pool = created
                poolSize = (width, height)
            }
            guard let pool else { return nil }
            var converted: CVPixelBuffer?
            guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &converted) == kCVReturnSuccess,
                  let converted,
                  VTPixelTransferSessionTransferImage(transfer, from: buffer, to: converted) == noErr
            else { return nil }
            nv12 = converted
        }

        guard let packed = Self.tightNV12(nv12, width: width, height: height, into: &scratch) else { return nil }
        let header = #"{"session":"\#(session)","w":\#(width),"h":\#(height),"seq":\#(sequence),"format":"NV12"}"#
        let headerBytes = Array(header.utf8)
        var packet = Data(capacity: 8 + headerBytes.count + packed)
        withUnsafeBytes(of: UInt32(0x4c554345).bigEndian) { packet.append(contentsOf: $0) }  // 'LUCE'
        withUnsafeBytes(of: UInt32(headerBytes.count).bigEndian) { packet.append(contentsOf: $0) }
        packet.append(contentsOf: headerBytes)
        packet.append(contentsOf: scratch.prefix(packed))
        return packet
    }

    static func sendSize(sourceWidth: Int, sourceHeight: Int, boxWidth: Int) -> (Int, Int) {
        let sendWhole = sourceWidth <= boxWidth
            || Double(sourceWidth) <= Double(max(boxWidth, 1)) * sendWholeFactor
        if sendWhole {
            return (max(2, sourceWidth & ~1), max(2, sourceHeight & ~1))
        }
        let width = max(2, boxWidth & ~1)
        let height = max(2, Int((Double(sourceHeight) * Double(width) / Double(sourceWidth)).rounded()) & ~1)
        return (width, height)
    }

    private static func is420(_ buffer: CVPixelBuffer) -> Bool {
        let format = CVPixelBufferGetPixelFormatType(buffer)
        return format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            || format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    }

    /// Tightly packed NV12: Y then interleaved UV, stride = width.
    private static func tightNV12(_ buffer: CVPixelBuffer, width: Int, height: Int, into scratch: inout [UInt8]) -> Int? {
        guard CVPixelBufferGetPlaneCount(buffer) >= 2 else { return nil }
        let yBytes = width * height
        let uvBytes = width * (height / 2)
        let total = yBytes + uvBytes
        if scratch.count != total { scratch = [UInt8](repeating: 0, count: total) }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let yBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
              let uvBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)
        else { return nil }
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        let uvStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        let yHeight = CVPixelBufferGetHeightOfPlane(buffer, 0)
        let uvHeight = CVPixelBufferGetHeightOfPlane(buffer, 1)
        scratch.withUnsafeMutableBytes { dest in
            guard let destBase = dest.baseAddress else { return }
            for row in 0..<min(height, yHeight) {
                memcpy(destBase.advanced(by: row * width), yBase.advanced(by: row * yStride), min(width, yStride))
            }
            let uvDest = destBase.advanced(by: yBytes)
            for row in 0..<min(height / 2, uvHeight) {
                memcpy(uvDest.advanced(by: row * width), uvBase.advanced(by: row * uvStride), min(width, uvStride))
            }
        }
        return total
    }
}
