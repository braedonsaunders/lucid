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

import Accelerate
import CoreVideo
import Foundation
import VideoToolbox

final class EnhancedFrameSender: @unchecked Sendable {
    /// Longest edge sent back, in physical pixels. This must be at least the
    /// width the video box occupies on screen: anything less is downscaled here
    /// and scaled back up by the browser, which destroys exactly the fine detail
    /// the reconstruction just produced. The session sets it from the page's
    /// reported box, so nothing larger than the display needs is ever sent.
    var maximumWidth = 1280 {
        didSet { if maximumWidth != oldValue { lock.lock(); pool = nil; lock.unlock() } }
    }

    private let lock = NSLock()
    private var transfer: VTPixelTransferSession?
    private var pool: CVPixelBufferPool?
    private var poolSize = (width: 0, height: 0)
    private var scratch = [UInt8]()

    init() {
        var session: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session)
        if let session {
            VTSessionSetProperty(session, key: kVTPixelTransferPropertyKey_ScalingMode,
                                 value: kVTScalingMode_Trim)
            transfer = session
        }
    }

    /// Converts to RGBA at a sensible size and hands the bytes over.
    /// Returns nil when nothing needs sending.
    func packet(for buffer: CVPixelBuffer, sequence: Int, session: String) -> Data? {
        lock.lock(); defer { lock.unlock() }
        guard let transfer else { return nil }

        let sourceWidth = CVPixelBufferGetWidth(buffer)
        let sourceHeight = CVPixelBufferGetHeight(buffer)
        guard sourceWidth > 0, sourceHeight > 0 else { return nil }
        let scale = min(1.0, Double(maximumWidth) / Double(sourceWidth))
        let width = max(2, Int(Double(sourceWidth) * scale) & ~1)
        let height = max(2, Int(Double(sourceHeight) * scale) & ~1)

        if pool == nil || poolSize != (width, height) {
            let attributes: [String: Any] = [
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
                kCVPixelBufferBytesPerRowAlignmentKey as String: 4,
            ]
            var created: CVPixelBufferPool?
            guard CVPixelBufferPoolCreate(kCFAllocatorDefault,
                                          [kCVPixelBufferPoolMinimumBufferCountKey as String: 3] as CFDictionary,
                                          attributes as CFDictionary, &created) == kCVReturnSuccess
            else { return nil }
            pool = created
            poolSize = (width, height)
        }
        guard let pool else { return nil }
        var converted: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &converted) == kCVReturnSuccess,
              let rgb = converted,
              VTPixelTransferSessionTransferImage(transfer, from: buffer, to: rgb) == noErr
        else { return nil }

        CVPixelBufferLockBaseAddress(rgb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(rgb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(rgb) else { return nil }
        let stride = CVPixelBufferGetBytesPerRow(rgb)
        let rowBytes = width * 4
        let total = rowBytes * height
        if scratch.count != total { scratch = [UInt8](repeating: 0, count: total) }

        // Core Video gives BGRA; a canvas wants RGBA. vImage does the channel
        // swap in one pass rather than a per-pixel loop in JavaScript.
        scratch.withUnsafeMutableBytes { destination in
            guard let destinationBase = destination.baseAddress else { return }
            var source = vImage_Buffer(data: base, height: vImagePixelCount(height),
                                       width: vImagePixelCount(width), rowBytes: stride)
            var target = vImage_Buffer(data: destinationBase, height: vImagePixelCount(height),
                                       width: vImagePixelCount(width), rowBytes: rowBytes)
            var map: [UInt8] = [2, 1, 0, 3]   // BGRA -> RGBA
            vImagePermuteChannels_ARGB8888(&source, &target, &map, vImage_Flags(kvImageNoFlags))
        }

        let header = #"{"session":"\#(session)","w":\#(width),"h":\#(height),"seq":\#(sequence)}"#
        let headerBytes = Array(header.utf8)
        var packet = Data(capacity: 8 + headerBytes.count + total)
        withUnsafeBytes(of: UInt32(0x4c554345).bigEndian) { packet.append(contentsOf: $0) }  // 'LUCE'
        withUnsafeBytes(of: UInt32(headerBytes.count).bigEndian) { packet.append(contentsOf: $0) }
        packet.append(contentsOf: headerBytes)
        packet.append(contentsOf: scratch)
        return packet
    }
}
