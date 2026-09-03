//
//  CapturedFrame.swift
//  Lucid
//
//  One video frame moving through the pipeline, pairing the pixel storage
//  with where on screen those pixels came from. The presenter needs both:
//  the buffer to enhance, and the window-local rect it was read from so the
//  enhanced image can be pinned back over the live video box.
//

import CoreMedia
import CoreVideo
import IOSurface
import ScreenCaptureKit

/// A single frame plus the provenance the overlay needs to place it.
struct CapturedFrame: @unchecked Sendable {
    /// Backing store shared with the GPU; kept alive with the buffer.
    let surface: IOSurface
    let pixelBuffer: CVPixelBuffer
    let presentationTimestamp: CMTime
    /// What ScreenCaptureKit says it handed over this tick.
    let contentRect: CGRect
    let contentScale: CGFloat
    let scaleFactor: CGFloat
    /// Window-local rect (points, top-left origin) the pixels were read from.
    /// Empty until the controller programs a crop; presentation treats empty
    /// as "the whole buffer".
    var sourceRect: CGRect = .zero

    var size: CGSize { contentRect.size }

    /// Direct construction for frames that did not come off the screen, e.g.
    /// decoded bytes rebuilt by the bridge. The caller supplies the box the
    /// frame maps onto; an empty box means "the full buffer".
    init(
        surface: IOSurface,
        pixelBuffer: CVPixelBuffer,
        presentationTimestamp: CMTime,
        contentRect: CGRect,
        contentScale: CGFloat,
        scaleFactor: CGFloat,
        sourceRect: CGRect = .zero
    ) {
        self.surface = surface
        self.pixelBuffer = pixelBuffer
        self.presentationTimestamp = presentationTimestamp
        self.contentRect = contentRect
        self.contentScale = contentScale
        self.scaleFactor = scaleFactor
        self.sourceRect = sourceRect
    }

    /// Reads one ScreenCaptureKit sample into a frame. Yields nil for
    /// anything that is not a finished video frame: the stream marks
    /// idle ticks complete-but-empty and flags dropped content with a
    /// non-complete status, and neither is safe to enhance.
    static func from(sampleBuffer: CMSampleBuffer, sourceRect: CGRect) -> CapturedFrame? {
        guard sampleBuffer.isValid,
              let firstAttachment = CMSampleBufferGetSampleAttachmentsArray(
                sampleBuffer, createIfNecessary: false
              ) as? [[SCStreamFrameInfo: Any]],
              let info = firstAttachment.first,
              let rawStatus = info[SCStreamFrameInfo.status] as? Int,
              SCFrameStatus(rawValue: rawStatus) == .complete,
              let pixelBuffer = sampleBuffer.imageBuffer,
              let rawSurface = CVPixelBufferGetIOSurface(pixelBuffer)?.takeUnretainedValue(),
              let boxDictionary = info[.contentRect] as? [String: CGFloat],
              let contentScale = info[.contentScale] as? CGFloat,
              let scaleFactor = info[.scaleFactor] as? CGFloat
        else { return nil }

        let box = CGRect(
            x: boxDictionary["X"] ?? 0,
            y: boxDictionary["Y"] ?? 0,
            width: boxDictionary["Width"] ?? 0,
            height: boxDictionary["Height"] ?? 0
        )
        guard box.width > 0, box.height > 0 else { return nil }

        return CapturedFrame(
            surface: unsafeBitCast(rawSurface, to: IOSurface.self),
            pixelBuffer: pixelBuffer,
            presentationTimestamp: CMSampleBufferGetPresentationTimeStamp(sampleBuffer),
            contentRect: box,
            contentScale: contentScale,
            scaleFactor: scaleFactor,
            sourceRect: sourceRect
        )
    }
}
