//
//  FramePresenter.swift
//  Lucid
//
//  Timestamp-paced presentation of enhanced frames through an
//  AVSampleBufferDisplayLayer whose timebase is the host clock, so
//  ScreenCaptureKit presentation timestamps map directly. Frames are shown at
//  capture time plus a fixed latency budget instead of "as soon as possible",
//  which keeps cadence steady. Safe to call from any thread.
//

import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import QuartzCore

final class FramePresenter: @unchecked Sendable {
    let layer: AVSampleBufferDisplayLayer
    private let renderer: AVSampleBufferVideoRenderer
    private let timebase: CMTimebase?
    private let lock = NSLock()
    private var cachedFormat: (width: Int, height: Int, format: OSType, description: CMVideoFormatDescription)?
    /// When a frame was last handed to the display layer. The layer holds its
    /// last image indefinitely, so if frames stop arriving the overlay would sit
    /// frozen over a video that is still playing underneath.
    private(set) var lastPresented = ContinuousClock.now
    private(set) var presentedFrames = 0
    private(set) var droppedFrames = 0
    /// Rect each queued frame was captured from, so the layer can be offset to
    /// wherever those pixels are on screen right now.
    private var queuedRects: [(pts: CMTime, rect: CGRect)] = []
    private var displayedRect: CGRect = .zero

    /// Fixed delay between capture and display. Video is not interactive, so a
    /// small constant budget hides processing jitter without visible lag.
    var latency: CMTime

    init(latencySeconds: Double = 0.040) {
        let layer = AVSampleBufferDisplayLayer()
        layer.videoGravity = .resize
        layer.isOpaque = true
        layer.backgroundColor = nil
        var timebase: CMTimebase?
        CMTimebaseCreateWithSourceClock(
            allocator: kCFAllocatorDefault,
            sourceClock: CMClockGetHostTimeClock(),
            timebaseOut: &timebase
        )
        if let timebase {
            CMTimebaseSetTime(timebase, time: CMClockGetTime(CMClockGetHostTimeClock()))
            CMTimebaseSetRate(timebase, rate: 1.0)
            layer.controlTimebase = timebase
        }
        self.layer = layer
        self.renderer = layer.sampleBufferRenderer
        self.timebase = timebase
        self.latency = CMTime(seconds: latencySeconds, preferredTimescale: 1_000_000_000)
    }

    /// Enqueues `pixelBuffer` to be shown at `capturePTS + latency`.
    func present(_ pixelBuffer: CVPixelBuffer, capturePTS: CMTime, sourceRect: CGRect = .zero) {
        let pts = capturePTS.isNumeric ? CMTimeAdd(capturePTS, latency) : currentTime
        guard let sample = makeSampleBuffer(pixelBuffer, pts: pts) else { return }
        if renderer.isReadyForMoreMediaData {
            renderer.enqueue(sample)
            lock.lock()
            lastPresented = .now
            presentedFrames += 1
            if sourceRect != .zero {
                queuedRects.append((pts, sourceRect))
                if queuedRects.count > 16 { queuedRects.removeFirst(queuedRects.count - 16) }
            }
            lock.unlock()
        } else {
            lock.lock(); droppedFrames += 1; lock.unlock()
        }
    }

    /// The capture rect of the frame currently on screen, or `.zero` if unknown.
    func rectOnScreen() -> CGRect {
        let now = currentTime
        lock.lock(); defer { lock.unlock() }
        while let first = queuedRects.first, CMTimeCompare(first.pts, now) <= 0 {
            displayedRect = first.rect
            queuedRects.removeFirst()
        }
        return displayedRect
    }

    /// Shows `pixelBuffer` immediately (discontinuities such as a new source).
    func presentNow(_ pixelBuffer: CVPixelBuffer) {
        renderer.flush()
        guard let sample = makeSampleBuffer(pixelBuffer, pts: currentTime) else { return }
        if renderer.isReadyForMoreMediaData { renderer.enqueue(sample) }
    }

    func flush() {
        renderer.flush()
        lock.lock(); queuedRects.removeAll(); displayedRect = .zero; lock.unlock()
    }

    var currentTime: CMTime {
        guard let timebase else { return CMClockGetTime(CMClockGetHostTimeClock()) }
        return CMTimebaseGetTime(timebase)
    }

    private func makeSampleBuffer(_ pixelBuffer: CVPixelBuffer, pts: CMTime) -> CMSampleBuffer? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)

        lock.lock()
        var description = cachedFormat
            .flatMap { $0.width == width && $0.height == height && $0.format == format ? $0.description : nil }
        if let cached = description, !CMVideoFormatDescriptionMatchesImageBuffer(cached, imageBuffer: pixelBuffer) {
            description = nil
        }
        if description == nil {
            var created: CMVideoFormatDescription?
            CMVideoFormatDescriptionCreateForImageBuffer(
                allocator: kCFAllocatorDefault,
                imageBuffer: pixelBuffer,
                formatDescriptionOut: &created
            )
            if let created {
                cachedFormat = (width, height, format, created)
                description = created
            }
        }
        lock.unlock()
        guard let description else { return nil }

        var timing = CMSampleTimingInfo(duration: .invalid, presentationTimeStamp: pts, decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: description,
            sampleTiming: &timing,
            sampleBufferOut: &sample
        )
        return sample
    }
}
