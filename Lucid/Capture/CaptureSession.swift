//
//  CaptureSession.swift
//  Lucid
//
//  ScreenCaptureKit window capture of one rectangle, delivered as an async
//  stream with newest-frame buffering (bounded backpressure). The rectangle,
//  output size and pixel format can be updated live without restarting.
//

import CoreMedia
import CoreVideo
import Foundation
import ScreenCaptureKit

@available(macOS 14.0, *)
final class CaptureSession: @unchecked Sendable {
    struct Configuration: Equatable, Sendable {
        var windowID: CGWindowID
        /// Window-local crop in points (top-left origin), nil for the whole window.
        var sourceRect: CGRect?
        /// Output buffer size in pixels.
        var outputSize: CGSize
        var pixelFormat: OSType
        var frameRate: Int
    }

    enum CaptureError: Error {
        case windowNotFound
        case notRunning
    }

    /// Rect currently programmed into the stream, read by the output callback
    /// so every frame knows where it came from.
    final class RectBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value: CGRect
        init(_ value: CGRect) { self.value = value }
        var rect: CGRect {
            get { lock.lock(); defer { lock.unlock() }; return value }
            set { lock.lock(); value = newValue; lock.unlock() }
        }
    }

    private let rectBox: RectBox
    private var stream: SCStream?
    private var output: StreamOutput?
    private var continuation: AsyncThrowingStream<CapturedFrame, Error>.Continuation?
    private(set) var configuration: Configuration

    init(configuration: Configuration) {
        self.configuration = configuration
        self.rectBox = RectBox(configuration.sourceRect ?? .zero)
    }

    func start() async throws -> AsyncThrowingStream<CapturedFrame, Error> {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        guard let window = content.windows.first(where: { $0.windowID == configuration.windowID }) else {
            throw CaptureError.windowNotFound
        }
        let filter = SCContentFilter(desktopIndependentWindow: window)
        let (frames, continuation) = AsyncThrowingStream.makeStream(
            of: CapturedFrame.self, bufferingPolicy: .bufferingNewest(1)
        )
        self.continuation = continuation
        let output = StreamOutput(continuation: continuation, rectBox: rectBox)
        self.output = output

        let stream = SCStream(filter: filter, configuration: makeStreamConfiguration(), delegate: nil)
        try stream.addStreamOutput(
            output, type: .screen,
            sampleHandlerQueue: DispatchQueue(label: "com.lucid.capture", qos: .userInteractive)
        )
        try await stream.startCapture()
        self.stream = stream
        return frames
    }

    func update(_ configuration: Configuration) async throws {
        guard let stream else { throw CaptureError.notRunning }
        guard configuration != self.configuration else { return }
        self.configuration = configuration
        rectBox.rect = configuration.sourceRect ?? .zero
        try await stream.updateConfiguration(makeStreamConfiguration())
    }

    func stop() async {
        if let stream {
            try? await stream.stopCapture()
        }
        stream = nil
        continuation?.finish()
        continuation = nil
        output = nil
    }

    private func makeStreamConfiguration() -> SCStreamConfiguration {
        let config = SCStreamConfiguration()
        config.width = Int(configuration.outputSize.width)
        config.height = Int(configuration.outputSize.height)
        if let rect = configuration.sourceRect { config.sourceRect = rect }
        config.pixelFormat = configuration.pixelFormat
        config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(configuration.frameRate))
        config.queueDepth = 4
        config.showsCursor = false
        config.capturesAudio = false
        config.scalesToFit = true
        config.ignoreShadowsSingleWindow = true
        config.ignoreGlobalClipSingleWindow = true
        // Chrome keeps its tab strip and toolbar in separate windows that
        // overlap the top of the content window. With child windows included
        // (the default) ScreenCaptureKit captures the union of them all and
        // squeezes it into the requested size, which offsets every coordinate,
        // scales the page down and paints browser UI into the video.
        config.includeChildWindows = false
        config.captureResolution = .best
        // Without an explicit matrix the display layer has to guess how to turn
        // YUV back into RGB, which shows up as washed-out colour.
        config.colorMatrix = CGDisplayStream.yCbCrMatrix_ITU_R_709_2
        config.colorSpaceName = CGColorSpace.sRGB
        return config
    }
}

@available(macOS 12.3, *)
private final class StreamOutput: NSObject, SCStreamOutput {
    private let continuation: AsyncThrowingStream<CapturedFrame, Error>.Continuation
    private let rectBox: CaptureSession.RectBox

    init(continuation: AsyncThrowingStream<CapturedFrame, Error>.Continuation, rectBox: CaptureSession.RectBox) {
        self.continuation = continuation
        self.rectBox = rectBox
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen,
              let frame = CapturedFrame.from(sampleBuffer: sampleBuffer, sourceRect: rectBox.rect)
        else { return }
        continuation.yield(frame)
    }
}
