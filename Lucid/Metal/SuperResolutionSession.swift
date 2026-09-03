//
//  SuperResolutionSession.swift
//  Lucid
//
//  One Apple low-latency super-resolution session for one fixed tile size.
//  An actor: all VideoToolbox state is confined here, and several sessions
//  run concurrently for different tiles of the same frame.
//

import CoreVideo
import Foundation
@preconcurrency import VideoToolbox

@available(macOS 26.0, *)
actor SuperResolutionSession {
    enum Failure: Error {
        case unsupported
        case pool
        case buffer
        case frame
        case closed
    }

    let inputWidth: Int
    let inputHeight: Int
    let scale: Float
    nonisolated let configuration: VTLowLatencySuperResolutionScalerConfiguration
    private let processor = VTFrameProcessor()
    private let inputPool: CVPixelBufferPool
    private let outputPool: CVPixelBufferPool
    private var started = false
    private var closed = false
    private var inFlight = 0
    private(set) var processedFrames = 0

    /// `preferredScale` picks between the factors the hardware reports. The
    /// scaler offers 2 and 4; going straight to 4 is one pass instead of two,
    /// and the model was trained for that factor rather than for being run
    /// twice over its own output.
    init(inputWidth: Int, inputHeight: Int, preferredScale: Float = 2) throws {
        guard VTLowLatencySuperResolutionScalerConfiguration.isSupported else { throw Failure.unsupported }
        let factors = VTLowLatencySuperResolutionScalerConfiguration.supportedScaleFactors(
            frameWidth: inputWidth, frameHeight: inputHeight
        )
        guard let scale = factors.first(where: { abs($0 - preferredScale) < 0.01 })
            ?? factors.first(where: { abs($0 - 2) < 0.01 }) else { throw Failure.unsupported }
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.scale = scale
        configuration = VTLowLatencySuperResolutionScalerConfiguration(
            frameWidth: inputWidth, frameHeight: inputHeight, scaleFactor: scale
        )
        inputPool = try Self.makePool(configuration.sourcePixelBufferAttributes, minimum: 3)
        outputPool = try Self.makePool(configuration.destinationPixelBufferAttributes, minimum: 3)
    }

    nonisolated var inputPixelFormat: OSType {
        configuration.sourcePixelBufferAttributes[kCVPixelBufferPixelFormatTypeKey as String] as? OSType
            ?? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
    }

    /// A buffer the caller may fill (by Metal blit) and hand back to `process`.
    func makeInputBuffer() throws -> CVPixelBuffer {
        try Self.makeBuffer(inputPool)
    }

    func process(_ input: CVPixelBuffer, pts: CMTime) async throws -> CVPixelBuffer {
        guard !closed else { throw Failure.closed }
        if !started {
            try processor.startSession(configuration: configuration)
            started = true
        }
        let output = try Self.makeBuffer(outputPool)
        guard let source = VTFrameProcessorFrame(buffer: input, presentationTimeStamp: pts),
              let destination = VTFrameProcessorFrame(buffer: output, presentationTimeStamp: pts)
        else { throw Failure.frame }
        let parameters = VTLowLatencySuperResolutionScalerParameters(sourceFrame: source, destinationFrame: destination)
        inFlight += 1
        defer { inFlight -= 1 }
        try await processor.process(parameters: parameters)
        processedFrames += 1
        return output
    }

    /// Ends the VideoToolbox session once no frame is being processed. Ending
    /// while a frame is in flight crashes inside VideoToolbox.
    func end() async {
        closed = true
        while inFlight > 0 {
            try? await Task.sleep(for: .milliseconds(2))
        }
        if started {
            processor.endSession()
            started = false
        }
    }

    private static func makePool(_ attributes: [String: Any], minimum: Int) throws -> CVPixelBufferPool {
        var merged = attributes
        merged[kCVPixelBufferMetalCompatibilityKey as String] = true
        if merged[kCVPixelBufferIOSurfacePropertiesKey as String] == nil {
            merged[kCVPixelBufferIOSurfacePropertiesKey as String] = [:] as [String: Any]
        }
        var pool: CVPixelBufferPool?
        let poolAttributes: [String: Any] = [kCVPixelBufferPoolMinimumBufferCountKey as String: minimum]
        let status = CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttributes as CFDictionary, merged as CFDictionary, &pool)
        guard status == kCVReturnSuccess, let pool else { throw Failure.pool }
        return pool
    }

    private static func makeBuffer(_ pool: CVPixelBufferPool) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer)
        guard status == kCVReturnSuccess, let buffer else { throw Failure.buffer }
        return buffer
    }
}
