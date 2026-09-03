import CoreMedia
import CoreVideo
import Darwin
import Foundation
@preconcurrency import VideoToolbox

@available(macOS 26.0, *)
@main
struct SuperResolutionProbe {
    private struct TestSize {
        let name: String
        let width: Int
        let height: Int
    }

    private static let testSizes = [
        TestSize(name: "360p", width: 640, height: 360),
        TestSize(name: "540p", width: 960, height: 540),
        TestSize(name: "half-720p", width: 640, height: 720),
        TestSize(name: "overlap-half-720p", width: 672, height: 720),
        TestSize(name: "overlap-1080p-edge", width: 672, height: 572),
        TestSize(name: "overlap-1080p-center", width: 704, height: 572),
        TestSize(name: "half-1080p", width: 960, height: 1080),
        TestSize(name: "720p", width: 1280, height: 720),
        TestSize(name: "1080p", width: 1920, height: 1080),
        TestSize(name: "1440p", width: 2560, height: 1440),
    ]

    @MainActor
    static func main() async {
        guard #available(macOS 26.0, *) else {
            print("ERROR: macOS 26 or newer is required")
            exit(1)
        }

        let configurationType = VTLowLatencySuperResolutionScalerConfiguration.self
        print("VideoToolbox low-latency super resolution: \(configurationType.isSupported ? "SUPPORTED" : "UNSUPPORTED")")
        guard configurationType.isSupported else { exit(2) }

        let minimum = configurationType.minimumDimensions
        let maximum = configurationType.maximumDimensions
        let minimumText = minimum.map { "\($0.width)x\($0.height)" } ?? "unspecified"
        let maximumText = maximum.map { "\($0.width)x\($0.height)" } ?? "unspecified"
        print("Input dimension range: \(minimumText) ... \(maximumText)")

        var benchmarkCase: (TestSize, Float)?
        for size in testSizes {
            let factors = configurationType.supportedScaleFactors(
                frameWidth: size.width,
                frameHeight: size.height
            )
            let factorText = factors.isEmpty
                ? "none"
                : factors.map { String(format: "%.2fx", $0) }.joined(separator: ", ")
            print("\(size.name) \(size.width)x\(size.height): \(factorText)")
            if let factor = factors.first(where: { $0 >= 1.5 }) ?? factors.first {
                benchmarkCase = (size, factor)
            }
        }

        guard let (size, factor) = benchmarkCase else {
            print("BENCHMARK: skipped because no tested resolution is supported")
            exit(3)
        }

        do {
            try await benchmark(size: size, factor: factor, frames: 90)
        } catch {
            print("BENCHMARK ERROR: \(error)")
            exit(4)
        }
    }

    @MainActor
    private static func benchmark(size: TestSize, factor: Float, frames: Int) async throws {
        let configuration = VTLowLatencySuperResolutionScalerConfiguration(
            frameWidth: size.width,
            frameHeight: size.height,
            scaleFactor: factor
        )

        print("Source attributes: \(configuration.sourcePixelBufferAttributes)")
        print("Destination attributes: \(configuration.destinationPixelBufferAttributes)")
        let sourcePool = try makePool(attributes: configuration.sourcePixelBufferAttributes)
        let destinationPool = try makePool(attributes: configuration.destinationPixelBufferAttributes)
        let processor = VTFrameProcessor()

        print("Starting VideoToolbox session...")
        let sessionStart = ContinuousClock.now
        try processor.startSession(configuration: configuration)
        print("VideoToolbox session started")
        let sessionLoad = ContinuousClock.now - sessionStart
        defer { processor.endSession() }

        var samples: [Double] = []
        for index in 0..<frames {
            let source = try makeBuffer(pool: sourcePool)
            fill(buffer: source, value: UInt8(index & 0xff))
            let destination = try makeBuffer(pool: destinationPool)
            let time = CMTime(value: CMTimeValue(index), timescale: 60)

            guard let sourceFrame = VTFrameProcessorFrame(buffer: source, presentationTimeStamp: time),
                  let destinationFrame = VTFrameProcessorFrame(buffer: destination, presentationTimeStamp: time)
            else {
                throw ProbeError.frameCreationFailed
            }

            let parameters = VTLowLatencySuperResolutionScalerParameters(
                sourceFrame: sourceFrame,
                destinationFrame: destinationFrame
            )
            let start = ContinuousClock.now
            do {
                try await processor.process(parameters: parameters)
            } catch {
                throw ProbeError.processingFailed(frame: index, underlying: error)
            }
            let elapsed = ContinuousClock.now - start
            samples.append(elapsed.seconds)
        }

        let steady = Array(samples.dropFirst(min(10, samples.count / 3)))
        let mean = steady.reduce(0, +) / Double(steady.count)
        let sorted = steady.sorted()
        let p95 = sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * 0.95))]
        let outputWidth = Int(Float(size.width) * factor)
        let outputHeight = Int(Float(size.height) * factor)

        print(String(format: "Session/model load: %.1f ms", sessionLoad.seconds * 1_000))
        print(String(format: "BENCHMARK: %@ %dx%d -> %dx%d (%.2fx)", size.name, size.width, size.height, outputWidth, outputHeight, factor))
        print(String(format: "BENCHMARK: mean %.2f ms/frame (%.1f fps), p95 %.2f ms/frame", mean * 1_000, 1.0 / mean, p95 * 1_000))
    }

    private static func makePool(attributes: [String: any Sendable]) throws -> CVPixelBufferPool {
        var pool: CVPixelBufferPool?
        let status = CVPixelBufferPoolCreate(
            kCFAllocatorDefault,
            [kCVPixelBufferPoolMinimumBufferCountKey as String: 3] as CFDictionary,
            attributes as CFDictionary,
            &pool
        )
        guard status == kCVReturnSuccess, let pool else {
            throw ProbeError.poolCreationFailed(status)
        }
        return pool
    }

    private static func makeBuffer(pool: CVPixelBufferPool) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer)
        guard status == kCVReturnSuccess, let buffer else {
            throw ProbeError.bufferCreationFailed(status)
        }
        return buffer
    }

    private static func fill(buffer: CVPixelBuffer, value: UInt8) {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        if CVPixelBufferIsPlanar(buffer) {
            for plane in 0..<CVPixelBufferGetPlaneCount(buffer) {
                guard let base = CVPixelBufferGetBaseAddressOfPlane(buffer, plane) else { continue }
                memset(base, plane == 0 ? Int32(value) : 128, CVPixelBufferGetBytesPerRowOfPlane(buffer, plane) * CVPixelBufferGetHeightOfPlane(buffer, plane))
            }
        } else if let base = CVPixelBufferGetBaseAddress(buffer) {
            memset(base, Int32(value), CVPixelBufferGetBytesPerRow(buffer) * CVPixelBufferGetHeight(buffer))
        }
    }

    private enum ProbeError: Error {
        case poolCreationFailed(CVReturn)
        case bufferCreationFailed(CVReturn)
        case frameCreationFailed
        case processingFailed(frame: Int, underlying: Error)
    }
}

private extension Duration {
    var seconds: Double {
        let parts = components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
