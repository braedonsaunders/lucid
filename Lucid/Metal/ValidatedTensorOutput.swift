import CoreML
import CoreVideo
import Foundation

/// Admit the optimized boundary only on a measured backend and after checking
/// the actual bundled models. A failed check leaves image output available.
@available(macOS 26.0, *)
struct ValidatedTensorOutput {
    let model: MLModel
    let packer: CoreMLTensorImagePacker

    static func eligible(deviceName: String, version: OperatingSystemVersion,
                         arguments: [String], environment: [String: String]) -> Bool {
        deviceName == "Apple M4 Pro" && version.majorVersion == 26 &&
            version.minorVersion == 5 && version.patchVersion == 1 &&
            !arguments.contains("--pipeline-ms") &&
            environment["LUCID_MODEL_STEM"] == nil &&
            LearnedUpscaler.stemOverride == nil &&
            environment["LUCID_PIPELINE_MODEL"] == nil &&
            environment["LUCID_DISABLE_TENSOR_OUTPUT"] != "1"
    }

    static func load(reference: MLModel, url: URL, configuration: MLModelConfiguration) throws -> Self? {
        guard configuration.computeUnits == .cpuAndGPU,
              let input = reference.modelDescription.inputDescriptionsByName.first,
              let shape = input.value.imageConstraint,
              let referenceOutput = reference.modelDescription.outputDescriptionsByName.first,
              let imageShape = referenceOutput.value.imageConstraint,
              shape.pixelFormatType == kCVPixelFormatType_32BGRA,
              imageShape.pixelsWide == shape.pixelsWide * 4,
              imageShape.pixelsHigh == shape.pixelsHigh * 4 else { return nil }
        let model = try MLModel(contentsOf: url, configuration: configuration)
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        guard model.modelDescription.inputDescriptionsByName.count == 1,
              model.modelDescription.outputDescriptionsByName.count == 1,
              let candidateInput = model.modelDescription.inputDescriptionsByName[input.key]?.imageConstraint,
              candidateInput.pixelsWide == shape.pixelsWide, candidateInput.pixelsHigh == shape.pixelsHigh,
              candidateInput.pixelFormatType == shape.pixelFormatType,
              let output = model.modelDescription.outputDescriptionsByName[referenceOutput.key]?.multiArrayConstraint,
              output.dataType == .float32,
              output.shape.map(\.intValue) == [1,3,imageShape.pixelsHigh,imageShape.pixelsWide],
              metadata["lucid.checkpoint_sha256"] == "fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65",
              metadata["lucid.output_scale"] == "4", metadata["lucid.output_range"] == "0..255" else { return nil }
        let packer = try CoreMLTensorImagePacker(width: imageShape.pixelsWide, height: imageShape.pixelsHigh)
        for pattern in 0..<3 {
            let matches = try autoreleasepool {
                let frame = try probe(width: shape.pixelsWide, height: shape.pixelsHigh, pattern: pattern)
                let provider = try MLDictionaryFeatureProvider(dictionary: [input.key: MLFeatureValue(pixelBuffer: frame)])
                guard let expected = try reference.prediction(from: provider).featureValue(for: referenceOutput.key)?.imageBufferValue,
                      let array = try model.prediction(from: provider).featureValue(for: referenceOutput.key)?.multiArrayValue else { return false }
                return buffersMatch(expected, try packer.pack(array))
            }
            if !matches { return nil }
        }
        return Self(model: model, packer: packer)
    }

    static func buffersMatch(_ a: CVPixelBuffer, _ b: CVPixelBuffer) -> Bool {
        let width = CVPixelBufferGetWidth(a), height = CVPixelBufferGetHeight(a)
        guard width == CVPixelBufferGetWidth(b), height == CVPixelBufferGetHeight(b),
              CVPixelBufferGetPixelFormatType(a) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetPixelFormatType(b) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetIOSurface(a) == nil, CVPixelBufferGetIOSurface(b) == nil else { return false }
        for mode in [CVAttachmentMode.shouldPropagate, .shouldNotPropagate] {
            let lhs = CVBufferCopyAttachments(a, mode) ?? [:] as CFDictionary
            let rhs = CVBufferCopyAttachments(b, mode) ?? [:] as CFDictionary
            if !CFEqual(lhs, rhs) { return false }
        }
        guard CVPixelBufferLockBaseAddress(a, .readOnly) == kCVReturnSuccess else { return false }
        defer { CVPixelBufferUnlockBaseAddress(a, .readOnly) }
        guard CVPixelBufferLockBaseAddress(b, .readOnly) == kCVReturnSuccess else { return false }
        defer { CVPixelBufferUnlockBaseAddress(b, .readOnly) }
        guard let lhs = CVPixelBufferGetBaseAddress(a), let rhs = CVPixelBufferGetBaseAddress(b) else { return false }
        for y in 0..<height {
            if memcmp(lhs.advanced(by: y*CVPixelBufferGetBytesPerRow(a)),
                      rhs.advanced(by: y*CVPixelBufferGetBytesPerRow(b)), width*4) != 0 { return false }
        }
        return true
    }

    private static func probe(width: Int, height: Int, pattern: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer) == kCVReturnSuccess,
              let buffer else { throw LearnedUpscaler.Failure.pixelBuffer }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let pixels = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        var random: UInt32 = 20260905
        for y in 0..<height { for x in 0..<width {
            let offset = y*CVPixelBufferGetBytesPerRow(buffer)+x*4
            for c in 0..<3 {
                random = random &* 1664525 &+ 1013904223
                let value = pattern == 0 ? Int((random >> 16) & 255) :
                    (pattern == 1 ? (x*255/max(1,width-1)+c*47+y)%256 : ((x/8+y/8+c)%2)*255)
                pixels[offset+c] = UInt8(value)
            }
            pixels[offset+3] = 255
        } }
        return buffer
    }
}
