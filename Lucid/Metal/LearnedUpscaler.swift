//
//  LearnedUpscaler.swift
//  Lucid
//
//  Lucid's upscaler: SPAN, running on the Neural Engine. This is the whole
//  reconstruction path - there is no second engine underneath it.
//
//  Measured against a 1080p reference on compressed 270p source, fine-band
//  correlation with the truth - the metric that separates recovered detail from
//  invented detail:
//
//      SPAN ch48                     0.228
//      SPAN ch28                     0.221
//      lanczos anchor                0.180
//      Apple's scaler                0.174
//      Apple's scaler + our detail   0.152
//
//  Apple's scaler invents a mottled texture in foliage that is not in the
//  source; our sharpening then amplifies it. SPAN is the only option measured
//  that beats a plain Lanczos upscale, and it is also the cleanest to look at.
//  That result is why this file is the only upscaler left.
//

import CoreML
import CoreVideo
import Foundation
import VideoToolbox

@available(macOS 26.0, *)
final class LearnedUpscaler: @unchecked Sendable {
    enum Failure: Error { case noModel, pixelBuffer, prediction }

    /// Input sizes a model has been converted for, with what each measured on
    /// an M4 Pro's Neural Engine. Cost is roughly parameters x low-resolution
    /// pixels, so it grows quickly with the source size.
    struct Variant {
        let width: Int
        let height: Int
        let milliseconds: Double
    }

    /// Every width here is a multiple of 16, and that is the most important
    /// thing about this table. The Neural Engine tiles along width, so a width
    /// that is not a multiple of 16 pays for an entire extra pass:
    ///
    ///     426x240   23.0 ms          432x240   13.6 ms
    ///     854x480   92.4 ms          864x480   48.2 ms
    ///
    /// Height alignment buys nothing - 480x272 measured *slower* than 480x270,
    /// by exactly the two extra rows of pixels. So the real streaming sizes
    /// that are not aligned (426 and 854 wide) are converted at the next
    /// multiple of 16 and the frame is stretched that last 1.4% on the way in.
    /// Nothing is lost: the page draws the result into the video box, which
    /// has the true aspect, so the stretch is undone on presentation.
    static let variants: [Variant] = [
        Variant(width: 256, height: 144, milliseconds: 4.5),
        Variant(width: 320, height: 180, milliseconds: 7.0),
        Variant(width: 432, height: 240, milliseconds: 13.6),   // covers 426x240
        Variant(width: 480, height: 270, milliseconds: 14.7),
        Variant(width: 640, height: 360, milliseconds: 26.7),
    ]

    /// One frame at 30fps, less the rest of the pipeline. Low-resolution
    /// streaming video runs at 24-30fps; 60fps material is published at 720p
    /// and above, which is outside the window Lucid works in anyway.
    ///
    /// A source with no variant under this budget is not enhanced at all.
    /// There is no second engine to fall back to: Apple's scaler measured
    /// *worse* than leaving the frame alone, so running it would be a
    /// disservice. Lucid declines instead, the same way RTX Video Super
    /// Resolution declines outside its own window.
    static let budgetMilliseconds = 28.0

    /// Whether this size is inside the window Lucid works in. Deliberately a
    /// question about the table above and not about what is on disk: if a build
    /// were missing its models this must not quietly answer "nothing is
    /// enhanceable" for every video. That case is a broken install, and `init`
    /// throws so it is reported rather than swallowed.
    static func supports(width: Int, height: Int) -> Bool {
        variant(width: width, height: height) != nil
    }

    /// The cheapest variant that covers the source. Scaling a frame *up* to the
    /// model's input costs nothing real; scaling it down would throw away the
    /// detail we are here to recover, so a variant is only ever used when it is
    /// at least as large as the source in both dimensions. The area bound stops
    /// a small odd size from reaching for a model far bigger than it needs.
    static func variant(width: Int, height: Int) -> Variant? {
        variants
            .filter { $0.width >= width && $0.height >= height && $0.milliseconds <= budgetMilliseconds }
            .filter { Double($0.width * $0.height) <= Double(max(width * height, 1)) * 1.5 }
            .min { $0.milliseconds < $1.milliseconds }
    }

    private static func model(width: Int, height: Int) -> URL? {
        guard let variant = variant(width: width, height: height) else { return nil }
        let name = "SPAN_x4_ch28_\(variant.width)x\(variant.height)"
        return Bundle.main.url(forResource: name, withExtension: "mlmodelc")
            ?? Bundle.main.url(forResource: name, withExtension: "mlpackage")
    }

    let scale = 4
    let inputWidth: Int
    let inputHeight: Int
    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let inputFormat: OSType
    private var transfer: VTPixelTransferSession?
    private var rgbPool: CVPixelBufferPool?
    private var outputPool: CVPixelBufferPool?

    init(width: Int, height: Int) throws {
        guard let url = Self.model(width: width, height: height) else { throw Failure.noModel }
        let compiled = url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at: url)
        let configuration = MLModelConfiguration()
        // The Neural Engine is the only placement that meets the budget: the
        // same graph measured roughly twice as slow on the GPU.
        configuration.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: compiled, configuration: configuration)
        guard let input = model.modelDescription.inputDescriptionsByName.first,
              let output = model.modelDescription.outputDescriptionsByName.first,
              let constraint = input.value.imageConstraint
        else { throw Failure.noModel }
        inputName = input.key
        outputName = output.key
        inputFormat = constraint.pixelFormatType
        inputWidth = constraint.pixelsWide
        inputHeight = constraint.pixelsHigh
        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transfer)
    }

    var outputWidth: Int { inputWidth * scale }
    var outputHeight: Int { inputHeight * scale }

    /// Runs the model. The pipeline works in 420v and the model wants an RGB
    /// image, so the frame is converted in and back out again.
    func upscale(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        let sourceFormat = CVPixelBufferGetPixelFormatType(source)
        let rgb: CVPixelBuffer
        if sourceFormat == inputFormat {
            rgb = source
        } else {
            rgb = try convert(source, to: inputFormat, width: inputWidth, height: inputHeight, pool: &rgbPool)
        }

        let provider = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(pixelBuffer: rgb)])
        guard let result = try? model.prediction(from: provider),
              let value = result.featureValue(for: outputName)?.imageBufferValue
        else { throw Failure.prediction }

        // Back to the pipeline's own format so every stage after this one is
        // unchanged by the choice of upscaler.
        return try convert(value, to: sourceFormat,
                           width: outputWidth, height: outputHeight, pool: &outputPool)
    }

    private func convert(_ source: CVPixelBuffer, to format: OSType,
                         width: Int, height: Int, pool: inout CVPixelBufferPool?) throws -> CVPixelBuffer {
        if pool == nil {
            let attributes: [String: Any] = [
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferPixelFormatTypeKey as String: format,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
            var created: CVPixelBufferPool?
            guard CVPixelBufferPoolCreate(kCFAllocatorDefault,
                                          [kCVPixelBufferPoolMinimumBufferCountKey as String: 4] as CFDictionary,
                                          attributes as CFDictionary, &created) == kCVReturnSuccess
            else { throw Failure.pixelBuffer }
            pool = created
        }
        var destination: CVPixelBuffer?
        guard let pool, CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &destination) == kCVReturnSuccess,
              let destination, let transfer
        else { throw Failure.pixelBuffer }
        VTPixelTransferSessionTransferImage(transfer, from: source, to: destination)
        if let attachments = CVBufferCopyAttachments(source, .shouldPropagate) {
            CVBufferSetAttachments(destination, attachments, .shouldPropagate)
        }
        TiledVideoToolboxUpscaler.ensureColorDescription(destination)
        return destination
    }
}
