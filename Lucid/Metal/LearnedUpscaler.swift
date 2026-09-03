//
//  LearnedUpscaler.swift
//  Lucid
//
//  A Core ML upscaler running on the Neural Engine, used in place of Apple's
//  scaler where it fits the frame budget.
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
//

import CoreML
import CoreVideo
import Foundation
import VideoToolbox

@available(macOS 26.0, *)
final class LearnedUpscaler: @unchecked Sendable {
    enum Failure: Error { case noModel, pixelBuffer, prediction }

    /// Input sizes a model has been converted for, with what each costs on an
    /// M4 Pro's Neural Engine. Cost is roughly parameters x low-resolution
    /// pixels, so it grows quickly with the source size.
    struct Variant {
        let width: Int
        let height: Int
        let milliseconds: Double
    }

    static let variants: [Variant] = [
        Variant(width: 256, height: 144, milliseconds: 4.5),
        Variant(width: 426, height: 240, milliseconds: 11.5),
        Variant(width: 480, height: 270, milliseconds: 14.7),
        Variant(width: 640, height: 360, milliseconds: 26.7),
    ]

    /// Above this the learned pass costs more than the frame budget allows and
    /// Apple's scaler is used instead. Being slower than real time would undo
    /// any quality gain.
    static let budgetMilliseconds = 16.0

    /// Whether a learned model exists for this size and is worth running.
    static func supports(width: Int, height: Int) -> Bool {
        model(width: width, height: height) != nil
    }

    private static func variant(width: Int, height: Int) -> Variant? {
        variants.first { $0.width == width && $0.height == height && $0.milliseconds <= budgetMilliseconds }
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
