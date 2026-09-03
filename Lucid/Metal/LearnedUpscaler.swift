//
//  LearnedUpscaler.swift
//  Lucid
//
//  Lucid's upscaler: SPAN, running on the Neural Engine. This is the whole
//  reconstruction path - there is no second engine underneath it.
//
//  Scored on 120 pairs of live-action footage in neither training corpus,
//  fine-band correlation with the truth - the metric that separates recovered
//  detail from invented detail:
//
//      ch32u, trained on our codec corpus   0.272
//      the same model on animation only     0.252
//      lanczos anchor                       0.229
//
//  And on the 1080p bench references, against the models it replaces:
//
//      SPAN ch48    0.2278 fine, 21.8 ms      ch32u   0.2242 fine, 7.9 ms
//      SPAN ch28    0.2214 fine, 14.7 ms      (at 480x270)
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

    /// Input sizes a model has been converted for, with what each costs per
    /// frame on an M4 Pro. Cost is roughly parameters x low-resolution pixels,
    /// so it grows quickly with the source size.
    ///
    /// `milliseconds` is the whole call, not the Core ML prediction: the
    /// pipeline works in 420v and the model wants RGB, so a frame is converted
    /// in and converted back, and the way back is at 4x. Measured separately
    /// with Tools/PipelineBench.swift, that pair costs 0.5 ms at 256x144 and
    /// 4.2 ms at 864x480 - small at the bottom of the ladder, and decisive at
    /// the top, which is why it is counted here rather than left implicit.
    struct Variant {
        let width: Int
        let height: Int
        /// Colour conversion in, model, colour conversion back out.
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
        Variant(width: 256, height: 144, milliseconds: 3.5),    // 2.38 model + 0.5 convert + detail
        Variant(width: 320, height: 180, milliseconds: 5.2),    // 3.54 + 0.7
        Variant(width: 432, height: 240, milliseconds: 9.1),    // 6.40 + 1.1, covers 426x240
        Variant(width: 480, height: 270, milliseconds: 11.6),   // 7.92 + 1.7
        Variant(width: 640, height: 360, milliseconds: 18.2),   // 12.53 + 2.8
        Variant(width: 864, height: 480, milliseconds: 32.7),   // 23.50 + 4.2, covers 854x480
    ]

    /// The target window is Microsoft Edge's: enabled below 720p. Edge arrived
    /// at that independently, from inside a browser compositor, which is worth
    /// something - it is the range where a source is short of what the display
    /// shows, and above it there is progressively less to recover.
    ///
    /// That needs the 854x480 tier, which ch28 could not carry (48.2 ms).
    /// The trained ch32u does it in 23.5 ms, but 480p is 3456x1920 out - 1.8x
    /// the pixels of 360p - so the colour conversions cost 4.2 ms and the
    /// detail stage scales with them. The total lands at 32.7 ms against a
    /// 33.3 ms frame: it fits, with no room to spare. If the ablation finds
    /// detail stages that are not earning their milliseconds, cutting them is
    /// what turns that from marginal into comfortable.
    ///
    /// Where the next speed comes from, measured 2026-09-03 so it is not
    /// re-derived: this trunk is activation-bandwidth bound, not MAC bound.
    /// Latency scales with channels rather than channels squared (ch16 0.63x,
    /// ch20 0.75x, ch24 0.90x against ch28), and int8 weight quantisation moved
    /// 30.6ms to 29.3ms - nothing, because weights are not the traffic.
    ///
    /// So the lever is pixels, not channels. PixelUnshuffle(2) at the input
    /// puts the 18 trunk convs at quarter area with the head doing x8 instead
    /// of x4, and at 640x360 that is 27.9ms -> 12.6ms at *more* capacity than
    /// ships today (1.06M parameters against 1.03M). Its ladder also brings
    /// 864x480 to 24.7ms, back inside this budget.
    ///
    /// It had no pretrained weights, so it was trained from scratch on 1800
    /// pairs of real codec degradation - x264 and VP9 at 90 kbps to 1.5 Mbps,
    /// GOP 30 to 250, 64% live action and 35% animation across 21 sources.
    /// That is what ships now.
    ///
    /// A 30fps frame is 33.3 ms. This leaves 3 ms of it for the detail pass and
    /// for getting the result back to the page. Low-resolution streaming video
    /// runs at 24-30fps; 60fps material is published at 720p and above, which
    /// is outside the window Lucid works in anyway.
    ///
    /// 864x480 sits at 32.7 ms, so it is the one size with almost no headroom.
    /// If frames start dropping in real use, it is the first thing to remove.
    ///
    /// A source with no variant under this budget is not enhanced at all.
    /// There is no second engine to fall back to: Apple's scaler measured
    /// *worse* than leaving the frame alone, so running it would be a
    /// disservice. Lucid declines instead, the same way RTX Video Super
    /// Resolution declines outside its own window.
    static let budgetMilliseconds = 33.0

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
        let name = "SPAN_x4_ch32u_\(variant.width)x\(variant.height)"
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
