//
//  LearnedUpscaler.swift
//  Lucid
//
//  Lucid's upscaler: SPAN, running on the GPU by default. This is the whole
//  reconstruction path - there is no second engine underneath it.
//
//  Scored on 120 pairs of live-action footage in neither training corpus,
//  against a plain Lanczos upscale:
//
//                     LPIPS     DISTS   BRISQUE   detail
//      ch32utc        0.5336    0.1986     52.1     0.38
//      ch32u          0.5664    0.1967     64.5     0.32
//      lanczos        0.6129    0.2243     65.7     0.24
//
//  LPIPS and DISTS are perceptual distances and BRISQUE is a no-reference
//  quality score; lower is better for all three. `detail` is fine-band energy
//  as a fraction of the reference's, where the truth is 1.00.
//
//  The `tc` model is trained with a temporal consistency term: it is penalised
//  for changing its mind where the source did not move. Two frames of a still
//  scene are the same picture under two draws of codec noise, so forbidding the
//  output to follow that difference stops the model spending capacity fitting
//  noise - which is why a term aimed at stability also raised detail and
//  lowered every perceptual score but DISTS.
//
//  It costs nothing here. The term needs a second frame during training and
//  none at inference, so this is still a single-frame model, the graph is
//  unchanged and so is every latency below.
//
//  Measured on the shipping pipeline, it shimmers 11.1% less than the model it
//  replaces on a near-still clip and 2.7% more on a fast-motion one - the trade
//  it was chosen for, since a still scene is where a viewer sees shimmer.
//
//  Against the models it replaces, at 480x270: ch32u runs in 7.92 ms where the
//  previous ch28 took 14.7 ms and ch48 took 21.8 ms.
//

import Accelerate
import CoreML
import Metal
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
    /// thing about this table. An unaligned width costs a multiple, not a
    /// margin, and it is not a Neural Engine quirk - the GPU cares more:
    ///
    ///                        ANE      GPU
    ///     432x240  aligned   6.24     4.36
    ///     426x240  native   14.61    13.41     3.1x on the GPU
    ///     864x480  aligned            14.68
    ///     854x480  native             50.82    3.5x on the GPU
    ///
    /// That was worth checking rather than assuming, because the rule was
    /// originally measured on the Neural Engine and the model now runs on the
    /// GPU by default - so the obvious guess was that two of these six rungs
    /// existed to solve a problem we no longer had. The opposite is true.
    ///
    /// Height alignment buys nothing - 480x272 measured *slower* than 480x270,
    /// by exactly the two extra rows of pixels. So the real streaming sizes
    /// that are not aligned (426 and 854 wide) are converted at the next
    /// multiple of 16 and the frame is stretched that last 1.4% on the way in.
    /// Nothing is lost: the page draws the result into the video box, which
    /// has the true aspect, so the stretch is undone on presentation.
    /// Measured end to end on the shipping path - model, colour conversions and
    /// the detail stages together - on an M4 Pro's GPU, 60 frames after 8 warm
    /// up, two repeats each. Not the model in isolation, and not arithmetic:
    /// the first version of this table added an estimate for everything that
    /// was not the model, and the estimate was the part that was wrong.
    static let variants: [Variant] = [
        Variant(width: 256, height: 144, milliseconds: 2.9),
        Variant(width: 320, height: 180, milliseconds: 3.7),
        Variant(width: 432, height: 240, milliseconds: 5.8),    // covers 426x240
        Variant(width: 480, height: 270, milliseconds: 6.8),
        Variant(width: 640, height: 360, milliseconds: 10.7),
        Variant(width: 864, height: 480, milliseconds: 17.8),   // covers 854x480
        // 1280x720 -> 2560x1440 at 21.0 ms on an M4 Pro (CPU+GPU); the 2x
        // model beats every interpolation on the frozen 720p screen
        // (Benchmarks/frontier/paired-ladder-protocol.md). Holds 30 fps, not 60.
        Variant(width: 1280, height: 720, milliseconds: 21.0),
    ]

    /// A 30 fps frame is 33.3 ms. Every rung above leaves room inside it for
    /// the pre- and post-stages and for returning the result to the page; the
    /// 720p rung is the tightest and holds cadence at 30 fps, not 60.
    ///
    /// A source with no rung under this budget is not enhanced at all. There
    /// is no second engine to fall back to: Apple's scalers measured worse
    /// than leaving the frame alone, so Lucid declines instead.
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

    static var computeUnits: MLComputeUnits {
        switch ProcessInfo.processInfo.environment["LUCID_COMPUTE_UNITS"]?.lowercased() {
        case "ane": return .cpuAndNeuralEngine
        case "gpu": return .cpuAndGPU
        case "all": return .all
        case "cpu": return .cpuOnly
        default: return .cpuAndGPU
        }
    }

    static var computeUnitsLabel: String {
        switch computeUnits {
        case .cpuAndGPU: return "cpuAndGPU"
        case .all: return "all"
        case .cpuOnly: return "cpuOnly"
        default: return "cpuAndNeuralEngine"
        }
    }

    static func permitsTensorOutput(arguments: [String], environment: [String: String]) -> Bool {
        arguments.contains("--pipeline-ms") ||
            (environment["LUCID_EPHEMERAL"] == "1" && environment["LUCID_EXPERIMENTAL_TENSOR_OUTPUT"] == "1")
    }

    /// The family the app starts on. Promoted 2026-09-06 from SPAN_x4_ch32utc_ (4x, folded)
    /// to the 2,000-step reference-target paired-critic 80% blend (direct 2x): +9.5% LPIPS /
    /// +14.7% DISTS through this pipeline on the 960-pair holdout, every source up.
    /// 2026-09-07: promoted again to lucidbig2k_, the same recipe trained on the
    /// 756-sequence full-frame stream bank (raw checkpoint, no blend): native holdout
    /// +11.80% LPIPS / +16.04% DISTS vs SPAN, every source up, same cost.
    static let shippingStem = "lucidbig2k_"
    /// The only family with validated tensor-output alternatives bundled.
    static let tensorFamilyStem = "SPAN_x4_ch32utc_"
    /// Set from the lab page to swap the reconstruction model without a
    /// relaunch. nil means the launch-time choice (LUCID_MODEL_STEM or shipping).
    /// `LUCID_MODEL_STEM` lets the measurement harness run a candidate ladder
    /// through the real pipeline; the product has exactly one model family.
    static var currentStem: String {
        ProcessInfo.processInfo.environment["LUCID_MODEL_STEM"] ?? shippingStem
    }

    private static func model(width: Int, height: Int) -> URL? {
        // --pipeline-ms can load an exact-size package that is not in the
        // variants table (or is over budget). isEnhanceable still reads the
        // table; this path is measurement only.
        if CommandLine.arguments.contains("--pipeline-ms") {
            if let path = ProcessInfo.processInfo.environment["LUCID_PIPELINE_MODEL"], !path.isEmpty {
                let url = URL(fileURLWithPath: path)
                guard ["mlpackage", "mlmodelc"].contains(url.pathExtension),
                      FileManager.default.fileExists(atPath: url.path) else { return nil }
                return url
            }
            let stem = currentStem
            let exact = "\(stem)\(width)x\(height)"
            if let url = Bundle.main.url(forResource: exact, withExtension: "mlmodelc")
                ?? Bundle.main.url(forResource: exact, withExtension: "mlpackage") {
                return url
            }
        }
        guard let variant = variant(width: width, height: height) else { return nil }
        // LUCID_MODEL_STEM swaps the model without changing what ships, so a
        // candidate can be measured through the real pipeline by the same
        // harness that measures the shipping one. Comparing two models by
        // rebuilding the app between them compares two builds.
        let stem = currentStem
        let name = "\(stem)\(variant.width)x\(variant.height)"
        return Bundle.main.url(forResource: name, withExtension: "mlmodelc")
            ?? Bundle.main.url(forResource: name, withExtension: "mlpackage")
    }

    let scale: Int
    /// The quantized presentation graph changes output geometry without
    /// retraining reconstruction. Keep nominal sharpening gain for that graph.
    let detailReferenceRadius: Int
    let inputWidth: Int
    let inputHeight: Int
    private var model: MLModel
    private let inputName: String
    private let outputName: String
    private var tensorPacker: CoreMLTensorImagePacker?
    private var fallbackModel: MLModel?
    typealias Prediction = (MLModel, MLFeatureProvider) throws -> MLFeatureProvider
    private let predict: Prediction
    private let inputFormat: OSType
    /// Experiment: cycle the model input through the four flip orientations,
    /// one per frame, and flip the output back. This is orientation cycling,
    /// not an ensemble: the pipeline's temporal accumulator runs BEFORE this
    /// model. The native experiment showed small, mixed spatial changes.
    nonisolated(unsafe) static var flipCycle = false
    private var frameCounter = 0
    private var flipInPool: CVPixelBufferPool?
    private var flipOutPool: CVPixelBufferPool?
    private var transfer: VTPixelTransferSession?
    private var rgbPool: CVPixelBufferPool?
    private var outputPool: CVPixelBufferPool?
    private let ownedCompiledURL: URL?

    static func reconstructionScale(inputWidth: Int, inputHeight: Int, outputWidth: Int, outputHeight: Int) -> Int? {
        guard inputWidth > 0, inputHeight > 0, outputWidth > 0, outputHeight > 0,
              outputWidth % inputWidth == 0, outputHeight % inputHeight == 0 else { return nil }
        let scale = outputWidth / inputWidth
        guard [2, 4].contains(scale), outputHeight / inputHeight == scale else { return nil }
        return scale
    }

    static func detailReferenceRadius(scale: Int, metadata: [String: String]) -> Int {
        guard scale == 2,
              metadata["lucid.transformation"] == "quantized 4x-to-2x bicubic presentation",
              metadata["lucid.checkpoint_sha256"] == "fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65"
        else { return 4 }
        return 2
    }

    init(width: Int, height: Int,
         prediction: @escaping Prediction = { try $0.prediction(from: $1) }) throws {
        predict = prediction
        guard let url = Self.model(width: width, height: height) else { throw Failure.noModel }
        let compiled = url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at: url)
        let ownsCompilation = url.pathExtension != "mlmodelc"
        var keepCompilation = false
        defer {
            if ownsCompilation && !keepCompilation { try? FileManager.default.removeItem(at: compiled) }
        }
        ownedCompiledURL = ownsCompilation ? compiled : nil
        let configuration = MLModelConfiguration()
        // Placement is not settled, and the comment that used to sit here was
        // wrong. It said the Neural Engine was the only placement that met the
        // budget and that the GPU measured twice as slow. That was true of
        // ch28. It did not survive the architecture change: with the trunk
        // running at quarter area and the head doing x8, measured on the
        // shipping packages, the GPU is faster at every tier -
        //
        //     tier        ANE      GPU
        //     256x144    2.30     1.66
        //     432x240    6.25     4.29
        //     640x360   11.80     8.38
        //     864x480   22.68    14.36
        //
        // - and Core ML's automatic placement lands between the two. Nobody
        // re-measured after the model changed, because a comment already
        // answered the question.
        //
        // The obvious objection to moving off the Neural Engine is that the
        // detail stages are Metal compute on that same GPU, so the model would
        // contend with our own pipeline instead of running on separate silicon.
        // Measured end to end at 854x480 with the detail stages on and the
        // shipping tuning, 60 frames, two repeats:
        //
        //                total          model          detail
        //     ANE     32.21 / 29.39   28.33 / 26.11   3.25 / 2.70
        //     GPU     18.77 / 19.68   16.62 / 17.18   1.78 / 2.04
        //
        // The contention did not materialise. The GPU is about 10 ms faster on
        // the whole frame, and the detail stage is *faster* alongside it than
        // it was alongside the Neural Engine - so the two were competing for
        // something even when they looked independent. On the Neural Engine
        // 480p sat at a 33.3 ms frame with p95 over budget; on the GPU it has
        // room.
        //
        // Same protocol at 640x360 (the tier that had only an isolated
        // modelbench number). Placement is the same, not per-tier:
        //
        //                total          model          detail
        //     ANE     16.98 / 17.24   13.04 / 13.18   3.29 / 3.37
        //     GPU     10.70 / 10.72    9.15 /  9.18   1.22 / 1.19
        //
        // GPU still wins, by less in absolute milliseconds (~6 ms against
        // ~10 ms at 480p) and about the same fraction of the frame. Detail
        // is faster alongside the GPU again.
        //
        // LUCID_COMPUTE_UNITS=ane|gpu|all|cpu overrides this. Worth keeping:
        // these numbers are an M4 Pro, the balance between the two engines
        // differs across the range, and this should be re-measured rather than
        // assumed on any machine whose GPU is smaller relative to its Neural
        // Engine.
        configuration.computeUnits = Self.computeUnits
        print("   🧠 SPAN compute units: \(Self.computeUnitsLabel)")
        let referenceModel = try MLModel(contentsOf: compiled, configuration: configuration)
        var optimized: ValidatedTensorOutput?
        if configuration.computeUnits == .cpuAndGPU,
           Self.currentStem == Self.tensorFamilyStem,
           ValidatedTensorOutput.eligible(deviceName: MTLCreateSystemDefaultDevice()?.name ?? "",
                version: ProcessInfo.processInfo.operatingSystemVersion,
                arguments: CommandLine.arguments, environment: ProcessInfo.processInfo.environment),
           let input = referenceModel.modelDescription.inputDescriptionsByName.first?.value.imageConstraint,
           let tensorURL = Bundle.main.url(forResource: "\(Self.tensorFamilyStem)tensor_\(input.pixelsWide)x\(input.pixelsHigh)", withExtension: "mlmodelc") {
            do {
                optimized = try ValidatedTensorOutput.load(reference: referenceModel, url: tensorURL, configuration: configuration)
                print(optimized == nil ? "   Tensor compatibility check declined; using image output" :
                    "   Tensor compatibility check passed; image fallback retained")
            } catch {
                print("   Tensor compatibility check failed; using image output: \(error)")
            }
            fflush(stdout)
        }
        model = optimized?.model ?? referenceModel
        fallbackModel = optimized == nil ? nil : referenceModel
        guard let input = model.modelDescription.inputDescriptionsByName.first,
              let output = model.modelDescription.outputDescriptionsByName.first,
              let constraint = input.value.imageConstraint else { throw Failure.noModel }
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        let outputWidth: Int, outputHeight: Int
        if let image = output.value.imageConstraint {
            outputWidth = image.pixelsWide; outputHeight = image.pixelsHigh
            tensorPacker = nil
        } else if (optimized != nil || Self.permitsTensorOutput(arguments: CommandLine.arguments, environment: ProcessInfo.processInfo.environment)),
                  configuration.computeUnits == .cpuAndGPU,
                  let tensor = output.value.multiArrayConstraint,
                  tensor.dataType == .float32,
                  tensor.shape.map(\.intValue) == [1, 3, constraint.pixelsHigh * 4, constraint.pixelsWide * 4],
                  metadata["lucid.output_range"] == "0..255",
                  metadata["lucid.output_scale"] == "4",
                  metadata["lucid.checkpoint_sha256"] == "fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65" {
            // Bundled alternatives require runtime admission; overrides remain explicit experiments.
            outputWidth = constraint.pixelsWide * 4; outputHeight = constraint.pixelsHigh * 4
            tensorPacker = try optimized?.packer ?? CoreMLTensorImagePacker(width: outputWidth, height: outputHeight)
            print("   Tensor RGB8 output: shared Metal storage, binary16 precision then nearest-even")
        } else { throw Failure.noModel }
        guard let reconstructionScale = Self.reconstructionScale(
            inputWidth: constraint.pixelsWide, inputHeight: constraint.pixelsHigh,
            outputWidth: outputWidth, outputHeight: outputHeight) else { throw Failure.noModel }
        scale = reconstructionScale
        detailReferenceRadius = Self.detailReferenceRadius(scale: reconstructionScale, metadata: metadata)
        inputName = input.key
        outputName = output.key
        inputFormat = constraint.pixelFormatType
        inputWidth = constraint.pixelsWide
        inputHeight = constraint.pixelsHigh
        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transfer)
        keepCompilation = true
    }

    deinit {
        if let ownedCompiledURL { try? FileManager.default.removeItem(at: ownedCompiledURL) }
    }

    var outputWidth: Int { inputWidth * scale }
    var outputHeight: Int { inputHeight * scale }

    /// Runs the model. The pipeline works in 420v and the model wants an RGB
    /// image, so the frame is converted in and back out again.
    func upscale(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        let sourceFormat = CVPixelBufferGetPixelFormatType(source)
        let rgb: CVPixelBuffer
        if sourceFormat == inputFormat, CVPixelBufferGetWidth(source) == inputWidth, CVPixelBufferGetHeight(source) == inputHeight {
            rgb = source
        } else {
            rgb = try convert(source, to: inputFormat, width: inputWidth, height: inputHeight, pool: &rgbPool)
        }

        let orientation = Self.flipCycle ? frameCounter % 4 : 0
        frameCounter &+= 1
        let modelInput = orientation == 0 ? rgb : try flipped(rgb, orientation: orientation, pool: &flipInPool)
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(pixelBuffer: modelInput)])
        var value: CVPixelBuffer
        do {
            let result = try predict(model, provider)
            if let tensorPacker {
                guard let array = result.featureValue(for: outputName)?.multiArrayValue else { throw Failure.prediction }
                value = try tensorPacker.pack(array)
            } else {
                guard let image = result.featureValue(for: outputName)?.imageBufferValue else { throw Failure.prediction }
                value = image
            }
        } catch {
            guard let fallbackModel else { throw error }
            model = fallbackModel
            tensorPacker = nil
            self.fallbackModel = nil
            print("   Tensor prediction failed; restored image output: \(error)")
            fflush(stdout)
            guard let image = try predict(model, provider).featureValue(for: outputName)?.imageBufferValue else { throw Failure.prediction }
            value = image
        }
        guard CVPixelBufferGetWidth(value) == outputWidth,
              CVPixelBufferGetHeight(value) == outputHeight else { throw Failure.prediction }
        if orientation != 0 { value = try flipped(value, orientation: orientation, pool: &flipOutPool) }

        // The image model predicts RGB samples in its input encoding. Core ML
        // returns an untagged image; defaulting it to 709 would lose sRGB here.
        var predictedColor = VideoColorInfo.read(from: source)
        predictedColor.matrix = "rgb"; predictedColor.fullRange = true
        guard predictedColor.apply(to: value) else { throw Failure.prediction }

        // Back to the pipeline's own format so every stage after this one is
        // unchanged by the choice of upscaler.
        return try convert(value, to: sourceFormat,
                           width: outputWidth, height: outputHeight, pool: &outputPool)
    }

    /// Mirrors a 4-channel 8-bit buffer: 1 = horizontal, 2 = vertical, 3 = both.
    /// The operation is its own inverse, so the same call un-flips the output.
    private func flipped(_ source: CVPixelBuffer, orientation: Int, pool: inout CVPixelBufferPool?) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(source), height = CVPixelBufferGetHeight(source)
        let format = CVPixelBufferGetPixelFormatType(source)
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
              let destination else { throw Failure.pixelBuffer }
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(destination, [])
        }
        guard let sourceBase = CVPixelBufferGetBaseAddress(source), let destinationBase = CVPixelBufferGetBaseAddress(destination)
        else { throw Failure.pixelBuffer }
        var src = vImage_Buffer(data: sourceBase, height: vImagePixelCount(height), width: vImagePixelCount(width),
                                rowBytes: CVPixelBufferGetBytesPerRow(source))
        var dst = vImage_Buffer(data: destinationBase, height: vImagePixelCount(height), width: vImagePixelCount(width),
                                rowBytes: CVPixelBufferGetBytesPerRow(destination))
        if orientation & 1 != 0 {
            guard vImageHorizontalReflect_ARGB8888(&src, &dst, 0) == kvImageNoError else { throw Failure.pixelBuffer }
        }
        if orientation & 2 != 0 {
            if orientation & 1 != 0 {
                guard vImageVerticalReflect_ARGB8888(&dst, &dst, 0) == kvImageNoError else { throw Failure.pixelBuffer }
            } else {
                guard vImageVerticalReflect_ARGB8888(&src, &dst, 0) == kvImageNoError else { throw Failure.pixelBuffer }
            }
        }
        if let attachments = CVBufferCopyAttachments(source, .shouldPropagate) {
            CVBufferSetAttachments(destination, attachments, .shouldPropagate)
        }
        return destination
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
        // Tag the source first. VT samples 4:2:0 chroma using the source's
        // location; tagging only the destination (after the transfer) cannot
        // change what the model sees.
        TiledVideoToolboxUpscaler.ensureColorDescription(source)
        CVBufferRemoveAllAttachments(destination)
        guard VTPixelTransferSessionTransferImage(transfer, from: source, to: destination) == noErr
        else { throw Failure.pixelBuffer }
        if let attachments = CVBufferCopyAttachments(source, .shouldPropagate) {
            CVBufferSetAttachments(destination, attachments, .shouldPropagate)
        }
        TiledVideoToolboxUpscaler.ensureColorDescription(destination)
        return destination
    }
}
