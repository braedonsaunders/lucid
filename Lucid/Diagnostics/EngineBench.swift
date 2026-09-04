//
//  EngineBench.swift
//  Lucid
//
//  Offline engine comparison, run with:
//    Lucid.app/Contents/MacOS/Lucid --bench <input.mp4> <reference.mp4> <outDir> [firstFrame] [count]
//
//  Decodes a low-resolution clip, runs it through both reconstruction engines,
//  and writes PNGs beside the matching reference frames so quality can be
//  scored against ground truth rather than eyeballed.
//

import AVFoundation
import AppKit
import CoreImage
import CoreMedia
import CoreVideo
import Foundation
@preconcurrency import VideoToolbox

@available(macOS 26.0, *)
enum EngineBench {
    static func describe(_ buffer: CVPixelBuffer, _ label: String) {
        let attachments = CVBufferCopyAttachments(buffer, .shouldPropagate) as? [String: Any] ?? [:]
        func value(_ key: CFString) -> String {
            (attachments[key as String] as? String) ?? (attachments[key as String].map { "\($0)" } ?? "nil")
        }
        let format = CVPixelBufferGetPixelFormatType(buffer)
        let fourCC = String(bytes: [UInt8(format >> 24 & 255), UInt8(format >> 16 & 255), UInt8(format >> 8 & 255), UInt8(format & 255)], encoding: .ascii) ?? "?"
        print("  \(label): \(CVPixelBufferGetWidth(buffer))x\(CVPixelBufferGetHeight(buffer)) \(fourCC) matrix=\(value(kCVImageBufferYCbCrMatrixKey)) primaries=\(value(kCVImageBufferColorPrimariesKey)) transfer=\(value(kCVImageBufferTransferFunctionKey)) chroma=\(value(kCVImageBufferChromaLocationTopFieldKey))")
        fflush(stdout)
    }

    static func run() async {
        setvbuf(stdout, nil, _IOLBF, 0)
        let args = CommandLine.arguments
        guard let benchIndex = args.firstIndex(of: "--bench"), args.count > benchIndex + 3 else {
            print("usage: Lucid --bench <input.mp4> <reference.mp4> <outDir> [firstFrame] [count]")
            exit(2)
        }
        let inputPath = args[benchIndex + 1]
        let referencePath = args[benchIndex + 2]
        let outDir = args[benchIndex + 3]
        let first = args.count > benchIndex + 4 ? Int(args[benchIndex + 4]) ?? 0 : 0
        let count = args.count > benchIndex + 5 ? Int(args[benchIndex + 5]) ?? 6 : 6
        try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)

        do {
            try await compare(inputPath: inputPath, referencePath: referencePath, outDir: outDir, first: first, count: count)
        } catch {
            print("bench failed: \(error)")
            exit(1)
        }
        exit(0)
    }

    private static func makeReader(_ path: String) async throws -> (AVAssetReader, AVAssetReaderTrackOutput) {
        let asset = AVURLAsset(url: URL(fileURLWithPath: path))
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "no video track in \(path)"])
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ])
        reader.add(output)
        reader.startReading()
        return (reader, output)
    }

    private static func makePool(_ attributes: [String: Any]) -> CVPixelBufferPool {
        var merged = attributes
        merged[kCVPixelBufferMetalCompatibilityKey as String] = true
        if merged[kCVPixelBufferIOSurfacePropertiesKey as String] == nil {
            merged[kCVPixelBufferIOSurfacePropertiesKey as String] = [:] as [String: Any]
        }
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(kCFAllocatorDefault,
                                [kCVPixelBufferPoolMinimumBufferCountKey as String: 4] as CFDictionary,
                                merged as CFDictionary, &pool)
        return pool!
    }

    private static func makeBuffer(_ pool: CVPixelBufferPool) -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer)
        return buffer!
    }

    private static func writePNG(_ buffer: CVPixelBuffer, to path: String, context: CIContext) {
        try? context.writePNGRepresentation(
            of: CIImage(cvPixelBuffer: buffer), to: URL(fileURLWithPath: path),
            format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!
        )
    }

    private static func compare(inputPath: String, referencePath: String, outDir: String, first: Int, count: Int) async throws {
        let context = CIContext()
        let (inputReader, input) = try await makeReader(inputPath)
        let (referenceReader, reference) = try await makeReader(referencePath)
        defer { withExtendedLifetime((inputReader, referenceReader)) {} }
        let compositor = try MetalTileCompositor()
        var transferSession: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transferSession)
        guard let transfer = transferSession else { throw NSError(domain: "bench", code: 3) }

        // The stages that follow behave differently depending on what they sit
        // on - deblocking earns more over a blockier base, sharpening earns
        // less over one that already invented texture. So an ablation is only
        // meaningful against the upscaler that actually ships, and this points
        // the bench at it.
        var learned: LearnedUpscaler?
        let useLearned = ProcessInfo.processInfo.environment["LUCID_LEARNED"] == "1"
        var lowLatency: TiledVideoToolboxUpscaler?
        var processor: VTFrameProcessor?
        var sourcePool: CVPixelBufferPool?
        var destinationPool: CVPixelBufferPool?
        var previousSource: CVPixelBuffer?
        var previousOutput: CVPixelBuffer?
        let t = EnhancementSession.Tuning.load()
        // Chroma siting lives on a static. Setting it is not enough: the
        // incoming AVAssetReader buffer is usually already tagged, and VT
        // samples chroma from the source, so ensureColorDescription must
        // overwrite siting on that buffer before the 420→RGB convert.
        TiledVideoToolboxUpscaler.chromaSitingLeft = t.stageSiting > 0.5
        let detail = try DetailEnhancer(device: compositor.device, settings: DetailSettings(
            sharpness: t.sharpness, fine: t.fine,
            micro: t.micro, lobeScale: t.lobeScale, mid: t.mid,
            flatThreshold: 0.004, edgeThreshold: 0.030, deblock: t.deblock,
            sourceDeblock: t.sourceDeblock, sourceDeblockRadius: 1.6, presharpen: t.presharpen, adaptive: t.adaptive,
            temporal: t.temporal, motionLow: 0.02, motionHigh: 0.08,
            radius: ProcessInfo.processInfo.environment["LUCID_RADIUS"].flatMap(Int.init) ?? 4,
            blackPoint: t.blackPoint, whitePoint: t.whitePoint,
            contrast: t.contrast, saturation: t.saturation,
            stageLoopFilter: t.stageLoopFilter > 0.5,
            stageCdef: t.stageCdef > 0.5,
            stageDeband: t.stageDeband > 0.5,
            stageTaa: t.stageTaa > 0.5,
            stageOklab: t.stageOklab > 0.5,
            loopFilterQuant: t.loopFilterQuant,
            cdefPrimary: t.cdefPrimary,
            cdefSecondary: t.cdefSecondary,
            debandThreshold: t.debandThreshold,
            grain: t.grain,
            grainPhase: t.grainPhase,
            taaGamma: t.taaGamma,
            taaFeedback: t.taaFeedback,
            skinProtect: t.skinProtect
        ))
        var detailTimes: [Double] = []
        var lowLatencyTimes: [Double] = []
        var temporalTimes: [Double] = []
        var index = 0

        while index < first + count, let sample = input.copyNextSampleBuffer(), let frame = sample.imageBuffer {
            // Tag the SOURCE, not just whatever comes out of it. The reader
            // propagates the container's colour attachments, so the incoming
            // frame already claims Left siting and the pixel transfer resamples
            // chroma at that claim before anything downstream gets a say.
            // Setting the static alone changed nothing: two bench runs with
            // opposite settings produced byte-identical output, which read as
            // "this stage does nothing" when it meant "this stage was never
            // reached".
            TiledVideoToolboxUpscaler.ensureColorDescription(frame)
            let referenceFrame = reference.copyNextSampleBuffer()?.imageBuffer
            let width = CVPixelBufferGetWidth(frame), height = CVPixelBufferGetHeight(frame)
            if index == first { describe(frame, "captured input (as decoded)") }
            let frame = TiledVideoToolboxUpscaler.prepareSource(frame)
            if index == first {
                describe(frame, "captured input (after siting prepare)")
                print("bench chroma flag=\(TiledVideoToolboxUpscaler.chromaSitingLeft ? "left" : "center") resampled=\(TiledVideoToolboxUpscaler.chromaSitingLeft ? "none" : "center")")
            }

            if lowLatency == nil {
                let factor = Int(t.scalerFactor.rounded())
                let stage = try TiledVideoToolboxUpscaler(width: width, height: height, compositor: compositor, preferredScale: factor)
                if factor < 4 {
                    stage.nextStage = try TiledVideoToolboxUpscaler(
                        width: width * 2, height: height * 2, compositor: compositor, preferredScale: factor)
                }
                lowLatency = stage
                guard let configuration = VTSuperResolutionScalerConfiguration(
                    frameWidth: width, frameHeight: height, scaleFactor: 4, inputType: .video,
                    usePrecomputedFlow: false, qualityPrioritization: .normal, revision: .revision1)
                else { throw NSError(domain: "bench", code: 2, userInfo: [NSLocalizedDescriptionKey: "temporal config unavailable for \(width)x\(height)"]) }
                if configuration.configurationModelStatus != .ready {
                    print("downloading temporal model…")
                    await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
                        configuration.downloadConfigurationModel { _ in c.resume() }
                    }
                }
                let vt = VTFrameProcessor()
                try vt.startSession(configuration: configuration)
                processor = vt
                sourcePool = makePool(configuration.sourcePixelBufferAttributes)
                destinationPool = makePool(configuration.destinationPixelBufferAttributes)
                let sourceFormat = configuration.sourcePixelBufferAttributes[kCVPixelBufferPixelFormatTypeKey as String] as? OSType ?? 0
                let fourCC = String(bytes: [UInt8(sourceFormat >> 24 & 255), UInt8(sourceFormat >> 16 & 255), UInt8(sourceFormat >> 8 & 255), UInt8(sourceFormat & 255)], encoding: .ascii) ?? "?"
                print("input \(width)x\(height) → 4× = \(width * 4)x\(height * 4); temporal wants \(fourCC)")
                fflush(stdout)
            }

            let pts = CMTime(value: CMTimeValue(index), timescale: 30)

            let cleaned = try detail.preprocess(frame)
            let startLowLatency = ContinuousClock.now
            let lowLatencyOutput: CVPixelBuffer
            if useLearned {
                if learned == nil {
                    learned = try LearnedUpscaler(width: width, height: height)
                    print("bench upscaler: SPAN 4× on the Neural Engine, \(width)x\(height)")
                }
                lowLatencyOutput = try learned!.upscale(cleaned)
            } else {
                lowLatencyOutput = try await lowLatency!.upscale(cleaned, pts: pts)
            }
            lowLatencyTimes.append((ContinuousClock.now - startLowLatency).milliseconds)

            let startDetail = ContinuousClock.now
            let detailOutput = try detail.process(lowLatencyOutput)
            detailTimes.append((ContinuousClock.now - startDetail).milliseconds)

            let source = makeBuffer(sourcePool!)
            // The temporal model may want a different pixel format than the
            // decoder produced, so convert rather than blit.
            VTPixelTransferSessionTransferImage(transfer, from: frame, to: source)
            let destination = makeBuffer(destinationPool!)
            guard let sourceFrame = VTFrameProcessorFrame(buffer: source, presentationTimeStamp: pts),
                  let destinationFrame = VTFrameProcessorFrame(buffer: destination, presentationTimeStamp: pts)
            else { index += 1; continue }
            let previousFrame = previousSource.flatMap {
                VTFrameProcessorFrame(buffer: $0, presentationTimeStamp: CMTime(value: CMTimeValue(index - 1), timescale: 30))
            }
            let previousOutputFrame = previousOutput.flatMap {
                VTFrameProcessorFrame(buffer: $0, presentationTimeStamp: CMTime(value: CMTimeValue(index - 1), timescale: 30))
            }
            guard let parameters = VTSuperResolutionScalerParameters(
                sourceFrame: sourceFrame, previousFrame: previousFrame, previousOutputFrame: previousOutputFrame,
                opticalFlow: nil, submissionMode: previousFrame == nil ? .random : .sequential,
                destinationFrame: destinationFrame)
            else { index += 1; continue }
            let startTemporal = ContinuousClock.now
            try await processor!.process(parameters: parameters)
            temporalTimes.append((ContinuousClock.now - startTemporal).milliseconds)
            previousSource = source
            previousOutput = destination

            if index == first {
                describe(frame, "captured input (after siting)")
                describe(lowLatencyOutput, "low-latency 4×")
                describe(destination, "temporal 4×")
                if let referenceFrame { describe(referenceFrame, "reference") }
            }
            if index >= first {
                print("frame \(index)"); fflush(stdout)
                let name = String(format: "%03d", index)
                writePNG(frame, to: "\(outDir)/\(name)-input.png", context: context)
                // The source after our own deblock and temporal pass. Every
                // learned upscaler on offer was trained on clean bicubic
                // downsampling, so what it is handed matters as much as which
                // model it is.
                writePNG(cleaned, to: "\(outDir)/\(name)-cleaned.png", context: context)
                writePNG(lowLatencyOutput, to: "\(outDir)/\(name)-lowlatency.png", context: context)
                writePNG(detailOutput, to: "\(outDir)/\(name)-detail.png", context: context)
                writePNG(destination, to: "\(outDir)/\(name)-temporal.png", context: context)
                if let referenceFrame { writePNG(referenceFrame, to: "\(outDir)/\(name)-reference.png", context: context) }
            }
            index += 1
        }

        func mean(_ values: [Double]) -> String {
            let warm = values.dropFirst(5)
            return warm.isEmpty ? "n/a" : String(format: "%.1f ms", warm.reduce(0, +) / Double(warm.count))
        }
        let report = "frames \(index)  ·  low-latency 4×: \(mean(lowLatencyTimes))  ·  detail stage: \(mean(detailTimes))  ·  temporal 4×: \(mean(temporalTimes))"
        print(report)
        try? report.write(toFile: "\(outDir)/report.txt", atomically: true, encoding: .utf8)
        await lowLatency?.end()
        processor?.endSession()
    }
}
