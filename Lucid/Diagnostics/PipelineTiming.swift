//
//  PipelineTiming.swift
//  Lucid
//
//  End-to-end frame time for the shipping path only: preprocess → SPAN →
//  detail. Used to A/B Neural Engine vs GPU without the browser bridge.
//  Isolated Core ML benches omit the Metal stages that share the GPU.
//
//    Lucid --pipeline-ms <input.mp4> [count]
//    LUCID_COMPUTE_UNITS=gpu Lucid --pipeline-ms <input.mp4> 60
//
//  An exact-size SPAN_x4_ch32u_<w>x<h> package in the app bundle is loaded
//  even when that size is missing from the variants table or over budget.
//

import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

@available(macOS 26.0, *)
enum PipelineTiming {
    static func run() async {
        setvbuf(stdout, nil, _IOLBF, 0)
        let args = CommandLine.arguments
        guard let index = args.firstIndex(of: "--pipeline-ms"), args.count > index + 1 else {
            print("usage: Lucid --pipeline-ms <input.mp4> [count]")
            exit(2)
        }
        let path = args[index + 1]
        let count = args.count > index + 2 ? max(Int(args[index + 2]) ?? 60, 1) : 60
        do {
            try await measure(path: path, count: count)
        } catch {
            print("pipeline-ms failed: \(error)")
            exit(1)
        }
        exit(0)
    }

    private static func measure(path: String, count: Int) async throws {
        let asset = AVURLAsset(url: URL(fileURLWithPath: path))
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw NSError(domain: "pipeline-ms", code: 1, userInfo: [NSLocalizedDescriptionKey: "no video track"])
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ])
        reader.add(output)
        reader.startReading()

        let compositor = try MetalTileCompositor()
        let t = EnhancementSession.Tuning.load()
        TiledVideoToolboxUpscaler.chromaSitingLeft = t.stageSiting > 0.5
        let detail = try DetailEnhancer(device: compositor.device, settings: DetailSettings(
            sharpness: t.sharpness, fine: t.fine,
            micro: t.micro, lobeScale: t.lobeScale, mid: t.mid,
            flatThreshold: 0.004, edgeThreshold: 0.030, deblock: t.deblock,
            sourceDeblock: t.sourceDeblock, sourceDeblockRadius: 1.6, presharpen: t.presharpen, adaptive: t.adaptive,
            temporal: t.temporal, motionLow: 0.02, motionHigh: 0.08,
            radius: 4,
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
            taaGamma: t.taaGamma,
            taaFeedback: t.taaFeedback,
            skinProtect: t.skinProtect
        ))

        var learned: LearnedUpscaler?
        var preprocess: [Double] = []
        var upscale: [Double] = []
        var finish: [Double] = []
        var total: [Double] = []
        var seen = 0
        let warmup = 8

        print("pipeline-ms compute=\(LearnedUpscaler.computeUnitsLabel) count=\(count) warmup=\(warmup)")
        while seen < warmup + count, let sample = output.copyNextSampleBuffer(), let frame = sample.imageBuffer {
            let width = CVPixelBufferGetWidth(frame)
            let height = CVPixelBufferGetHeight(frame)
            if learned == nil {
                learned = try LearnedUpscaler(width: width, height: height)
                print("pipeline-ms input \(width)x\(height) → SPAN \(learned!.inputWidth)x\(learned!.inputHeight)")
                let incoming = CVBufferCopyAttachment(frame, kCVImageBufferChromaLocationTopFieldKey, nil)
                    .map { "\($0)" } ?? "nil"
                print("pipeline-ms chroma incoming=\(incoming)")
            }
            let frame = TiledVideoToolboxUpscaler.prepareSource(frame)
            if seen == 0 {
                let applied = CVBufferCopyAttachment(frame, kCVImageBufferChromaLocationTopFieldKey, nil)
                    .map { "\($0)" } ?? "nil"
                print("pipeline-ms chroma applied=\(applied) flag=\(TiledVideoToolboxUpscaler.chromaSitingLeft ? "left" : "center") resampled=\(TiledVideoToolboxUpscaler.chromaSitingLeft ? "none" : "center")")
            }
            let started = ContinuousClock.now
            let t0 = ContinuousClock.now
            let cleaned = try detail.preprocess(frame)
            let t1 = ContinuousClock.now
            let reconstructed = try learned!.upscale(cleaned)
            let t2 = ContinuousClock.now
            _ = try detail.process(reconstructed)
            let t3 = ContinuousClock.now
            seen += 1
            if seen <= warmup { continue }
            preprocess.append((t1 - t0).milliseconds)
            upscale.append((t2 - t1).milliseconds)
            finish.append((t3 - t2).milliseconds)
            total.append((t3 - started).milliseconds)
        }

        func mean(_ values: [Double]) -> Double {
            guard !values.isEmpty else { return 0 }
            return values.reduce(0, +) / Double(values.count)
        }
        func percentile(_ values: [Double], _ p: Double) -> Double {
            guard !values.isEmpty else { return 0 }
            let sorted = values.sorted()
            let i = min(sorted.count - 1, max(0, Int((Double(sorted.count - 1) * p).rounded())))
            return sorted[i]
        }

        print(String(
            format: "pipeline-ms compute=%@ n=%d  total mean %.2f  p50 %.2f  p95 %.2f  max %.2f  |  preprocess %.2f  span %.2f  detail %.2f",
            LearnedUpscaler.computeUnitsLabel, total.count,
            mean(total), percentile(total, 0.50), percentile(total, 0.95), total.max() ?? 0,
            mean(preprocess), mean(upscale), mean(finish)
        ))
    }
}
