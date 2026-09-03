// Offline engine benchmark: decodes a low-resolution clip and its full-resolution
// reference, runs Apple's low-latency scaler (two chained 2× passes) and Apple's
// temporal super-resolution scaler (one 4× pass) on the same frames, and writes
// PNGs for metrics.  Usage: engine-bench <input.mp4> <reference.mp4> <firstFrame> <count> <outDir>
import AVFoundation
import AppKit
import CoreImage
import CoreMedia
import CoreVideo
import Foundation
@preconcurrency import VideoToolbox

@available(macOS 26.0, *)
@main
struct EngineBench {
    static func reader(_ path: String) throws -> (AVAssetReader, AVAssetReaderTrackOutput) {
        let asset = AVURLAsset(url: URL(fileURLWithPath: path))
        let sem = DispatchSemaphore(value: 0); var tracks: [AVAssetTrack] = []
        Task { tracks = (try? await asset.loadTracks(withMediaType: .video)) ?? []; sem.signal() }; sem.wait()
        guard let track = tracks.first else { throw NSError(domain: "bench", code: 1) }
        let r = try AVAssetReader(asset: asset)
        let o = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ])
        r.add(o); r.startReading(); return (r, o)
    }
    static func pool(_ attrs: [String: Any]) -> CVPixelBufferPool {
        var merged = attrs; merged[kCVPixelBufferMetalCompatibilityKey as String] = true
        if merged[kCVPixelBufferIOSurfacePropertiesKey as String] == nil { merged[kCVPixelBufferIOSurfacePropertiesKey as String] = [:] as [String: Any] }
        var p: CVPixelBufferPool?; CVPixelBufferPoolCreate(kCFAllocatorDefault, [kCVPixelBufferPoolMinimumBufferCountKey as String: 4] as CFDictionary, merged as CFDictionary, &p); return p!
    }
    static func buffer(_ p: CVPixelBufferPool) -> CVPixelBuffer { var b: CVPixelBuffer?; CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, p, &b); return b! }
    static func png(_ b: CVPixelBuffer, _ path: String, ctx: CIContext) {
        try? ctx.writePNGRepresentation(of: CIImage(cvPixelBuffer: b), to: URL(fileURLWithPath: path), format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
    }

    static func main() async throws {
        let a = CommandLine.arguments
        let inputPath = a[1], refPath = a[2], first = Int(a[3])!, count = Int(a[4])!, out = a[5]
        try? FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)
        let ctx = CIContext()
        let (_, input) = try reader(inputPath)
        let (_, ref) = try reader(refPath)
        let compositor = try MetalTileCompositor()

        var lowLatency: TiledVideoToolboxUpscaler?
        var temporal: (VTFrameProcessor, VTSuperResolutionScalerConfiguration, CVPixelBufferPool, CVPixelBufferPool)?
        var prevSrc: CVPixelBuffer?, prevOut: CVPixelBuffer?
        var llTimes: [Double] = [], tsrTimes: [Double] = []
        var index = 0
        while let sample = input.copyNextSampleBuffer(), let frame = sample.imageBuffer {
            let refSample = ref.copyNextSampleBuffer()
            let w = CVPixelBufferGetWidth(frame), h = CVPixelBufferGetHeight(frame)
            if lowLatency == nil {
                let stage1 = try TiledVideoToolboxUpscaler(width: w, height: h, compositor: compositor)
                stage1.nextStage = try TiledVideoToolboxUpscaler(width: w * 2, height: h * 2, compositor: compositor)
                lowLatency = stage1
                guard let cfg = VTSuperResolutionScalerConfiguration(frameWidth: w, frameHeight: h, scaleFactor: 4, inputType: .video, usePrecomputedFlow: false, qualityPrioritization: .normal, revision: .revision1) else { print("temporal config nil"); exit(1) }
                if cfg.configurationModelStatus != .ready {
                    await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in cfg.downloadConfigurationModel { _ in c.resume() } }
                }
                let proc = VTFrameProcessor(); try proc.startSession(configuration: cfg)
                temporal = (proc, cfg, pool(cfg.sourcePixelBufferAttributes), pool(cfg.destinationPixelBufferAttributes))
                print("input \(w)x\(h); temporal source format: \(cfg.sourcePixelBufferAttributes[kCVPixelBufferPixelFormatTypeKey as String] ?? "?")")
            }
            let pts = CMTime(value: CMTimeValue(index), timescale: 30)
            // Low-latency chain (4× via two passes).
            let t0 = ContinuousClock.now
            let ll = try await lowLatency!.upscale(frame, pts: pts)
            llTimes.append((ContinuousClock.now - t0).milliseconds)
            // Temporal 4×.
            let (proc, _, srcPool, dstPool) = temporal!
            let src = buffer(srcPool); try compositor.perform([.init(source: frame, sourceRect: CGRect(x: 0, y: 0, width: w, height: h), destination: src, destinationOrigin: .zero)])
            let dst = buffer(dstPool)
            guard let sf = VTFrameProcessorFrame(buffer: src, presentationTimeStamp: pts), let df = VTFrameProcessorFrame(buffer: dst, presentationTimeStamp: pts) else { continue }
            let pf = prevSrc.flatMap { VTFrameProcessorFrame(buffer: $0, presentationTimeStamp: CMTime(value: CMTimeValue(index - 1), timescale: 30)) }
            let pof = prevOut.flatMap { VTFrameProcessorFrame(buffer: $0, presentationTimeStamp: CMTime(value: CMTimeValue(index - 1), timescale: 30)) }
            guard let params = VTSuperResolutionScalerParameters(sourceFrame: sf, previousFrame: pf, previousOutputFrame: pof, opticalFlow: nil, submissionMode: pf == nil ? .random : .sequential, destinationFrame: df) else { continue }
            let t1 = ContinuousClock.now
            try await proc.process(parameters: params)
            tsrTimes.append((ContinuousClock.now - t1).milliseconds)
            prevSrc = src; prevOut = dst

            if index >= first && index < first + count {
                let n = String(format: "%03d", index)
                png(frame, "\(out)/\(n)-input.png", ctx: ctx)
                png(ll, "\(out)/\(n)-lowlatency.png", ctx: ctx)
                png(dst, "\(out)/\(n)-temporal.png", ctx: ctx)
                if let rb = refSample?.imageBuffer { png(rb, "\(out)/\(n)-reference.png", ctx: ctx) }
            }
            index += 1
            if index >= first + count { break }
        }
        func stats(_ t: [Double]) -> String { let w = t.dropFirst(5); return w.isEmpty ? "n/a" : String(format: "%.1f ms", w.reduce(0, +) / Double(w.count)) }
        print("frames \(index); low-latency 4×: \(stats(llTimes)); temporal 4×: \(stats(tsrTimes))")
        await lowLatency?.end(); temporal?.0.endSession()
    }
}
