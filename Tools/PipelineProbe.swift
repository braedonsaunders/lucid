// Headless check of the tiled super-resolution path: decode frames from a test
// MP4 as 420v, run TiledVideoToolboxUpscaler (Metal plane blits + N sessions),
// time it, and write the first output as PNG for inspection.
// Build: see Tools/build-pipeline-probe.sh
import AVFoundation
import AppKit
import CoreImage
import CoreMedia
import CoreVideo
import Foundation

@available(macOS 26.0, *)
@main
struct PipelineProbe {
    static func main() async throws {
        let args = CommandLine.arguments
        let path = args.count > 1 ? args[1] : "TestSite/low-bitrate-720p.mp4"
        let outPath = args.count > 2 ? args[2] : ".build/shots/pipeline-out.png"
        let frameCount = args.count > 3 ? Int(args[3]) ?? 60 : 60

        let asset = AVURLAsset(url: URL(fileURLWithPath: path))
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { print("no video track"); exit(1) }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ])
        reader.add(output)
        reader.startReading()

        let compositor = try MetalTileCompositor()
        var upscaler: TiledVideoToolboxUpscaler?
        var times: [Double] = []
        var written = false
        var index = 0
        while index < frameCount, let sample = output.copyNextSampleBuffer(), let buffer = sample.imageBuffer {
            let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
            if upscaler == nil {
                upscaler = try TiledVideoToolboxUpscaler(width: w, height: h, compositor: compositor)
                print("source \(w)x\(h) fmt=\(CVPixelBufferGetPixelFormatType(buffer)) layout=\(upscaler!.layout)")
            }
            let start = ContinuousClock.now
            let result = try await upscaler!.upscale(buffer, pts: CMSampleBufferGetPresentationTimeStamp(sample))
            let ms = (ContinuousClock.now - start).milliseconds
            times.append(ms)
            if index == 5, !written {
                let image = CIImage(cvPixelBuffer: result)
                let context = CIContext()
                if let cg = context.createCGImage(image, from: image.extent) {
                    let rep = NSBitmapImageRep(cgImage: cg)
                    if let png = rep.representation(using: .png, properties: [:]) {
                        try png.write(to: URL(fileURLWithPath: outPath))
                        print("wrote \(outPath) \(CVPixelBufferGetWidth(result))x\(CVPixelBufferGetHeight(result)) fmt=\(CVPixelBufferGetPixelFormatType(result))")
                        written = true
                    }
                }
                // Also write the source frame for side-by-side comparison.
                let src = CIImage(cvPixelBuffer: buffer)
                if let cg = context.createCGImage(src, from: src.extent) {
                    let rep = NSBitmapImageRep(cgImage: cg)
                    try rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: outPath.replacingOccurrences(of: ".png", with: "-source.png")))
                }
            }
            index += 1
        }
        let warm = times.dropFirst(10)
        let sorted = warm.sorted()
        print(String(format: "frames=%d first=%.1f ms  warm median=%.2f ms  p95=%.2f ms  max=%.2f ms", times.count, times.first ?? 0, sorted[sorted.count / 2], sorted[Int(Double(sorted.count) * 0.95)], sorted.last ?? 0))
        await upscaler?.end()
    }
}
