// Apple's temporal scaler fed with video instead of a game.
//
// MetalFX Temporal wants what a game engine has: a jittered colour frame, a
// depth buffer and per-pixel motion vectors. Video has none of those. The
// same trick the community uses to feed DLSS with plain video applies here:
// synthesize the contract. Motion comes from the app's own bounded block
// correspondence (the kernel the temporal stage already trusts), depth is a
// constant plane, jitter is zero. Without sub-pixel jitter the scaler cannot
// recover detail it was never shown, so this is a temporal accumulator with
// Apple's resolve, not a super-resolver. It exists so the lab can put Apple's
// best on the same footage as ours and the harness can score it.

import CoreVideo
import Foundation
import Metal
import MetalFX
import VideoToolbox

final class MetalFXTemporalUpscaler: FrameReconstructor, @unchecked Sendable {
    /// Pseudo-model stem: selecting it on the lab's Model row loads this path.
    static let stem = "metalfx_"

    enum Failure: Error { case unsupported, device, library, pipeline, textureCache, texture, pixelBuffer, format, execution }

    /// True when the Model row, `LUCID_MODEL_STEM`, or the harness's package
    /// path names the MetalFX pseudo-model.
    static var isSelected: Bool {
        if LearnedUpscaler.currentStem == stem { return true }
        if CommandLine.arguments.contains("--pipeline-ms"),
           let path = ProcessInfo.processInfo.environment["LUCID_PIPELINE_MODEL"],
           URL(fileURLWithPath: path).lastPathComponent.hasPrefix("metalfx") { return true }
        return false
    }

    static func supports(_ device: MTLDevice) -> Bool {
        MTLFXTemporalScalerDescriptor.supportsDevice(device)
    }

    let scale = 2
    let inputWidth: Int
    let inputHeight: Int
    var outputWidth: Int { inputWidth * scale }
    var outputHeight: Int { inputHeight * scale }
    /// A 2x output; the detail stage runs at the geometry the 2x learned
    /// graphs use so the two are compared through identical stages.
    let detailReferenceRadius = 2

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let scaler: MTLFXTemporalScaler
    private let textureCache: CVMetalTextureCache
    private let motionPipeline: MTLComputePipelineState
    private let expandPipeline: MTLComputePipelineState
    private let colorTexture: MTLTexture
    private let outputTexture: MTLTexture
    private let depthTexture: MTLTexture
    private let motionTexture: MTLTexture
    private let fieldTexture: MTLTexture
    private let previousLuma: MTLTexture
    private var historyValid = false
    private var transfer: VTPixelTransferSession?
    private var rgbPool: CVPixelBufferPool?
    private var outPool: CVPixelBufferPool?
    private var finalPool: CVPixelBufferPool?

    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // Bounded block correspondence at source resolution, the temporal stage's
    // own estimator. The field holds (dx, dy, confidence, error) per 8x8 block,
    // where source(p) matches previous(p + d).
    kernel void mfx_motion_blocks(texture2d<float, access::read> source [[texture(0)]],
                                  texture2d<float, access::read> previous [[texture(1)]],
                                  texture2d<float, access::write> field [[texture(2)]],
                                  constant float& valid [[buffer(0)]],
                                  uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= field.get_width() || gid.y >= field.get_height()) return;
        if (valid < 0.5f) { field.write(float4(0), gid); return; }
        int2 limit = int2(source.get_width(), source.get_height()) - 1;
        int2 centre = min(int2(gid) * 8 + 4, limit);
        float stationaryError = 0.0f;
        for (int py = -4; py <= 4; py += 2) {
            for (int px = -4; px <= 4; px += 2) {
                const uint2 at = uint2(clamp(centre + int2(px, py), int2(0), limit));
                stationaryError += abs(source.read(at).r - previous.read(at).r);
            }
        }
        stationaryError /= 25.0f;
        if (stationaryError <= 6.5f / 255.0f) {
            field.write(float4(0, 0, 1, stationaryError), gid);
            return;
        }
        float best = 1e6f, error = 1.0f;
        int2 displacement = int2(0);
        for (int pass = 0; pass < 2; ++pass) {
            int2 origin = pass == 0 ? int2(0) : displacement;
            int radius = pass == 0 ? 8 : 1;
            int step = pass == 0 ? 2 : 1;
            for (int y = -radius; y <= radius; y += step) {
                for (int x = -radius; x <= radius; x += step) {
                    int2 delta = origin + int2(x, y);
                    float cost = 0.0f;
                    for (int py = -4; py <= 4; py += 2) {
                        for (int px = -4; px <= 4; px += 2) {
                            int2 a = clamp(centre + int2(px, py), int2(0), limit);
                            int2 b = clamp(a + delta, int2(0), limit);
                            cost += abs(source.read(uint2(a)).r - previous.read(uint2(b)).r);
                        }
                    }
                    cost /= 25.0f;
                    float regularized = cost + 0.00015f * dot(float2(delta), float2(delta));
                    if (regularized < best) { best = regularized; error = cost; displacement = delta; }
                }
            }
        }
        if (stationaryError - error < 2.0f / 255.0f) {
            displacement = int2(0);
            error = stationaryError;
        }
        const float confidence = 1.0f - smoothstep(6.0f / 255.0f, 16.0f / 255.0f, error);
        field.write(float4(float2(displacement), confidence, error), gid);
    }

    // Per-pixel motion in input pixels, MetalFX convention: the vector points
    // to where this pixel was in the previous colour frame. Blocks the
    // estimator does not trust are declared stationary; the scaler's own
    // neighbourhood clip then limits what the history may contribute.
    kernel void mfx_motion_expand(texture2d<float, access::read> field [[texture(0)]],
                                  texture2d<float, access::write> motion [[texture(1)]],
                                  uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= motion.get_width() || gid.y >= motion.get_height()) return;
        const float4 block = field.read(gid / 8);
        const float2 vector = block.z > 0.5f ? block.xy : float2(0);
        motion.write(float4(vector, 0, 0), gid);
    }
    """

    init(width: Int, height: Int) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw Failure.device }
        guard Self.supports(device) else { throw Failure.unsupported }
        guard width > 0, height > 0, width % 8 == 0, height % 8 == 0 else { throw Failure.format }
        self.device = device
        inputWidth = width
        inputHeight = height
        guard let queue = device.makeCommandQueue() else { throw Failure.device }
        self.queue = queue

        let library: MTLLibrary
        do { library = try device.makeLibrary(source: Self.source, options: nil) } catch { throw Failure.library }
        guard let motionFunction = library.makeFunction(name: "mfx_motion_blocks"),
              let expandFunction = library.makeFunction(name: "mfx_motion_expand"),
              let motionPipeline = try? device.makeComputePipelineState(function: motionFunction),
              let expandPipeline = try? device.makeComputePipelineState(function: expandFunction)
        else { throw Failure.pipeline }
        self.motionPipeline = motionPipeline
        self.expandPipeline = expandPipeline

        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache) == kCVReturnSuccess,
              let cache else { throw Failure.textureCache }
        textureCache = cache

        let descriptor = MTLFXTemporalScalerDescriptor()
        descriptor.inputWidth = width
        descriptor.inputHeight = height
        descriptor.outputWidth = width * scale
        descriptor.outputHeight = height * scale
        descriptor.colorTextureFormat = .bgra8Unorm
        descriptor.depthTextureFormat = .r32Float
        descriptor.motionTextureFormat = .rg16Float
        descriptor.outputTextureFormat = .bgra8Unorm
        descriptor.isAutoExposureEnabled = false
        descriptor.isInputContentPropertiesEnabled = false
        descriptor.requiresSynchronousInitialization = true
        guard let scaler = descriptor.makeTemporalScaler(device: device) else { throw Failure.unsupported }
        self.scaler = scaler

        func texture(_ format: MTLPixelFormat, _ w: Int, _ h: Int, _ usage: MTLTextureUsage) throws -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h, mipmapped: false)
            d.usage = usage
            d.storageMode = .private
            guard let t = device.makeTexture(descriptor: d) else { throw Failure.texture }
            return t
        }
        colorTexture = try texture(.bgra8Unorm, width, height, scaler.colorTextureUsage.union(.shaderWrite))
        outputTexture = try texture(.bgra8Unorm, width * scale, height * scale, scaler.outputTextureUsage.union(.shaderRead))
        depthTexture = try texture(.r32Float, width, height, scaler.depthTextureUsage.union(.shaderWrite))
        motionTexture = try texture(.rg16Float, width, height, scaler.motionTextureUsage.union(.shaderWrite))
        fieldTexture = try texture(.rgba32Float, (width + 7) / 8, (height + 7) / 8, [.shaderRead, .shaderWrite])
        previousLuma = try texture(.r8Unorm, width, height, [.shaderRead, .shaderWrite])

        // A constant far plane: the scaler only uses depth to disambiguate
        // motion at object edges, and video has one layer.
        let ones = [Float](repeating: 1, count: width * height)
        let staging = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
        staging.storageMode = .shared
        guard let shared = device.makeTexture(descriptor: staging) else { throw Failure.texture }
        ones.withUnsafeBytes { bytes in
            shared.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                           withBytes: bytes.baseAddress!, bytesPerRow: width * MemoryLayout<Float>.stride)
        }
        guard let commands = queue.makeCommandBuffer(), let blit = commands.makeBlitCommandEncoder() else { throw Failure.execution }
        blit.copy(from: shared, to: depthTexture)
        blit.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()

        scaler.isDepthReversed = false
        scaler.jitterOffsetX = 0
        scaler.jitterOffsetY = 0
        scaler.motionVectorScaleX = 1
        scaler.motionVectorScaleY = 1
        scaler.preExposure = 1
        scaler.inputContentWidth = width
        scaler.inputContentHeight = height
        scaler.colorTexture = colorTexture
        scaler.depthTexture = depthTexture
        scaler.motionTexture = motionTexture
        scaler.outputTexture = outputTexture

        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transfer)
    }

    func upscale(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        let sourceFormat = CVPixelBufferGetPixelFormatType(source)
        guard CVPixelBufferIsPlanar(source),
              CVPixelBufferGetWidth(source) == inputWidth, CVPixelBufferGetHeight(source) == inputHeight
        else { throw Failure.format }

        // The scaler works in RGB; the pipeline in 4:2:0. Same conversion the
        // learned path makes, so the two see identical colour.
        let rgb = try convert(source, to: kCVPixelFormatType_32BGRA, width: inputWidth, height: inputHeight, pool: &rgbPool)
        let out = try makeBuffer(width: outputWidth, height: outputHeight, format: kCVPixelFormatType_32BGRA, pool: &outPool)
        let (rgbTexture, rgbRef) = try texture(rgb, plane: 0, format: .bgra8Unorm)
        let (lumaTexture, lumaRef) = try texture(source, plane: 0, format: .r8Unorm)
        let (outTexture, outRef) = try texture(out, plane: 0, format: .bgra8Unorm)
        defer { _ = (rgbRef, lumaRef, outRef) }

        guard let commands = queue.makeCommandBuffer() else { throw Failure.execution }
        let threads = MTLSize(width: 16, height: 16, depth: 1)
        func grid(_ w: Int, _ h: Int) -> MTLSize { MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1) }

        if let blit = commands.makeBlitCommandEncoder() {
            blit.copy(from: rgbTexture, to: colorTexture)
            blit.endEncoding()
        }
        guard let motion = commands.makeComputeCommandEncoder() else { throw Failure.execution }
        motion.setComputePipelineState(motionPipeline)
        motion.setTexture(lumaTexture, index: 0)
        motion.setTexture(previousLuma, index: 1)
        motion.setTexture(fieldTexture, index: 2)
        var valid: Float = historyValid ? 1 : 0
        motion.setBytes(&valid, length: MemoryLayout<Float>.stride, index: 0)
        motion.dispatchThreadgroups(grid(fieldTexture.width, fieldTexture.height), threadsPerThreadgroup: threads)
        motion.endEncoding()
        guard let expand = commands.makeComputeCommandEncoder() else { throw Failure.execution }
        expand.setComputePipelineState(expandPipeline)
        expand.setTexture(fieldTexture, index: 0)
        expand.setTexture(motionTexture, index: 1)
        expand.dispatchThreadgroups(grid(inputWidth, inputHeight), threadsPerThreadgroup: threads)
        expand.endEncoding()

        scaler.reset = !historyValid
        scaler.encode(commandBuffer: commands)

        if let blit = commands.makeBlitCommandEncoder() {
            blit.copy(from: outputTexture, to: outTexture)
            blit.copy(from: lumaTexture, to: previousLuma)
            blit.endEncoding()
        }
        commands.commit()
        commands.waitUntilCompleted()
        guard commands.status == .completed else { historyValid = false; throw Failure.execution }
        historyValid = true

        // Predicted RGB in the input encoding, exactly as the learned path tags it.
        var predictedColor = VideoColorInfo.read(from: source)
        predictedColor.matrix = "rgb"; predictedColor.fullRange = true
        guard predictedColor.apply(to: out) else { throw Failure.pixelBuffer }
        return try convert(out, to: sourceFormat, width: outputWidth, height: outputHeight, pool: &finalPool)
    }

    private func texture(_ buffer: CVPixelBuffer, plane: Int, format: MTLPixelFormat) throws -> (MTLTexture, CVMetalTexture) {
        let planar = CVPixelBufferIsPlanar(buffer)
        let width = planar ? CVPixelBufferGetWidthOfPlane(buffer, plane) : CVPixelBufferGetWidth(buffer)
        let height = planar ? CVPixelBufferGetHeightOfPlane(buffer, plane) : CVPixelBufferGetHeight(buffer)
        var ref: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, buffer, nil, format, width, height, plane, &ref)
        guard status == kCVReturnSuccess, let ref, let texture = CVMetalTextureGetTexture(ref) else { throw Failure.texture }
        return (texture, ref)
    }

    private func makeBuffer(width: Int, height: Int, format: OSType, pool: inout CVPixelBufferPool?) throws -> CVPixelBuffer {
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
        return destination
    }

    private func convert(_ source: CVPixelBuffer, to format: OSType,
                         width: Int, height: Int, pool: inout CVPixelBufferPool?) throws -> CVPixelBuffer {
        let destination = try makeBuffer(width: width, height: height, format: format, pool: &pool)
        guard let transfer else { throw Failure.pixelBuffer }
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
