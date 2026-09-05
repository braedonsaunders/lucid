import CoreML
import CoreVideo
import Foundation
import Metal

/// Serial-use adapter. GPU reads finish within Core ML's storage borrow.
@available(macOS 26.0, *)
final class CoreMLTensorImagePacker {
    private static func failure(_ message: String) -> NSError {
        NSError(domain: "Lucid.TensorImagePacker", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
    private static func pixelBuffer(_ width: Int, _ height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
            [kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary,
            &buffer) == kCVReturnSuccess, let buffer else { throw Self.failure("pixel allocation failed") }
        return buffer
    }


    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLComputePipelineState
    let output: CVPixelBuffer
    let texture: MTLTexture
    let retainedTexture: CVMetalTexture
    let cache: CVMetalTextureCache
    var transferModes: Set<String> = []
    var storage: [String: Any] = [:]

    init(width: Int, height: Int) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw Self.failure("Metal unavailable")
        }
        self.device = device; self.queue = queue
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void pack_rgb(device const uchar* raw [[buffer(0)]], constant uint4& p [[buffer(1)]],
            texture2d<float, access::write> output [[texture(0)]], uint2 xy [[thread_position_in_grid]]) {
            if (xy.x >= output.get_width() || xy.y >= output.get_height()) return;
            uint offset = xy.y * p.y + xy.x * p.z;
            float3 rgb;
            for (uint c=0; c<3; ++c) {
                uint at = offset + c*p.x;
                rgb[c] = p.w == 16 ? float(reinterpret_cast<device const half*>(raw)[at])
                                  : reinterpret_cast<device const float*>(raw)[at];
            }
            output.write(float4(floor(clamp(rgb, 0.0f, 255.0f) + 0.5f)/255.0f, 1.0f), xy);
        }
        """
        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: "pack_rgb") else { throw Self.failure("missing kernel") }
        pipeline = try device.makeComputePipelineState(function: function)
        output = try Self.pixelBuffer(width, height)
        var createdCache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(nil, nil, device, nil, &createdCache) == kCVReturnSuccess,
              let createdCache else { throw Self.failure("texture cache failed") }
        cache = createdCache
        var createdTexture: CVMetalTexture?
        guard CVMetalTextureCacheCreateTextureFromImage(nil, cache, output, nil, .bgra8Unorm,
            width, height, 0, &createdTexture) == kCVReturnSuccess,
              let createdTexture, let texture = CVMetalTextureGetTexture(createdTexture) else {
            throw Self.failure("output texture failed")
        }
        retainedTexture = createdTexture; self.texture = texture
    }

    func pack(_ array: MLMultiArray) throws -> CVPixelBuffer {
        let shape = array.shape.map(\.intValue)
        guard shape == [1, 3, texture.height, texture.width],
              array.dataType == .float16 || array.dataType == .float32 else { throw Self.failure("unexpected tensor") }
        let scalarBytes = array.dataType == .float16 ? 2 : 4
        try array.withUnsafeMutableBytes { raw, strides in
            guard strides.count == 4, strides.allSatisfy({ $0 > 0 && $0 <= Int(UInt32.max) }),
                  let address = raw.baseAddress else { throw Self.failure("invalid tensor strides") }
            let required = (2 * strides[1] + (texture.height-1)*strides[2] + (texture.width-1)*strides[3] + 1)*scalarBytes
            guard required <= raw.count, required / scalarBytes <= Int(UInt32.max) else { throw Self.failure("tensor storage shorter than its strides") }
            // A no-copy Metal buffer requires page-aligned, page-sized storage.
            // Keep GPU access inside the Core ML borrowing closure and wait for it.
            let page = Int(getpagesize())
            let canWrap = device.hasUnifiedMemory && Int(bitPattern: address) % page == 0 && raw.count % page == 0
            let wrapped = canWrap ? device.makeBuffer(bytesNoCopy: address, length: raw.count,
                options: .storageModeShared, deallocator: nil) : nil
            guard let buffer = wrapped ?? device.makeBuffer(bytes: address, length: raw.count, options: .storageModeShared),
                  let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
                throw Self.failure("Metal buffer or encoder allocation failed")
            }
            transferModes.insert(wrapped == nil ? "explicit copy" : "page-aligned shared storage")
            storage = ["shape": shape, "strides": strides, "bytes": raw.count,
                       "dtype_bits": scalarBytes * 8, "page_size": page]
            var params = SIMD4<UInt32>(UInt32(strides[1]), UInt32(strides[2]), UInt32(strides[3]), UInt32(scalarBytes*8))
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&params, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 1)
            encoder.setTexture(texture, index: 0)
            encoder.dispatchThreads(MTLSize(width: texture.width, height: texture.height, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
            guard command.status == .completed else { throw command.error ?? Self.failure("Metal pack failed") }
        }
        return output
    }
}

