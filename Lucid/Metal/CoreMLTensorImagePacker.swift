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
    private final class StorageOwner {
        let buffer: MTLBuffer
        init(_ buffer: MTLBuffer) { self.buffer = buffer }
    }

    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLComputePipelineState
    let output: CVPixelBuffer
    let outputStorage: MTLBuffer
    let width: Int
    let height: Int
    var transferModes: Set<String> = []
    var storage: [String: Any] = [:]

    init(width: Int, height: Int) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw Self.failure("Metal unavailable")
        }
        guard width > 0, height > 0, width <= 16384, height <= 16384 else {
            throw Self.failure("invalid output geometry")
        }
        self.device = device; self.queue = queue
        self.width = width; self.height = height
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void pack_rgb(device const uchar* raw [[buffer(0)]], constant uint4& p [[buffer(1)]],
            device uchar4* output [[buffer(2)]], constant uint2& size [[buffer(3)]],
            uint2 xy [[thread_position_in_grid]]) {
            if (xy.x >= size.x || xy.y >= size.y) return;
            uint offset = xy.y * p.y + xy.x * p.z;
            float3 rgb;
            for (uint c=0; c<3; ++c) {
                uint at = offset + c*p.x;
                rgb[c] = p.w == 16 ? float(reinterpret_cast<device const half*>(raw)[at])
                                  : reinterpret_cast<device const float*>(raw)[at];
            }
            // Match the observed Core ML GPU RGB8 image boundary: first
            // binary16 rounding, then integer ties-to-even. Direct FP32
            // rounding changes thresholds (141.4453125 must become 142).
            // A half-to-float cast round trip was eliminated on the target
            // GPU even with safe math. Round the significand explicitly.
            // RGB is nonnegative and <=255; half-subnormal values all round
            // to integer zero, so only normal-half precision matters here.
            uint3 bits = as_type<uint3>(clamp(rgb, 0.0f, 255.0f));
            bits = (bits + 0xfffu + ((bits >> 13u) & 1u)) & 0xffffe000u;
            float3 imageValues = as_type<float3>(bits);
            uchar3 encoded = uchar3(rint(imageValues));
            output[xy.y * size.x + xy.x] = uchar4(encoded.b, encoded.g, encoded.r, 255);
        }
        """
        let options = MTLCompileOptions()
        options.mathMode = .safe
        let library = try device.makeLibrary(source: source, options: options)
        guard let function = library.makeFunction(name: "pack_rgb") else { throw Self.failure("missing kernel") }
        pipeline = try device.makeComputePipelineState(function: function)
        // The matching Core ML image is not IOSurface-backed. Preserve that
        // representation for VideoToolbox while letting Metal write directly
        // into unified shared memory. The pixel buffer retains its storage even
        // if it outlives this adapter; no CPU RGB copy or escaped Core ML borrow.
        guard let buffer = device.makeBuffer(length: width * height * 4, options: .storageModeShared) else {
            throw Self.failure("output storage allocation failed")
        }
        outputStorage = buffer
        let owner = Unmanaged.passRetained(StorageOwner(buffer)).toOpaque()
        var created: CVPixelBuffer?
        let status = CVPixelBufferCreateWithBytes(nil, width, height, kCVPixelFormatType_32BGRA,
            buffer.contents(), width * 4, { context, _ in
                if let context { Unmanaged<StorageOwner>.fromOpaque(context).release() }
            }, owner, nil, &created)
        guard status == kCVReturnSuccess, let created else {
            Unmanaged<StorageOwner>.fromOpaque(owner).release()
            throw Self.failure("output pixel buffer creation failed")
        }
        output = created
    }

    func pack(_ array: MLMultiArray) throws -> CVPixelBuffer {
        let shape = array.shape.map(\.intValue)
        guard shape == [1, 3, height, width],
              array.dataType == .float16 || array.dataType == .float32 else { throw Self.failure("unexpected tensor") }
        let scalarBytes = array.dataType == .float16 ? 2 : 4
        try array.withUnsafeMutableBytes { raw, strides in
            guard strides.count == 4, strides.allSatisfy({ $0 > 0 && $0 <= Int(UInt32.max) }),
                  let address = raw.baseAddress else { throw Self.failure("invalid tensor strides") }
            let required = (2 * strides[1] + (height-1)*strides[2] + (width-1)*strides[3] + 1)*scalarBytes
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
                       "dtype_bits": scalarBytes * 8, "page_size": page,
                       "output_backing": "shared Metal buffer, non-IOSurface CVPixelBuffer"]
            var params = SIMD4<UInt32>(UInt32(strides[1]), UInt32(strides[2]), UInt32(strides[3]), UInt32(scalarBytes*8))
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&params, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 1)
            var size = SIMD2<UInt32>(UInt32(width), UInt32(height))
            encoder.setBuffer(outputStorage, offset: 0, index: 2)
            encoder.setBytes(&size, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 3)
            encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
            guard command.status == .completed else { throw command.error ?? Self.failure("Metal pack failed") }
        }
        return output
    }
}
