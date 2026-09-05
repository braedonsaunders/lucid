import CoreML
import CoreVideo
import Foundation
import Metal

func failure(_ message: String) -> NSError {
    NSError(domain: "tensor-output-bench", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
}

func ms(_ duration: Duration) -> Double {
    let c = duration.components
    return Double(c.seconds) * 1000 + Double(c.attoseconds) / 1e15
}

func pixelBuffer(_ width: Int, _ height: Int) throws -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
        [kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary,
        &buffer) == kCVReturnSuccess, let buffer else { throw failure("pixel allocation failed") }
    return buffer
}

final class TensorPacker {
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
            throw failure("Metal unavailable")
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
        guard let function = library.makeFunction(name: "pack_rgb") else { throw failure("missing kernel") }
        pipeline = try device.makeComputePipelineState(function: function)
        output = try pixelBuffer(width, height)
        var createdCache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(nil, nil, device, nil, &createdCache) == kCVReturnSuccess,
              let createdCache else { throw failure("texture cache failed") }
        cache = createdCache
        var createdTexture: CVMetalTexture?
        guard CVMetalTextureCacheCreateTextureFromImage(nil, cache, output, nil, .bgra8Unorm,
            width, height, 0, &createdTexture) == kCVReturnSuccess,
              let createdTexture, let texture = CVMetalTextureGetTexture(createdTexture) else {
            throw failure("output texture failed")
        }
        retainedTexture = createdTexture; self.texture = texture
    }

    func pack(_ array: MLMultiArray) throws -> CVPixelBuffer {
        let shape = array.shape.map(\.intValue)
        guard shape == [1, 3, texture.height, texture.width],
              array.dataType == .float16 || array.dataType == .float32 else { throw failure("unexpected tensor") }
        let scalarBytes = array.dataType == .float16 ? 2 : 4
        try array.withUnsafeMutableBytes { raw, strides in
            guard strides.count == 4, strides.allSatisfy({ $0 > 0 && $0 <= Int(UInt32.max) }),
                  let address = raw.baseAddress else { throw failure("invalid tensor strides") }
            let required = (2 * strides[1] + (texture.height-1)*strides[2] + (texture.width-1)*strides[3] + 1)*scalarBytes
            guard required <= raw.count else { throw failure("tensor storage shorter than its strides") }
            // A no-copy Metal buffer requires page-aligned, page-sized storage.
            // Keep GPU access inside the Core ML borrowing closure and wait for it.
            let page = Int(getpagesize())
            let canWrap = device.hasUnifiedMemory && Int(bitPattern: address) % page == 0 && raw.count % page == 0
            let wrapped = canWrap ? device.makeBuffer(bytesNoCopy: address, length: raw.count,
                options: .storageModeShared, deallocator: nil) : nil
            guard let buffer = wrapped ?? device.makeBuffer(bytes: address, length: raw.count, options: .storageModeShared),
                  let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
                throw failure("Metal buffer or encoder allocation failed")
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
            guard command.status == .completed else { throw command.error ?? failure("Metal pack failed") }
        }
        return output
    }
}

func difference(_ a: CVPixelBuffer, _ b: CVPixelBuffer) throws -> [String: Double] {
    let w = CVPixelBufferGetWidth(a), h = CVPixelBufferGetHeight(a)
    guard CVPixelBufferGetWidth(b) == w, CVPixelBufferGetHeight(b) == h,
          CVPixelBufferGetPixelFormatType(a) == kCVPixelFormatType_32BGRA else { throw failure("comparison geometry or format changed") }
    CVPixelBufferLockBaseAddress(a, .readOnly); CVPixelBufferLockBaseAddress(b, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(a, .readOnly); CVPixelBufferUnlockBaseAddress(b, .readOnly) }
    let x = CVPixelBufferGetBaseAddress(a)!.assumingMemoryBound(to: UInt8.self)
    let y = CVPixelBufferGetBaseAddress(b)!.assumingMemoryBound(to: UInt8.self)
    var maximum = 0, sum: Int64 = 0
    for row in 0..<h { for col in 0..<w { for c in 0..<3 {
        let delta = abs(Int(x[row*CVPixelBufferGetBytesPerRow(a)+col*4+c]) - Int(y[row*CVPixelBufferGetBytesPerRow(b)+col*4+c]))
        maximum = max(maximum, delta); sum += Int64(delta)
    } } }
    return ["max_rgb": Double(maximum), "mean_rgb": Double(sum)/Double(w*h*3)]
}

func main() throws {
    let args = CommandLine.arguments
    guard args.count == 4, let count = Int(args[3]), count >= 20 else {
        throw failure("usage: TensorOutputBench MODEL_DIRECTORY REPORT COUNT>=20")
    }
    let directory = URL(fileURLWithPath: args[1]), reportURL = URL(fileURLWithPath: args[2])
    guard !FileManager.default.fileExists(atPath: reportURL.path) else { throw failure("fresh report required") }
    let labels = ["image4x", "tensor4x_fp32", "tensor4x_fp16"]
    var compiledURLs: [URL] = []
    defer { for url in compiledURLs { try? FileManager.default.removeItem(at: url) } }
    let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuAndGPU
    let models = try labels.map { label -> MLModel in
        let compiled = try MLModel.compileModel(at: directory.appendingPathComponent(label+".mlpackage"))
        compiledURLs.append(compiled)
        return try MLModel(contentsOf: compiled, configuration: configuration)
    }
    guard let input = models[0].modelDescription.inputDescriptionsByName["input"]?.imageConstraint,
          input.pixelFormatType == kCVPixelFormatType_32BGRA else { throw failure("BGRA input required") }
    let width = input.pixelsWide, height = input.pixelsHigh
    let packers = try [TensorPacker(width: width*4,height: height*4), TensorPacker(width: width*4,height: height*4)]
    let frames = try (0..<4).map { index -> CVPixelBuffer in
        let frame = try pixelBuffer(width,height)
        CVPixelBufferLockBaseAddress(frame, [])
        let bytes = CVPixelBufferGetBaseAddress(frame)!.assumingMemoryBound(to: UInt8.self)
        var random: UInt32 = 20260905 + UInt32(index)
        for y in 0..<height { for x in 0..<width {
            let at = y*CVPixelBufferGetBytesPerRow(frame)+x*4
            for c in 0..<3 { random = random &* 1664525 &+ 1013904223; bytes[at+c] = UInt8(truncatingIfNeeded: random >> 16) }
            bytes[at+3] = 255
        } }
        CVPixelBufferUnlockBaseAddress(frame, [])
        return frame
    }
    var accuracy: [String: [[String: Double]]] = [:]
    for frame in frames {
        let provider = try MLDictionaryFeatureProvider(dictionary:["input":MLFeatureValue(pixelBuffer:frame)])
        let baseline = try models[0].prediction(from:provider).featureValue(for:"output")!.imageBufferValue!
        for index in 1..<3 {
            let array = try models[index].prediction(from:provider).featureValue(for:"output")!.multiArrayValue!
            let packed = try packers[index-1].pack(array)
            accuracy[labels[index],default:[]].append(try difference(baseline,packed))
        }
    }
    var predictions = Array(repeating:[Double](),count:3), packing = predictions
    let correct = accuracy.values.flatMap { $0 }.allSatisfy { $0["max_rgb"]! <= 3 && $0["mean_rgb"]! <= 0.6 }
    if correct {
        for step in 0..<(count+10) {
            try autoreleasepool {
                let provider = try MLDictionaryFeatureProvider(dictionary:["input":MLFeatureValue(pixelBuffer:frames[step%4])])
                for index in (step%2 == 0 ? [0,1,2] : [2,1,0]) {
                    let start = ContinuousClock.now
                    let result = try models[index].prediction(from:provider)
                    let inferred = ContinuousClock.now
                    if index > 0 { _ = try packers[index-1].pack(result.featureValue(for:"output")!.multiArrayValue!) }
                    else { guard result.featureValue(for:"output")?.imageBufferValue != nil else { throw failure("missing image") } }
                    let end = ContinuousClock.now
                    if step >= 10 { predictions[index].append(ms(inferred-start)); packing[index].append(ms(end-inferred)) }
                }
            }
        }
    }
    var results: [String: Any] = [:]
    for index in 0..<3 {
        let total = zip(predictions[index],packing[index]).map(+)
        var row: [String: Any] = ["prediction_samples_ms":predictions[index],"packing_samples_ms":packing[index]]
        if correct {
            row["mean_ms"] = total.reduce(0,+)/Double(count)
            row["p95_ms"] = total.sorted()[Int(Double(count-1)*0.95)]
        }
        if index > 0 { row["storage"] = packers[index-1].storage; row["transfer_modes"] = packers[index-1].transferModes.sorted() }
        results[labels[index]] = row
    }
    let report: [String: Any] = ["complete":true,"correctness_pass":correct,"accuracy":accuracy,"results":results,
        "input":[width,height],"output":[width*4,height*4],"samples":correct ? count : 0,"warmup":10,
        "device":packers[0].device.name,"os":ProcessInfo.processInfo.operatingSystemVersionString,
        "purpose":"Same 4x reconstruction; Core ML image output versus tensor output plus Metal RGB8 packing",
        "limitations":"Synthetic four-frame RGB check and native boundary timing; no native NV12/detail, browser, or quality promotion claim"]
    try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]).write(to:reportURL)
    print("tensor-output-bench complete; correctness=\(correct)")
}

do { try main() } catch { fputs("\(error)\n",stderr); exit(1) }
