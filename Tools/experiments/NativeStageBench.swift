// Executes the actual embedded production kernels for Torch parity fixtures.
// Usage: NativeStageBench DetailEnhancer.swift fixture-directory width height
import Foundation
import Metal

let args = CommandLine.arguments
guard (args.count == 5 || args.count == 6), let width = Int(args[3]), let height = Int(args[4]),
      let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    fatalError("source, fixture directory, width, height and Metal device required")
}
let motionPolicy = args.count == 6 ? UInt32(args[5])! : 0
guard motionPolicy <= 2 else { fatalError("motion policy must be 0, 1 or 2") }
let source = try String(contentsOfFile: args[1], encoding: .utf8)
let start = source.range(of: "#include <metal_stdlib>")!.lowerBound
let end = source.range(of: "\"\"\"", range: start..<source.endIndex)!.lowerBound
let options = MTLCompileOptions()
options.mathMode = .safe
let library = try device.makeLibrary(source: String(source[start..<end]), options: options)
let folder = URL(fileURLWithPath: args[2])

func texture(_ name: String?, _ channels: Int = 1, _ w: Int = width, _ h: Int = height) throws -> MTLTexture {
    let format: MTLPixelFormat = channels == 1 ? .r32Float : channels == 2 ? .rg32Float : .rgba32Float
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h, mipmapped: false)
    descriptor.usage = [.shaderRead, .shaderWrite]
    descriptor.storageMode = .shared
    let texture = device.makeTexture(descriptor: descriptor)!
    if let name {
        let bytes = try Data(contentsOf: folder.appendingPathComponent(name + ".bin"))
        guard bytes.count == w * h * channels * 4 else { fatalError("fixture size mismatch: \(name)") }
        bytes.withUnsafeBytes { buffer in
            texture.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                            withBytes: buffer.baseAddress!, bytesPerRow: w * channels * 4)
        }
    }
    return texture
}

func execute(_ name: String, _ textures: [MTLTexture], _ params: [Float], _ output: MTLTexture,
             buffers: [Int: [UInt32]] = [:]) throws {
    let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    for (i, texture) in textures.enumerated() { encoder.setTexture(texture, index: i) }
    params.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 0) }
    for (index, values) in buffers {
        values.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: index) }
    }
    encoder.dispatchThreads(MTLSize(width: output.width, height: output.height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else { throw command.error! }
}

func save(_ texture: MTLTexture, _ name: String, _ channels: Int = 1) throws {
    var bytes = Data(count: texture.width * texture.height * channels * 4)
    bytes.withUnsafeMutableBytes { buffer in
        texture.getBytes(buffer.baseAddress!, bytesPerRow: texture.width * channels * 4,
                         from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0)
    }
    try bytes.write(to: folder.appendingPathComponent(name + ".bin"))
}

let raw = try texture("current"), previous = try texture("previous"), history = try texture("history")
let clean = try texture(nil), field = try texture(nil, 4, (width + 7) / 8, (height + 7) / 8)
try execute("deband_plane", [raw, clean], [0.008, 16, 2, 3, 0.005], clean)
try save(clean, "metal-deband")
try execute("motion_blocks", [raw, previous, field], [1], field, buffers: [1: [motionPolicy]])
try save(field, "metal-motion", 4)
let output = try texture(nil), historyOut = try texture(nil)
try execute("taa_luma", [clean, history, output, historyOut, field, previous, raw], [0.45, 0.5, 1.25, 1, 1], output)
try save(output, "metal-taa")
let sharpened = try texture(nil), graded = try texture(nil)
// DetailParams starts with int radius, followed by eight float fields.
try execute("detail_enhance_luma", [raw, sharpened],
            [Float(bitPattern: 2), 0.2, 0, 0, 0.3, 0, 0.004, 0.030, 0], sharpened)
try save(sharpened, "metal-sharpen")
try execute("grade_luma", [sharpened, graded], [0.02, 0.99, 0.2, 0.01], graded,
            buffers: [1: [0, 8192, 0, 100], 2: [0]])
try save(graded, "metal-grade")
print("Production source and output kernels completed")
