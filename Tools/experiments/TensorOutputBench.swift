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

func roundingDiagnostic(_ array: MLMultiArray, image: CVPixelBuffer) throws -> [String: Int] {
    let w = CVPixelBufferGetWidth(image), h = CVPixelBufferGetHeight(image)
    CVPixelBufferLockBaseAddress(image, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
    let pixels = CVPixelBufferGetBaseAddress(image)!.assumingMemoryBound(to: UInt8.self)
    let rowBytes = CVPixelBufferGetBytesPerRow(image)
    var halfUpMismatch = 0, nearestEvenMismatch = 0, halfWayDifferences = 0
    try array.withUnsafeMutableBytes { raw, strides in
        guard let base = raw.baseAddress else { throw failure("missing tensor storage") }
        for y in 0..<h { for x in 0..<w { for c in 0..<3 {
            let at = c*strides[1]+y*strides[2]+x*strides[3]
            let value = array.dataType == .float16 ? Float(base.assumingMemoryBound(to: Float16.self)[at]) : base.assumingMemoryBound(to: Float.self)[at]
            guard value.isFinite && value >= 0 && value <= 255 else { throw failure("unexpected tensor range") }
            let expected = Int(pixels[y*rowBytes+x*4+(2-c)])
            if Int(floor(value+0.5)) != expected {
                halfUpMismatch += 1
                if value-floor(value) == 0.5 { halfWayDifferences += 1 }
            }
            if Int(value.rounded(.toNearestOrEven)) != expected { nearestEvenMismatch += 1 }
        } } }
    }
    return ["channels_checked":w*h*3,"half_up_mismatches":halfUpMismatch,
            "nearest_even_mismatches":nearestEvenMismatch,"half_way_differences":halfWayDifferences]
}

@main
struct TensorOutputBench {
static func main() throws {
    let args = CommandLine.arguments
    guard [4,5].contains(args.count), let count = Int(args[3]), count >= 20 else {
        throw failure("usage: TensorOutputBench MODEL_DIRECTORY REPORT COUNT>=20 [tensor4x_fp32|tensor4x_fp16]")
    }
    let directory = URL(fileURLWithPath: args[1]), reportURL = URL(fileURLWithPath: args[2])
    guard !FileManager.default.fileExists(atPath: reportURL.path) else { throw failure("fresh report required") }
    if args.count == 5 && !["tensor4x_fp32","tensor4x_fp16"].contains(args[4]) { throw failure("unknown tensor variant") }
    let labels = args.count == 5 ? ["image4x",args[4]] : ["image4x", "tensor4x_fp32", "tensor4x_fp16"]
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
    let packers = try (1..<labels.count).map { _ in try CoreMLTensorImagePacker(width: width*4,height: height*4) }
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
    var rounding: [String: [[String: Int]]] = [:]
    for frame in frames {
        let provider = try MLDictionaryFeatureProvider(dictionary:["input":MLFeatureValue(pixelBuffer:frame)])
        let baseline = try models[0].prediction(from:provider).featureValue(for:"output")!.imageBufferValue!
        for index in 1..<labels.count {
            let array = try models[index].prediction(from:provider).featureValue(for:"output")!.multiArrayValue!
            let packed = try packers[index-1].pack(array)
            accuracy[labels[index],default:[]].append(try difference(baseline,packed))
            rounding[labels[index],default:[]].append(try roundingDiagnostic(array,image:baseline))
        }
    }
    var predictions = Array(repeating:[Double](),count:labels.count), packing = predictions
    let correct = accuracy.values.flatMap { $0 }.allSatisfy { $0["max_rgb"]! <= 3 && $0["mean_rgb"]! <= 0.6 }
    if correct {
        for step in 0..<(count+10) {
            try autoreleasepool {
                let provider = try MLDictionaryFeatureProvider(dictionary:["input":MLFeatureValue(pixelBuffer:frames[step%4])])
                for index in (step%2 == 0 ? Array(labels.indices) : Array(labels.indices.reversed())) {
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
    for index in labels.indices {
        let total = zip(predictions[index],packing[index]).map(+)
        var row: [String: Any] = ["prediction_samples_ms":predictions[index],"packing_samples_ms":packing[index]]
        if correct {
            row["mean_ms"] = total.reduce(0,+)/Double(count)
            row["p95_ms"] = total.sorted()[Int(Double(count-1)*0.95)]
        }
        if index > 0 { row["storage"] = packers[index-1].storage; row["transfer_modes"] = packers[index-1].transferModes.sorted() }
        results[labels[index]] = row
    }
    let report: [String: Any] = ["complete":true,"correctness_pass":correct,"accuracy":accuracy,"results":results,"rounding_diagnostics":rounding,
        "input":[width,height],"output":[width*4,height*4],"samples":correct ? count : 0,"warmup":10,
        "device":packers[0].device.name,"os":ProcessInfo.processInfo.operatingSystemVersionString,
        "purpose":"Same 4x reconstruction; Core ML image output versus tensor output plus Metal RGB8 packing",
        "limitations":"Synthetic four-frame RGB check and native boundary timing; no native NV12/detail, browser, or quality promotion claim"]
    try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]).write(to:reportURL)
    print("tensor-output-bench complete; correctness=\(correct)")
}

}
