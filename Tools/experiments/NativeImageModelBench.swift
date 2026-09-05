import CoreML
import CoreVideo
import CryptoKit
import Foundation

func failure(_ text: String) -> NSError {
    NSError(domain: "NativeImageModelBench", code: 1, userInfo: [NSLocalizedDescriptionKey: text])
}
func milliseconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds)*1000 + Double(duration.components.attoseconds)/1e15
}
func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

@main struct NativeImageModelBench {
    static func main() throws {
        let args = CommandLine.arguments
        guard args.count == 4, let count = Int(args[3]), count >= 20 else {
            throw failure("usage: NativeImageModelBench MODEL_DIRECTORY REPORT COUNT>=20")
        }
        let directory = URL(fileURLWithPath: args[1]), reportURL = URL(fileURLWithPath: args[2])
        guard !FileManager.default.fileExists(atPath: reportURL.path) else { throw failure("fresh report required") }
        let labels = ["shipping4x", "direct2x_trained", "direct2x_area"]
        var compiled: [URL] = []
        defer { for url in compiled { try? FileManager.default.removeItem(at: url) } }
        let config = MLModelConfiguration(); config.computeUnits = .cpuAndGPU
        var rows: [[String: Any]] = []
        for (width, height) in [(640,360), (1280,720)] {
            let models = try labels.map { label in
                let package = directory.appendingPathComponent("\(label)_\(width)x\(height).mlpackage")
                let url = try MLModel.compileModel(at: package); compiled.append(url)
                return try MLModel(contentsOf: url, configuration: config)
            }
            var providers: [MLFeatureProvider] = []
            var inputHashes: [String] = []
            for index in 0..<4 {
                var created: CVPixelBuffer?
                guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                    [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &created) == kCVReturnSuccess,
                    let buffer = created else { throw failure("input allocation failed") }
                CVPixelBufferLockBaseAddress(buffer, [])
                let stride = CVPixelBufferGetBytesPerRow(buffer)
                let pixels = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
                var random = UInt32(20260905 + index)
                var bytes = Data(capacity: width*height*4)
                for y in 0..<height {
                    for x in 0..<width {
                        let at = y*stride+x*4
                        for c in 0..<3 { random = random &* 1664525 &+ 1013904223; pixels[at+c] = UInt8(truncatingIfNeeded: random >> 16) }
                        pixels[at+3] = 255
                    }
                    bytes.append(pixels+y*stride, count: width*4)
                }
                CVPixelBufferUnlockBaseAddress(buffer, [])
                inputHashes.append(digest(bytes))
                providers.append(try MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(pixelBuffer: buffer)]))
            }
            var samples = Array(repeating: [Double](), count: models.count)
            for step in 0..<(count+10) {
                try autoreleasepool {
                    for index in (step%2 == 0 ? Array(models.indices) : Array(models.indices.reversed())) {
                        let started = ContinuousClock.now
                        let result = try models[index].prediction(from: providers[step%4])
                        guard let output = result.featureValue(for: "output")?.imageBufferValue else { throw failure("image output required") }
                        let ended = ContinuousClock.now
                        let scale = index == 0 ? 4 : 2
                        guard CVPixelBufferGetWidth(output) == width*scale, CVPixelBufferGetHeight(output) == height*scale else {
                            throw failure("output geometry differs")
                        }
                        if step >= 10 { samples[index].append(milliseconds(ended-started)) }
                    }
                }
            }
            var timings: [String: Any] = [:]
            for index in labels.indices {
                let values = samples[index]
                timings[labels[index]] = ["mean_ms": values.reduce(0,+)/Double(values.count),
                    "p95_ms": values.sorted()[Int(ceil(Double(values.count)*0.95))-1], "samples_ms": values]
            }
            rows.append(["input_size": "\(width)x\(height)", "compute_units": "CPU_AND_GPU", "timings": timings,
                         "input_hashes_bgra": inputHashes])
            print("\(width)x\(height): \(timings.mapValues { ($0 as! [String:Any])["mean_ms"]! })")
        }
        var files: [String: String] = [:]
        if let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey]) {
            for case let url as URL in enumerator where (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true && url.path.contains(".mlpackage/") {
                files[String(url.path.dropFirst(directory.standardizedFileURL.path.count+1))] = digest(try Data(contentsOf: url))
            }
        }
        let report: [String: Any] = ["complete": true, "rows": rows, "samples": count, "warmup": 10,
            "model_files": files, "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "scope": "Swift Core ML prediction plus image-buffer access; no export work, Python/PIL, native postprocessing or browser timing"]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted,.sortedKeys]).write(to: reportURL)
    }
}
