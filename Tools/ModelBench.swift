//
//  ModelBench.swift
//
//  Measures what a Core ML model actually costs on this machine, and where it
//  runs. Published Core ML latencies in the super-resolution literature are not
//  portable - two papers disagree by 18x on the same model - so the only number
//  worth having is one measured here.
//
//  swiftc -O Tools/ModelBench.swift -o .build/modelbench
//  .build/modelbench Model/SPAN_x4_ch48_480x270.mlpackage
//

import CoreML
import Foundation

@available(macOS 14.0, *)
func describe(_ model: MLModel) -> (input: String, shape: [Int], output: String)? {
    guard let input = model.modelDescription.inputDescriptionsByName.first,
          let output = model.modelDescription.outputDescriptionsByName.first
    else { return nil }
    var shape: [Int] = []
    if let image = input.value.imageConstraint {
        shape = [1, 3, image.pixelsHigh, image.pixelsWide]
    } else if let array = input.value.multiArrayConstraint {
        shape = array.shape.map(\.intValue)
    }
    return (input.key, shape, output.key)
}

@available(macOS 14.0, *)
func makeInput(_ model: MLModel, name: String, shape: [Int]) throws -> MLFeatureProvider {
    guard let constraint = model.modelDescription.inputDescriptionsByName[name] else {
        throw NSError(domain: "bench", code: 1)
    }
    if let image = constraint.imageConstraint {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, image.pixelsWide, image.pixelsHigh,
                            image.pixelFormatType,
                            [kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]] as CFDictionary,
                            &buffer)
        guard let buffer else { throw NSError(domain: "bench", code: 2) }
        return try MLDictionaryFeatureProvider(dictionary: [name: MLFeatureValue(pixelBuffer: buffer)])
    }
    let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float16)
    return try MLDictionaryFeatureProvider(dictionary: [name: MLFeatureValue(multiArray: array)])
}

@available(macOS 14.0, *)
func bench(_ path: String, units: MLComputeUnits, label: String, iterations: Int = 40) {
    let url = URL(fileURLWithPath: path)
    do {
        let compiled = url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at: url)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        let model = try MLModel(contentsOf: compiled, configuration: configuration)
        guard let info = describe(model) else { print("  \(label): no input/output"); return }
        let input = try makeInput(model, name: info.input, shape: info.shape)

        // Warm up: the first call includes model load and graph specialisation.
        for _ in 0..<5 { _ = try model.prediction(from: input) }

        var samples: [Double] = []
        for _ in 0..<iterations {
            let started = CFAbsoluteTimeGetCurrent()
            _ = try model.prediction(from: input)
            samples.append((CFAbsoluteTimeGetCurrent() - started) * 1000)
        }
        samples.sort()
        let median = samples[samples.count / 2]
        let best = samples.first ?? 0
        print(String(format: "  %-22@ median %6.2f ms   best %6.2f ms", label as NSString, median, best))
    } catch {
        print("  \(label): failed — \(error.localizedDescription)")
    }
}

guard #available(macOS 14.0, *) else { fatalError("needs macOS 14+") }
let paths = Array(CommandLine.arguments.dropFirst())
guard !paths.isEmpty else {
    print("usage: modelbench <model.mlpackage> [more...]")
    exit(2)
}
for path in paths {
    let name = URL(fileURLWithPath: path).lastPathComponent
    print("\n\(name)")
    if let model = try? MLModel(contentsOf: (try? MLModel.compileModel(at: URL(fileURLWithPath: path))) ?? URL(fileURLWithPath: path)),
       let info = describe(model) {
        let pixels = info.shape.count >= 4 ? info.shape[2] * info.shape[3] : 0
        print("  input \(info.input) \(info.shape)  (\(pixels) px)  ->  \(info.output)")
    }
    // Comparing the two placements is the only way to tell whether the Neural
    // Engine actually took the graph: Core ML falls back silently.
    bench(path, units: .cpuAndNeuralEngine, label: "ANE + CPU")
    bench(path, units: .cpuAndGPU, label: "GPU + CPU")
    bench(path, units: .all, label: "all")
}
