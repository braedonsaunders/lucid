//
//  PipelineTiming.swift
//  Lucid
//
//  Native enhancement-stage time: preprocess → reconstruction → detail.
//  Used to A/B models or compute placement without the browser bridge.
//  Isolated Core ML benches omit the Metal stages that share the GPU.
//
//    Lucid --pipeline-ms <input.mp4> [count]
//    LUCID_COMPUTE_UNITS=gpu Lucid --pipeline-ms <input.mp4> 60
//
//  An exact-size SPAN_x4_ch32u_<w>x<h> package in the app bundle is loaded
//  even when that size is missing from the variants table or over budget.
//

import AVFoundation
import CoreMedia
import CoreVideo
import CoreML
import CryptoKit
import Foundation

/// Compare different reconstruction scales at the same delivered NV12 size.
/// No model selection or shipping enhancement behavior is changed by this probe.
@available(macOS 15.0, *)
enum PresentedNativeTiming {
    static func run() {
        do {
            let args = CommandLine.arguments
            guard let i = args.firstIndex(of: "--presented-native-ms"), args.count > i + 3 else {
                throw failure("usage: --presented-native-ms SHIPPING_4X DIRECT_2X REPORT [COUNT]")
            }
            try measure(urls: [URL(fileURLWithPath: args[i + 1]), URL(fileURLWithPath: args[i + 2])],
                        reportURL: URL(fileURLWithPath: args[i + 3]),
                        count: args.count > i + 4 ? max(20, Int(args[i + 4]) ?? 60) : 60)
            exit(0)
        } catch { print("presented-native-ms failed: \(error)"); exit(1) }
    }

    private static func failure(_ message: String) -> NSError {
        NSError(domain: "presented-native-ms", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private static func measure(urls: [URL], reportURL: URL, count: Int) throws {
        var temporary: [URL] = []
        defer { for url in temporary { try? FileManager.default.removeItem(at: url) } }
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuAndGPU
        let models = try urls.map { url in
            let compiled: URL
            if url.pathExtension == "mlmodelc" { compiled = url }
            else { compiled = try MLModel.compileModel(at: url); temporary.append(compiled) }
            return try MLModel(contentsOf: compiled, configuration: configuration)
        }
        guard let input = models[0].modelDescription.inputDescriptionsByName["input"]?.imageConstraint,
              let other = models[1].modelDescription.inputDescriptionsByName["input"]?.imageConstraint,
              input.pixelsWide == other.pixelsWide, input.pixelsHigh == other.pixelsHigh,
              input.pixelFormatType == kCVPixelFormatType_32BGRA,
              other.pixelFormatType == input.pixelFormatType else { throw failure("matching BGRA inputs required") }
        let width = input.pixelsWide, height = input.pixelsHigh
        let color = VideoColorInfo(primaries: "bt709", transfer: "iec61966-2-1", matrix: "rgb", fullRange: true)
        var frames: [CVPixelBuffer] = []
        for frame in 0..<4 {
            var buffer: CVPixelBuffer?
            guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                [kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary,
                &buffer) == kCVReturnSuccess, let buffer else { throw failure("input allocation failed") }
            CVPixelBufferLockBaseAddress(buffer, [])
            let bytes = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
            for y in 0..<height { for x in 0..<width {
                let p = y * CVPixelBufferGetBytesPerRow(buffer) + x * 4
                bytes[p] = UInt8((x * 13 + y * 7 + frame * 11) % 256)
                bytes[p + 1] = UInt8((x * 3 + y * 17 + frame * 5) % 256)
                bytes[p + 2] = UInt8((x * 23 + y * 3 + frame * 7) % 256)
                bytes[p + 3] = 255
            }}
            CVPixelBufferUnlockBaseAddress(buffer, []); color.apply(to: buffer); frames.append(buffer)
        }
        let senders = [EnhancedFrameSender(), EnhancedFrameSender()]
        for sender in senders { sender.maximumWidth = width * 2 }
        var graph = [[Double](), [Double]()], packing = [[Double](), [Double]()]
        var packetSizes = [0, 0], packetHashes = ["", ""]
        for step in 0..<(count + 10) {
            let provider = try MLDictionaryFeatureProvider(dictionary:
                ["input": MLFeatureValue(pixelBuffer: frames[step % 4])])
            for index in (step % 2 == 0 ? [0, 1] : [1, 0]) {
                let start = ContinuousClock.now
                let result = try models[index].prediction(from: provider)
                guard let image = result.featureValue(for: "output")?.imageBufferValue,
                      CVPixelBufferGetWidth(image) == width * (index == 0 ? 4 : 2),
                      CVPixelBufferGetHeight(image) == height * (index == 0 ? 4 : 2) else {
                    throw failure("unexpected reconstruction scale")
                }
                let inferred = ContinuousClock.now
                color.apply(to: image)
                guard let packet = senders[index].packet(for: image, sequence: step, session: "presented-bench") else {
                    throw failure("NV12 packet creation failed")
                }
                let end = ContinuousClock.now
                packetSizes[index] = packet.count
                packetHashes[index] = SHA256.hash(data: packet).description
                if step >= 10 {
                    graph[index].append((inferred - start).milliseconds)
                    packing[index].append((end - inferred).milliseconds)
                }
            }
            guard packetSizes[0] == packetSizes[1] else { throw failure("delivery sizes differ") }
        }
        var rows: [[String: Any]] = []
        for index in 0..<2 {
            let total = zip(graph[index], packing[index]).map(+)
            rows.append(["variant": index == 0 ? "shipping4x" : "direct2x",
                "graph_mean_ms": graph[index].reduce(0, +) / Double(count),
                "packet_mean_ms": packing[index].reduce(0, +) / Double(count),
                "total_mean_ms": total.reduce(0, +) / Double(count),
                "total_p95_ms": total.sorted()[Int(Double(count - 1) * 0.95)],
                "graph_samples_ms": graph[index], "packet_samples_ms": packing[index],
                "packet_bytes": packetSizes[index], "last_packet_sha256": packetHashes[index]])
        }
        let report: [String: Any] = ["purpose": "Native graph plus real NV12 serialization at common delivered size",
            "input": [width, height], "delivered_output": [width * 2, height * 2], "rows": rows,
            "samples": count, "warmup": 10, "compute_units": "CPU_AND_GPU",
            "limitations": ["Synthetic BGRA; capture/decode/input conversion, detail postprocessing, network and rendering excluded",
                "Shipping 4x uses sender's native downscaler; quality is not equivalent to PIL bicubic development adapter",
                "Different reconstruction filters; numerical equivalence between variants is not claimed"]]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: reportURL, options: .atomic)
        print("presented-native-ms completed: \(reportURL.path)")
    }
}

/// Compare the candidate's two state representations through native Core ML.
/// Runs before app state, and also times complete full-resolution NV12 packets.
@available(macOS 15.0, *)
enum CausalNativeTiming {
    static func run() {
        do {
            let args = CommandLine.arguments
            guard let i = args.firstIndex(of: "--causal-native-ms"), args.count > i + 3 else {
                throw failure("usage: --causal-native-ms EXPLICIT_MODEL STATEFUL_MODEL REPORT [COUNT]")
            }
            let count = args.count > i + 4 ? max(10, Int(args[i + 4]) ?? 60) : 60
            try measure(explicitURL: URL(fileURLWithPath: args[i + 1]),
                        statefulURL: URL(fileURLWithPath: args[i + 2]),
                        reportURL: URL(fileURLWithPath: args[i + 3]), count: count)
            exit(0)
        } catch { print("causal-native-ms failed: \(error)"); exit(1) }
    }

    private static func failure(_ message: String) -> NSError {
        NSError(domain: "causal-native-ms", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private static func measure(explicitURL: URL, statefulURL: URL, reportURL: URL, count: Int) throws {
        var temporary: [URL] = []
        defer { for url in temporary { try? FileManager.default.removeItem(at: url) } }
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuAndGPU
        func load(_ url: URL) throws -> MLModel {
            let compiled: URL
            if url.pathExtension == "mlmodelc" { compiled = url }
            else { compiled = try MLModel.compileModel(at: url); temporary.append(compiled) }
            return try MLModel(contentsOf: compiled, configuration: configuration)
        }
        let models = try [load(explicitURL), load(statefulURL)]
        guard let input = models[0].modelDescription.inputDescriptionsByName["input"]?.imageConstraint,
              let other = models[1].modelDescription.inputDescriptionsByName["input"]?.imageConstraint,
              input.pixelsWide == other.pixelsWide, input.pixelsHigh == other.pixelsHigh,
              let historyShape = models[0].modelDescription.inputDescriptionsByName["history_features"]?.multiArrayConstraint?.shape,
              models[1].modelDescription.stateDescriptionsByName["history"] != nil else {
            throw failure("incompatible candidate model interfaces")
        }
        let width = input.pixelsWide, height = input.pixelsHigh
        let color = VideoColorInfo(primaries: "bt709", transfer: "iec61966-2-1", matrix: "rgb", fullRange: true)
        var frames: [CVPixelBuffer] = []
        for frame in 0..<4 {
            var buffer: CVPixelBuffer?
            guard CVPixelBufferCreate(nil, width, height, input.pixelFormatType,
                [kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary,
                &buffer) == kCVReturnSuccess, let buffer,
                input.pixelFormatType == kCVPixelFormatType_32BGRA else { throw failure("BGRA input required") }
            CVPixelBufferLockBaseAddress(buffer, [])
            let bytes = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRow(buffer)
            for y in 0..<height { for x in 0..<width {
                let p = y * stride + x * 4
                bytes[p] = UInt8((x * 13 + y * 7 + frame * 11) % 256)
                bytes[p + 1] = UInt8((x * 3 + y * 17 + frame * 5) % 256)
                bytes[p + 2] = UInt8((x * 23 + y * 3 + frame * 7) % 256)
                bytes[p + 3] = 255
            }}
            CVPixelBufferUnlockBaseAddress(buffer, []); color.apply(to: buffer); frames.append(buffer)
        }
        let valid = try MLMultiArray(shape: [1,1,1,1], dataType: .float32)
        var reportRows: [[String: Any]] = []
        var parityMaximum = 0
        for deliver in [false, true] {
            var history = try MLMultiArray(shape: historyShape, dataType: .float32)
            memset(history.dataPointer, 0, history.count * MemoryLayout<Float>.size)
            let state = models[1].makeState()
            let senders = [EnhancedFrameSender(), EnhancedFrameSender()]
            for sender in senders { sender.maximumWidth = width * 2 }
            var timings = [[Double](), [Double]()], packing = [[Double](), [Double]()]
            var packetBytes = 0
            var packetDigests = ["", ""]
            for step in 0..<(count + 10) {
                valid[0] = step == 0 ? 0 : 1
                var outputs: [Int: CVPixelBuffer] = [:]
                for index in (step % 2 == 0 ? [0, 1] : [1, 0]) {
                    var features: [String: MLFeatureValue] = ["input": MLFeatureValue(pixelBuffer: frames[step % 4]),
                                                             "valid": MLFeatureValue(multiArray: valid)]
                    if index == 0 { features["history_features"] = MLFeatureValue(multiArray: history) }
                    let provider = try MLDictionaryFeatureProvider(dictionary: features)
                    let start = ContinuousClock.now
                    let result = index == 0 ? try models[index].prediction(from: provider)
                        : try models[index].prediction(from: provider, using: state)
                    if index == 0 {
                        guard let next = result.featureValue(for: "next_state")?.multiArrayValue else { throw failure("missing history") }
                        history = next // Native object reuse: do not manufacture a Python-style copy.
                    }
                    guard let image = result.featureValue(for: "output")?.imageBufferValue else { throw failure("missing output") }
                    let inferred = ContinuousClock.now
                    var serialized: Data?
                    if deliver {
                        color.apply(to: image)
                        guard let packet = senders[index].packet(for: image, sequence: step, session: "native-bench") else {
                            throw failure("NV12 packet failed")
                        }
                        packetBytes = packet.count
                        serialized = packet
                    }
                    let end = ContinuousClock.now
                    // Consume the complete payload outside timing so an
                    // optimizer cannot turn packet creation into a size query.
                    if let serialized { packetDigests[index] = SHA256.hash(data: serialized).description }
                    outputs[index] = image
                    if step >= 10 {
                        timings[index].append((inferred - start).milliseconds)
                        packing[index].append((end - inferred).milliseconds)
                    }
                }
                if [0, 10, count + 9].contains(step), let a = outputs[0], let b = outputs[1] {
                    guard CVPixelBufferGetPixelFormatType(a) == kCVPixelFormatType_32BGRA,
                          CVPixelBufferGetWidth(a) == CVPixelBufferGetWidth(b),
                          CVPixelBufferGetHeight(a) == CVPixelBufferGetHeight(b) else { throw failure("incompatible output") }
                    CVPixelBufferLockBaseAddress(a, .readOnly); CVPixelBufferLockBaseAddress(b, .readOnly)
                    let pa = CVPixelBufferGetBaseAddress(a)!.assumingMemoryBound(to: UInt8.self)
                    let pb = CVPixelBufferGetBaseAddress(b)!.assumingMemoryBound(to: UInt8.self)
                    for y in 0..<CVPixelBufferGetHeight(a) { for x in 0..<CVPixelBufferGetWidth(a) { for c in 0..<3 {
                        parityMaximum = max(parityMaximum, abs(Int(pa[y * CVPixelBufferGetBytesPerRow(a) + x * 4 + c])
                            - Int(pb[y * CVPixelBufferGetBytesPerRow(b) + x * 4 + c])))
                    }}}
                    CVPixelBufferUnlockBaseAddress(a, .readOnly); CVPixelBufferUnlockBaseAddress(b, .readOnly)
                    guard parityMaximum <= 2 else { throw failure("state representation changed pixels: \(parityMaximum)") }
                }
            }
            for index in 0..<2 {
                let totals = zip(timings[index], packing[index]).map(+)
                let sorted = totals.sorted()
                reportRows.append(["state": index == 0 ? "explicit_native_object" : "MLState", "nv12_packet": deliver,
                    "graph_mean_ms": timings[index].reduce(0, +) / Double(count),
                    "packet_mean_ms": packing[index].reduce(0, +) / Double(count),
                    "total_mean_ms": totals.reduce(0, +) / Double(count),
                    "total_p95_ms": sorted[Int(Double(count - 1) * 0.95)],
                    "graph_samples_ms": timings[index], "packet_samples_ms": packing[index], "packet_bytes": packetBytes,
                    "last_packet_digest": packetDigests[index]])
            }
        }
        let report: [String: Any] = ["purpose": "Native Swift Core ML comparison; optional full-resolution NV12 serialization",
            "input": [width, height], "output": [width * 2, height * 2], "samples": count, "warmup": 10,
            "maximum_pixel_difference": parityMaximum, "rows": reportRows,
            "limitations": ["Synthetic BGRA input; capture/decode/input conversion and browser transport/render excluded",
                            "Explicit history reuses MLMultiArray directly, unlike the Python benchmark"]]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: reportURL, options: .atomic)
        print("causal-native-ms completed: \(reportURL.path), pixel difference \(parityMaximum)")
    }
}

/// Offline decoder boundary for codec-independent quality evaluation. RGB conversion
/// and all enhancement/delivery stages remain the app's normal implementations.
final class DiagnosticNV12Frames {
    struct Manifest: Codable {
        let format: String
        let width: Int, height: Int, frames: Int
        let fpsNumerator: Int32, fpsDenominator: Int32
        let data: String, sha256: String
        let colorSpace: VideoColorInfo
        let chromaLocation: String
        let spatialStride: Int
        let exportWarmup: Bool
    }
    let manifest: Manifest
    private let file: FileHandle
    private let pool: CVPixelBufferPool
    private var position = 0

    private static func failure(_ message: String) -> NSError {
        NSError(domain: "pipeline-nv12", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    init(manifestURL: URL) throws {
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        guard manifest.format == "NV12", manifest.width > 0, manifest.width <= 8192,
              manifest.height > 0, manifest.height <= 4320,
              manifest.width % 2 == 0, manifest.height % 2 == 0,
              manifest.frames > 0, manifest.frames <= 1_000_000,
              manifest.fpsNumerator > 0, manifest.fpsDenominator > 0,
              manifest.spatialStride > 0, manifest.spatialStride <= manifest.frames,
              manifest.colorSpace == .rec709, manifest.chromaLocation == "left",
              !manifest.data.isEmpty, manifest.data == (manifest.data as NSString).lastPathComponent
        else { throw Self.failure("invalid NV12 geometry, cadence, filename or explicit Rec.709/left-chroma contract") }
        let url = manifestURL.deletingLastPathComponent().appendingPathComponent(manifest.data)
        let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize
        guard size == manifest.width * manifest.height * 3 / 2 * manifest.frames else {
            throw Self.failure("raw stream byte count differs from declared frame count")
        }
        let handle = try FileHandle(forReadingFrom: url)
        var keepHandle = false
        defer { if !keepHandle { try? handle.close() } }
        var hash = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty { hash.update(data: chunk) }
        let actual = hash.finalize().map { String(format: "%02x", $0) }.joined()
        guard actual == manifest.sha256 else { throw Self.failure("raw stream SHA256 mismatch") }
        try handle.seek(toOffset: 0)
        var created: CVPixelBufferPool?
        let attributes: [String: Any] = [kCVPixelBufferWidthKey as String: manifest.width,
            kCVPixelBufferHeightKey as String: manifest.height,
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferMetalCompatibilityKey as String: true]
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault,
            [kCVPixelBufferPoolMinimumBufferCountKey as String: 4] as CFDictionary,
            attributes as CFDictionary, &created) == kCVReturnSuccess, let created else {
            throw Self.failure("NV12 pool allocation failed")
        }
        self.manifest = manifest; file = handle; pool = created; keepHandle = true
    }

    deinit { try? file.close() }

    func next() throws -> (CVPixelBuffer, CMTime)? {
        guard position < manifest.frames else { return nil }
        let width = manifest.width, height = manifest.height, count = width * height * 3 / 2
        var bytes = Data()
        while bytes.count < count {
            guard let chunk = try file.read(upToCount: count - bytes.count), !chunk.isEmpty else {
                throw Self.failure("raw stream truncated after validation")
            }
            bytes.append(chunk)
        }
        var created: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &created) == kCVReturnSuccess,
              let buffer = created else { throw Self.failure("NV12 frame allocation failed") }
        CVBufferRemoveAllAttachments(buffer)
        CVPixelBufferLockBaseAddress(buffer, [])
        bytes.withUnsafeBytes { source in
            for plane in 0..<2 {
                let destination = CVPixelBufferGetBaseAddressOfPlane(buffer, plane)!
                let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane)
                let offset = plane == 0 ? 0 : width * height
                for row in 0..<(plane == 0 ? height : height / 2) {
                    memcpy(destination.advanced(by: row * stride), source.baseAddress!.advanced(by: offset + row * width), width)
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        manifest.colorSpace.apply(to: buffer)
        CVBufferSetAttachment(buffer, kCVImageBufferChromaLocationTopFieldKey, kCVImageBufferChromaLocation_Left, .shouldPropagate)
        CVBufferSetAttachment(buffer, kCVImageBufferChromaLocationBottomFieldKey, kCVImageBufferChromaLocation_Left, .shouldPropagate)
        let time = CMTime(value: Int64(position) * Int64(manifest.fpsDenominator), timescale: manifest.fpsNumerator)
        position += 1
        return (buffer, time)
    }
}

@available(macOS 26.0, *)
enum PipelineTiming {
    static func run() async {
        setvbuf(stdout, nil, _IOLBF, 0)
        let args = CommandLine.arguments
        guard let index = args.firstIndex(of: "--pipeline-ms"), args.count > index + 1 else {
            print("usage: Lucid --pipeline-ms <input.mp4> [count]")
            exit(2)
        }
        let path = args[index + 1]
        let count = args.count > index + 2 ? max(Int(args[index + 2]) ?? 60, 1) : 60
        do {
            try await measure(path: path, count: count)
        } catch {
            print("pipeline-ms failed: \(error)")
            exit(1)
        }
        exit(0)
    }

    private static func measure(path: String, count: Int) async throws {
        let url = URL(fileURLWithPath: path)
        let raw = url.pathExtension == "json" ? try DiagnosticNV12Frames(manifestURL: url) : nil
        var reader: AVAssetReader?
        var output: AVAssetReaderTrackOutput?
        defer { reader?.cancelReading() }
        if raw == nil {
            let asset = AVURLAsset(url: url)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                throw NSError(domain: "pipeline-ms", code: 1, userInfo: [NSLocalizedDescriptionKey: "no video track"])
            }
            let assetReader = try AVAssetReader(asset: asset)
            let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
                kCVPixelBufferMetalCompatibilityKey as String: true])
            reader = assetReader; output = trackOutput
            assetReader.add(trackOutput)
            guard assetReader.startReading() else {
                throw assetReader.error ?? NSError(domain: "pipeline-ms", code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "could not start decoding"])
            }
        }
        func nextFrame() throws -> (CVPixelBuffer, CMTime)? {
            if let raw { return try raw.next() }
            guard let sample = output?.copyNextSampleBuffer(), let frame = sample.imageBuffer else { return nil }
            return (frame, CMSampleBufferGetPresentationTimeStamp(sample))
        }

        let compositor = try MetalTileCompositor()
        let t = EnhancementSession.Tuning.load()
        TiledVideoToolboxUpscaler.chromaSitingLeft = t.stageSiting > 0.5
        LearnedUpscaler.flipCycle = t.flipCycle > 0.5
        let detail = try DetailEnhancer(device: compositor.device, settings: t.detailSettings(radius: 4))

        var learned: LearnedUpscaler?
        var preprocess: [Double] = []
        var upscale: [Double] = []
        var finish: [Double] = []
        var total: [Double] = []
        var seen = 0
        let warmup = 8
        // Optional diagnostic export uses the actual browser packet writer.
        // Disk writes and packet packing are outside enhancement-stage timing.
        let packetDirectory = ProcessInfo.processInfo.environment["LUCID_PIPELINE_PACKETS"]
            .map { URL(fileURLWithPath: $0, isDirectory: true) }
        if let packetDirectory {
            guard !FileManager.default.fileExists(atPath: packetDirectory.path) else {
                throw NSError(domain: "pipeline-ms", code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "packet export requires a fresh directory"])
            }
            try FileManager.default.createDirectory(at: packetDirectory, withIntermediateDirectories: true)
            try JSONEncoder().encode(t).write(to: packetDirectory.appendingPathComponent("tuning.json"), options: .atomic)
        }
        let sender = EnhancedFrameSender()

        print("pipeline-ms compute=\(LearnedUpscaler.computeUnitsLabel) count=\(count) warmup=\(warmup)")
        while seen < warmup + count, let (frame, timestamp) = try nextFrame() {
            let width = CVPixelBufferGetWidth(frame)
            let height = CVPixelBufferGetHeight(frame)
            if learned == nil {
                learned = try LearnedUpscaler(width: width, height: height)
                detail.settings = t.detailSettings(radius: learned!.scale)
                // Diagnostic ablation: preserve the user's nominal gain when a
                // presentation transform changes output scale without retraining.
                if ProcessInfo.processInfo.environment["LUCID_PIPELINE_PRESERVE_GAIN"] == "1" {
                    detail.settings.referenceRadius = learned!.scale
                }
                print("pipeline-ms input \(width)x\(height) → model \(learned!.inputWidth)x\(learned!.inputHeight) → output \(learned!.outputWidth)x\(learned!.outputHeight)")
                print("pipeline-ms detail radius=\(detail.settings.radius)")
                print("pipeline-ms detail referenceRadius=\(detail.settings.referenceRadius)")
                print("pipeline-ms detail motionPolicy=\(DetailEnhancer.motionPolicy)")
                let incoming = CVBufferCopyAttachment(frame, kCVImageBufferChromaLocationTopFieldKey, nil)
                    .map { "\($0)" } ?? "nil"
                print("pipeline-ms chroma incoming=\(incoming)")
            }
            let frame = TiledVideoToolboxUpscaler.prepareSource(frame)
            if seen == 0 {
                let applied = CVBufferCopyAttachment(frame, kCVImageBufferChromaLocationTopFieldKey, nil)
                    .map { "\($0)" } ?? "nil"
                print("pipeline-ms chroma applied=\(applied) flag=\(TiledVideoToolboxUpscaler.chromaSitingLeft ? "left" : "center") resampled=\(TiledVideoToolboxUpscaler.chromaSitingLeft ? "none" : "center")")
            }
            let started = ContinuousClock.now
            let t0 = ContinuousClock.now
            let cleaned = try detail.preprocess(frame, timestamp: timestamp)
            let t1 = ContinuousClock.now
            let reconstructed = try learned!.upscale(cleaned)
            let t2 = ContinuousClock.now
            let enhanced = try detail.process(reconstructed)
            let t3 = ContinuousClock.now
            seen += 1
            if seen > warmup {
                preprocess.append((t1 - t0).milliseconds)
                upscale.append((t2 - t1).milliseconds)
                finish.append((t3 - t2).milliseconds)
                total.append((t3 - started).milliseconds)
            }
            let exportFrame = seen > warmup || raw?.manifest.exportWarmup == true
            if let packetDirectory, exportFrame, (seen - 1) % (raw?.manifest.spatialStride ?? 4) == 0 {
                sender.maximumWidth = width * 2
                guard let packet = sender.packet(for: enhanced, sequence: seen - 1, session: "pipeline-quality",
                    sourceTimestamp: timestamp.seconds) else {
                    throw NSError(domain: "pipeline-ms", code: 5,
                        userInfo: [NSLocalizedDescriptionKey: "could not export delivered NV12 frame"])
                }
                try packet.write(to: packetDirectory.appendingPathComponent(String(format: "%08d.luce", seen - 1)), options: .atomic)
                let exportStages = raw == nil || ProcessInfo.processInfo.environment["LUCID_PIPELINE_STAGES"] == "1"
                for (stage, buffer) in exportStages ? [("source", frame), ("preprocessed", cleaned), ("reconstructed", reconstructed)] : [] {
                    guard let intermediate = sender.packet(for: buffer, sequence: seen - 1, session: "pipeline-quality",
                        sourceTimestamp: timestamp.seconds) else {
                        throw NSError(domain: "pipeline-ms", code: 5,
                            userInfo: [NSLocalizedDescriptionKey: "could not export \(stage) stage"])
                    }
                    try intermediate.write(to: packetDirectory.appendingPathComponent(
                        String(format: "%08d-", seen - 1) + stage + ".luce"), options: .atomic)
                }
            }
        }

        guard total.count == count else {
            throw reader?.error ?? NSError(domain: "pipeline-ms", code: 3,
                userInfo: [NSLocalizedDescriptionKey:
                    "incomplete sample: measured \(total.count) of \(count) frames after \(warmup) warmups"])
        }

        func mean(_ values: [Double]) -> Double {
            guard !values.isEmpty else { return 0 }
            return values.reduce(0, +) / Double(values.count)
        }
        func percentile(_ values: [Double], _ p: Double) -> Double {
            guard !values.isEmpty else { return 0 }
            let sorted = values.sorted()
            let i = min(sorted.count - 1, max(0, Int((Double(sorted.count - 1) * p).rounded())))
            return sorted[i]
        }

        print(String(
            format: "pipeline-ms compute=%@ n=%d  total mean %.2f  p50 %.2f  p95 %.2f  max %.2f  |  preprocess %.2f  span %.2f  detail %.2f",
            LearnedUpscaler.computeUnitsLabel, total.count,
            mean(total), percentile(total, 0.50), percentile(total, 0.95), total.max() ?? 0,
            mean(preprocess), mean(upscale), mean(finish)
        ))
    }
}
