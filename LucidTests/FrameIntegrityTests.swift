import CoreMedia
import CoreML
import CoreVideo
import CryptoKit
import Metal
import Testing
@testable import Lucid

struct TensorImagePackerTests {
    @Test func paddedFloatStoragePacksRGBAndClampsWithoutEscapingBorrow() throws {
        // Deliberately sub-page and padded: exercises the explicit-copy path.
        let storage = UnsafeMutableRawPointer.allocate(byteCount: 256, alignment: 4)
        storage.initializeMemory(as: UInt8.self, repeating: 0, count: 256)
        let array = try MLMultiArray(dataPointer: storage, shape: [1,3,2,3], dataType: .float32,
            strides: [48,16,5,1], deallocator: { $0.deallocate() })
        let levels: [Float] = [-5,0.49,0.51,127.4,254.7,300]
        for c in 0..<3 { for y in 0..<2 { for x in 0..<3 {
            storage.assumingMemoryBound(to: Float.self)[c*16+y*5+x] = levels[(y*3+x+c)%6]
        } } }
        let packer = try CoreMLTensorImagePacker(width: 3, height: 2)
        let output = try packer.pack(array)
        #expect(packer.transferModes == ["explicit copy"])
        CVPixelBufferLockBaseAddress(output, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(output, .readOnly) }
        let pixels = CVPixelBufferGetBaseAddress(output)!.assumingMemoryBound(to: UInt8.self)
        let expected = [0,0,1,127,255,255]
        for y in 0..<2 { for x in 0..<3 {
            let at = y*CVPixelBufferGetBytesPerRow(output)+x*4
            for c in 0..<3 { #expect(Int(pixels[at+2-c]) == expected[(y*3+x+c)%6]) }
            #expect(pixels[at+3] == 255)
        } }
    }

    @Test func rejectsWrongTensorGeometryAndType() throws {
        let packer = try CoreMLTensorImagePacker(width: 3, height: 2)
        let wrongShape = try MLMultiArray(shape: [1,3,3,2], dataType: .float32)
        let wrongType = try MLMultiArray(shape: [1,3,2,3], dataType: .double)
        #expect(throws: (any Error).self) { try packer.pack(wrongShape) }
        #expect(throws: (any Error).self) { try packer.pack(wrongType) }
    }
}

struct LearnedReconstructionGeometryTests {
    @Test func nominalDetailGainIsLimitedToTheVerifiedPresentationTransform() {
        var metadata = [
            "lucid.transformation": "quantized 4x-to-2x bicubic presentation",
            "lucid.checkpoint_sha256": "fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65"
        ]
        let reference = LearnedUpscaler.detailReferenceRadius(scale: 2, metadata: metadata)
        #expect(reference == 2)
        #expect(DetailEnhancer.gainNormalisation(radius: 2, reference: reference) == 1)
        #expect(LearnedUpscaler.detailReferenceRadius(scale: 4, metadata: metadata) == 4)
        #expect(LearnedUpscaler.detailReferenceRadius(scale: 2, metadata: [:]) == 4)
        metadata["lucid.checkpoint_sha256"] = "unverified"
        #expect(LearnedUpscaler.detailReferenceRadius(scale: 2, metadata: metadata) == 4)
        metadata["lucid.checkpoint_sha256"] = "fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65"
        metadata["lucid.transformation"] = "area phase folding"
        #expect(LearnedUpscaler.detailReferenceRadius(scale: 2, metadata: metadata) == 4)
    }

    @Test func supportedScalesComeFromBothImageDimensions() {
        #expect(LearnedUpscaler.reconstructionScale(inputWidth: 640, inputHeight: 360, outputWidth: 1280, outputHeight: 720) == 2)
        #expect(LearnedUpscaler.reconstructionScale(inputWidth: 640, inputHeight: 360, outputWidth: 2560, outputHeight: 1440) == 4)
        for (width, height) in [(1281, 720), (1280, 1440), (1920, 1080), (0, 0)] {
            #expect(LearnedUpscaler.reconstructionScale(inputWidth: 640, inputHeight: 360, outputWidth: width, outputHeight: height) == nil)
        }
        #expect(LearnedUpscaler.reconstructionScale(inputWidth: 0, inputHeight: 360, outputWidth: 1280, outputHeight: 720) == nil)
    }
}

struct DiagnosticNV12FramesTests {
    @Test func tightPlanesSurvivePaddedBuffersAndCadenceIsExact() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let data = Data((0..<72).map { UInt8($0) })
        let dataURL = directory.appendingPathComponent("source.nv12")
        try data.write(to: dataURL)
        let manifestURL = directory.appendingPathComponent("input.json")
        var manifest: [String: Any] = ["format": "NV12", "width": 6, "height": 4, "frames": 2,
            "fpsNumerator": 24000, "fpsDenominator": 1001, "data": "source.nv12",
            "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
            "colorSpace": ["primaries": "bt709", "transfer": "bt709", "matrix": "bt709", "fullRange": false],
            "chromaLocation": "left", "spatialStride": 1, "exportWarmup": true]
        func writeManifest() throws {
            try JSONSerialization.data(withJSONObject: manifest).write(to: manifestURL)
        }
        try writeManifest()
        let reader = try DiagnosticNV12Frames(manifestURL: manifestURL)
        for index in 0..<2 {
            let (buffer, time) = try #require(try reader.next())
            #expect(time.value == Int64(index * 1001))
            #expect(time.timescale == 24000)
            #expect(VideoColorInfo.read(from: buffer) == .rec709)
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            for plane in 0..<2 {
                let base = CVPixelBufferGetBaseAddressOfPlane(buffer, plane)!.assumingMemoryBound(to: UInt8.self)
                let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane)
                for y in 0..<(plane == 0 ? 4 : 2) { for x in 0..<6 {
                    #expect(base[y * stride + x] == data[index * 36 + (plane == 0 ? 0 : 24) + y * 6 + x])
                }}
            }
        }
        #expect(try reader.next() == nil)
        for (key, value) in [("width", 7 as Any), ("fpsNumerator", 0 as Any), ("sha256", "wrong" as Any)] {
            let old = manifest[key]; manifest[key] = value; try writeManifest()
            #expect(throws: (any Error).self) { try DiagnosticNV12Frames(manifestURL: manifestURL) }
            manifest[key] = old
        }
        try writeManifest()
        let truncated = try DiagnosticNV12Frames(manifestURL: manifestURL)
        let writer = try FileHandle(forWritingTo: dataURL)
        try writer.truncate(atOffset: 10); try writer.close()
        #expect(throws: (any Error).self) { try truncated.next() }
    }
}

private func nv12(_ w: Int = 64, _ h: Int = 64, luma: (Int, Int) -> UInt8) throws -> CVPixelBuffer {
    var result: CVPixelBuffer?
    let attributes: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:], kCVPixelBufferMetalCompatibilityKey as String: true]
    #expect(CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, attributes as CFDictionary, &result) == kCVReturnSuccess)
    let buffer = try #require(result)
    CVPixelBufferLockBaseAddress(buffer, [])
    for p in 0..<2 {
        let pointer = CVPixelBufferGetBaseAddressOfPlane(buffer, p)!.assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, p)
        for y in 0..<(p == 0 ? h : h / 2) { for x in 0..<w { pointer[y * stride + x] = p == 0 ? luma(x, y) : 128 } }
    }
    CVPixelBufferUnlockBaseAddress(buffer, [])
    VideoColorInfo.rec709.apply(to: buffer)
    return buffer
}
private func lumaBytes(_ buffer: CVPixelBuffer) -> [Double] {
    CVPixelBufferLockBaseAddress(buffer, .readOnly); defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)!.assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
    return (0..<CVPixelBufferGetHeight(buffer)).flatMap { y in (0..<CVPixelBufferGetWidth(buffer)).map { x in Double(base[y * stride + x]) } }
}
struct FrameLayoutTests {
    @Test @MainActor func pageRenderingNeedsNoNativeWindow() {
        let r = BrowserVideoReport(type: .video, browser: "chrome", session: "test", title: "Video", url: nil,
            visible: true, screenX: 22, screenY: 55, outerWidth: 1200, outerHeight: 800,
            innerWidth: 1200, innerHeight: 700, dpr: 2, video: nil,
            frames: true, draws: true, moving: false, hover: false, cutouts: [], ts: 0)
        let geometry = WindowTracker.pageGeometry(for: r)
        #expect(geometry.id == 0)
        #expect(geometry.bounds.width == 1200)
    }
    @Test func malformedPlanesNeverReadRecycledBytes() {
        var h = DecodedFrame.Header(session: "s", w: 64, h: 64, format: "NV12", planes: [.init(offset: 0, stride: 64), .init(offset: 4096, stride: 64)], seq: 1, ts: 0)
        #expect(DecodedFrameSource.validLayout(h, payloadCount: 6144))
        #expect(!DecodedFrameSource.validLayout(h, payloadCount: 6143))
        h.planes[1].offset = Int.max
        #expect(!DecodedFrameSource.validLayout(h, payloadCount: 6144))
        h.planes[1].offset = 4096; h.planes[0].stride = Int.max
        #expect(!DecodedFrameSource.validLayout(h, payloadCount: 6144))
        h.ts = .nan
        #expect(!DecodedFrameSource.validLayout(h, payloadCount: 6144))
    }
    @Test func bgraAndRgbaProduceTheSameRed() async throws {
        var results: [[Double]] = []
        for format in ["RGBA", "BGRA"] {
            let source = DecodedFrameSource(); let stream = source.stream()
            let pixel: [UInt8] = format == "RGBA" ? [255, 0, 0, 255] : [0, 0, 255, 255]
            let h = DecodedFrame.Header(session: "s", w: 64, h: 64, format: format, planes: [.init(offset: 0, stride: 256)], seq: 7, ts: 1234,
                colorSpace: .init(primaries: "bt709", transfer: "iec61966-2-1", matrix: "rgb", fullRange: true))
            source.accept(.init(header: h, payload: Data(Array(repeating: pixel, count: 4096).flatMap { $0 })))
            source.finish()
            var iterator = stream.makeAsyncIterator()
            let next = await iterator.next(); let frame = try #require(next)
            #expect(frame.sequence == 7)
            #expect(VideoColorInfo.read(from: frame.pixelBuffer) == .init(
                primaries: "bt709", transfer: "iec61966-2-1", matrix: "bt709", fullRange: false))
            results.append(lumaBytes(frame.pixelBuffer))
        }
        #expect(results[0] == results[1])
        #expect(results[0][0] > 45 && results[0][0] < 90)
    }
    @Test func browserMidtonesKeepTheirTransferAcrossPooledFrames() async throws {
        let source = DecodedFrameSource(), stream = source.stream()
        var iterator = stream.makeAsyncIterator()
        for transfer in ["iec61966-2-1", "bt709", "iec61966-2-1"] {
            let color = VideoColorInfo(primaries: "bt709", transfer: transfer, matrix: "rgb", fullRange: true)
            let header = DecodedFrame.Header(session: "s", w: 64, h: 64, format: "RGBA",
                planes: [.init(offset: 0, stride: 256)], seq: 1, ts: 0, colorSpace: color)
            source.accept(.init(header: header, payload: Data(Array(repeating: [UInt8(128),128,128,255], count: 4096).flatMap { $0 })))
            let next = await iterator.next(); let frame = try #require(next)
            #expect(VideoColorInfo.read(from: frame.pixelBuffer).transfer == transfer)
            // Neutral 128 RGB is 126 in video-range Y; the old sRGB->709
            // conversion incorrectly produced 116, which Chrome drew dark.
            #expect(lumaBytes(frame.pixelBuffer).allSatisfy { abs($0 - 126) <= 1 })
        }
        source.finish()
    }
    @Test func senderPreservesTransferAndDescribesActualRange() throws {
        let sender = EnhancedFrameSender()
        for full in [true, false, true] {
            var buffer: CVPixelBuffer?
            let format = full ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            #expect(CVPixelBufferCreate(nil, 64, 64, format,
                [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer) == kCVReturnSuccess)
            let input = try #require(buffer)
            CVPixelBufferLockBaseAddress(input, [])
            for plane in 0..<2 {
                memset(CVPixelBufferGetBaseAddressOfPlane(input, plane), 128,
                       CVPixelBufferGetBytesPerRowOfPlane(input, plane) * CVPixelBufferGetHeightOfPlane(input, plane))
            }
            CVPixelBufferUnlockBaseAddress(input, [])
            VideoColorInfo(primaries: "bt709", transfer: "iec61966-2-1", matrix: "bt709", fullRange: full).apply(to: input)
            for width in [64, 32] {
                sender.maximumWidth = width
                let packet = try #require(sender.packet(for: input, sequence: 1, session: "s"))
                let count = packet[4..<8].reduce(0) { ($0 << 8) | Int($1) }
                let header = try #require(JSONSerialization.jsonObject(with: packet.subdata(in: 8..<(8 + count))) as? [String: Any])
                let color = try #require(header["colorSpace"] as? [String: Any])
                #expect(color["transfer"] as? String == "iec61966-2-1")
                #expect(color["fullRange"] as? Bool == (full && width == 64))
                #expect(abs(Int(packet[8 + count]) - (full && width == 32 ? 126 : 128)) <= 1)
            }
        }
    }
    @Test func hdrIsDeclined() async {
        let source = DecodedFrameSource(); let stream = source.stream()
        let h = DecodedFrame.Header(session: "s", w: 64, h: 64, format: "NV12", planes: [.init(offset: 0, stride: 64), .init(offset: 4096, stride: 64)], seq: 1, ts: 0,
            colorSpace: .init(primaries: "bt2020", transfer: "smpte2084", matrix: "bt2020-ncl", fullRange: false))
        source.accept(.init(header: h, payload: Data(repeating: 128, count: 6144))); source.finish()
        var iterator = stream.makeAsyncIterator(); let frame = await iterator.next()
        #expect(frame == nil)
    }
}
@Suite(.serialized)
struct MetalFrameIntegrityTests {
    @Test func learnedOutputPreservesInputEncoding() throws {
        let model = try LearnedUpscaler(width: 256, height: 144)
        for transfer in ["iec61966-2-1", "bt709", "iec61966-2-1"] {
            let input = try nv12(256, 144) { _, _ in 126 }
            let color = VideoColorInfo(primaries: "bt709", transfer: transfer, matrix: "bt709", fullRange: false)
            color.apply(to: input)
            let output = try model.upscale(input)
            #expect(VideoColorInfo.read(from: output) == color)
            let values = lumaBytes(output)
            #expect(abs(values.reduce(0, +) / Double(values.count) - 126) < 5)
        }
    }
    @Test func standaloneGradeAndTemporalToggleAreInitialized() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        var settings = DetailSettings.off; settings.stageDeband = true; settings.grain = 0.02
        let enhancer = try DetailEnhancer(device: device, settings: settings)
        let input = try nv12 { _, _ in 128 }
        let standalone = lumaBytes(try enhancer.process(input))
        #expect(standalone.allSatisfy { abs($0 - 128) <= 4 })
        _ = try enhancer.preprocess(input)
        settings.stageTaa = true; settings.grain = 0; enhancer.settings = settings
        let light = try nv12 { _, _ in 190 }
        #expect(lumaBytes(try enhancer.preprocess(light)).allSatisfy { abs($0 - 190) <= 1 })
    }
    @Test func movingGrainNeverChangesItsAmplitude() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        var settings = DetailSettings.off
        settings.stageDeband = true; settings.grain = 0.04; settings.grainPhase = 1
        let enhancer = try DetailEnhancer(device: device, settings: settings)
        let input = try nv12 { _, _ in 128 }
        var deviations: [Double] = []
        for _ in 0..<64 {
            _ = try enhancer.preprocess(input)
            let values = lumaBytes(try enhancer.process(input))
            deviations.append(values.map { abs($0 - 128) }.reduce(0, +) / Double(values.count))
        }
        #expect((deviations.min() ?? 0) > 0.01)
        #expect((deviations.max() ?? 100) < 3)
        #expect((deviations.max() ?? 100) - (deviations.min() ?? 0) < 0.7)
    }
    @Test func motionRejectsCutsAndTimestampDiscontinuities() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        var settings = DetailSettings.off; settings.stageTaa = true; settings.stageMotion = true; settings.taaFeedback = 0.9
        let enhancer = try DetailEnhancer(device: device, settings: settings)
        let dark = try nv12 { _, _ in 32 }; let light = try nv12 { _, _ in 210 }
        _ = try enhancer.preprocess(dark, timestamp: CMTime(value: 1, timescale: 30))
        let cut = try enhancer.preprocess(light, timestamp: CMTime(value: 2, timescale: 30))
        #expect(lumaBytes(cut).allSatisfy { abs($0 - 210) <= 1 })
        let seek = try enhancer.preprocess(dark, timestamp: CMTime(value: 1, timescale: 30))
        #expect(lumaBytes(seek).allSatisfy { abs($0 - 32) <= 1 })
    }
    @Test func translatedTextureKeepsItsEdges() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        var settings = DetailSettings.off; settings.stageTaa = true; settings.stageMotion = true; settings.taaFeedback = 0.9
        let enhancer = try DetailEnhancer(device: device, settings: settings)
        func pattern(_ x: Int, _ y: Int) -> UInt8 { UInt8(32 + ((max(0, x) * 29 + y * 53) % 170)) }
        let before = try nv12(luma: pattern)
        let after = try nv12 { x, y in pattern(x - 4, y) }
        _ = try enhancer.preprocess(before, timestamp: CMTime(value: 1, timescale: 30))
        let result = lumaBytes(try enhancer.preprocess(after, timestamp: CMTime(value: 2, timescale: 30)))
        let reference = lumaBytes(after)
        let error = zip(result, reference).map { abs($0 - $1) }.reduce(0, +) / Double(result.count)
        #expect(error < 2)
    }
    @Test func staticNoiseDoesNotLoseHistoryWhenMotionIsEnabled() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        var residuals: [Double] = []
        var flickers: [Double] = []
        for motion in [false, true] {
            var settings = DetailSettings.off
            settings.stageTaa = true; settings.stageMotion = motion; settings.taaFeedback = 0.9
            let enhancer = try DetailEnhancer(device: device, settings: settings)
            var errors: [Double] = [], changes: [Double] = [], previous: [Double] = []
            for frame in 0..<24 {
                func truth(_ x: Int, _ y: Int) -> Int { 64 + (x * 17 + y * 29) % 120 }
                let input = try nv12 { x, y in
                    let noise = ((x * 13 + y * 7 + frame) % 2 == 0) ? 3 : -3
                    return UInt8(truth(x, y) + noise)
                }
                let values = lumaBytes(try enhancer.preprocess(input, timestamp: CMTime(value: Int64(frame), timescale: 30)))
                if frame >= 8 {
                    // Ignore matcher boundary conditions; measure the stationary interior.
                    for y in 8..<56 { for x in 8..<56 {
                        let i = y * 64 + x
                        errors.append(pow(values[i] - Double(truth(x, y)), 2))
                        changes.append(abs(values[i] - previous[i]))
                    } }
                }
                previous = values
            }
            residuals.append(errors.reduce(0, +) / Double(errors.count))
            flickers.append(changes.reduce(0, +) / Double(changes.count))
        }
        #expect(residuals[1] <= residuals[0] * 1.05 + 0.05)
        #expect(flickers[1] <= flickers[0] * 1.05 + 0.05)
        #expect(residuals[1] < 3.0) // raw input MSE is 9; require at least a 2/3 reduction.
    }
    @Test func everyBundledModelActuallyPredicts() throws {
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuAndGPU
        for variant in LearnedUpscaler.variants {
            let name = "SPAN_x4_ch32utc_\(variant.width)x\(variant.height)"
            let url = try #require(Bundle.main.url(forResource: name, withExtension: "mlmodelc"))
            let model = try MLModel(contentsOf: url, configuration: configuration)
            let key = try #require(model.modelDescription.inputDescriptionsByName.keys.first)
            var buffer: CVPixelBuffer?
            #expect(CVPixelBufferCreate(nil, variant.width, variant.height, kCVPixelFormatType_32BGRA, [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer) == kCVReturnSuccess)
            let image = try #require(buffer)
            CVPixelBufferLockBaseAddress(image, []); memset(CVPixelBufferGetBaseAddress(image), 128, CVPixelBufferGetDataSize(image)); CVPixelBufferUnlockBaseAddress(image, [])
            let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [key: MLFeatureValue(pixelBuffer: image)]))
            let outputKey = try #require(output.featureNames.first)
            let result = try #require(output.featureValue(for: outputKey)?.imageBufferValue)
            #expect(CVPixelBufferGetWidth(result) == variant.width * 4)
            #expect(CVPixelBufferGetHeight(result) == variant.height * 4)
            CVPixelBufferLockBaseAddress(result, .readOnly)
            let pixel = CVPixelBufferGetBaseAddress(result)!.assumingMemoryBound(to: UInt8.self)
            #expect(pixel[0] > 32 && pixel[0] < 224)
            CVPixelBufferUnlockBaseAddress(result, .readOnly)
        }
    }
}

struct PresentationMetricTests {
    @Test func duplicatesAndInvalidLatenciesNeverInflatePresentationRate() {
        let start = ContinuousClock.now
        var window = PresentationWindow(now: start)
        #expect(window.record(.init(session: "s", seq: 1, latencyMilliseconds: 12), now: start) == nil)
        #expect(window.record(.init(session: "s", seq: 1, latencyMilliseconds: 12), now: start) == nil)
        #expect(window.record(.init(session: "s", seq: 2, latencyMilliseconds: .nan), now: start) == nil)
        let result = window.record(.init(session: "s", seq: 2, latencyMilliseconds: 40), now: start.advanced(by: .seconds(1)))
        #expect(result?.samples == 2)
        #expect(result?.framesPerSecond == 2)
        #expect(result?.p95Milliseconds == 40)
    }
    @Test @MainActor func shippedSettingsMatchTheBenchFile() throws {
        let path = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Tools/tuning.json")
        let file = try JSONDecoder().decode(EnhancementSession.Tuning.self, from: Data(contentsOf: path))
        #expect(file.detailSettings() == EnhancementSession.Tuning().detailSettings())
    }
}

struct ColorMetadataTests {
    @Test func hdrAndP3SurviveNativeAttachments() throws {
        for (wire, expected) in [("pq", "pq"), ("hlg", "hlg"), ("smpte2084", "pq"), ("arib-std-b67", "hlg")] {
            let buffer = try nv12 { _, _ in 128 }
            let color = VideoColorInfo(primaries: "bt2020", transfer: wire, matrix: "bt2020-ncl", fullRange: false)
            #expect(color.isHDR)
            #expect(color.apply(to: buffer))
            let read = VideoColorInfo.read(from: buffer)
            #expect(read.transfer == expected)
            #expect(read.primaries == "bt2020")
            #expect(read.isHDR)
            #expect(!read.supportsSDREnhancement)
        }
        let buffer = try nv12 { _, _ in 128 }
        let p3 = VideoColorInfo(primaries: "smpte432", transfer: "iec61966-2-1", matrix: "bt709", fullRange: false)
        #expect(p3.apply(to: buffer))
        #expect(VideoColorInfo.read(from: buffer) == p3)
    }
    @Test func sdrAliasesAndUnknownAttachmentsRemainDistinguishable() throws {
        let buffer = try nv12 { _, _ in 128 }
        let pal = VideoColorInfo(primaries: "bt470bg", transfer: "smpte170m", matrix: "bt470bg", fullRange: false)
        #expect(pal.apply(to: buffer))
        let read = VideoColorInfo.read(from: buffer)
        #expect(read.primaries == "bt470bg")
        #expect(read.matrix == "smpte170m")
        #expect(read.transfer == "bt709")
        // Core Video rejects arbitrary strings on this key; use a real transfer
        // understood by Core Video but not supported by Lucid's SDR contract.
        CVBufferSetAttachment(buffer, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_SMPTE_240M_1995, .shouldPropagate)
        #expect(!VideoColorInfo.read(from: buffer).supportsSDREnhancement)
        let unknown = VideoColorInfo(primaries: "future-gamut", transfer: "bt709", matrix: "bt709", fullRange: false)
        #expect(!unknown.apply(to: buffer))
        #expect(!unknown.supportsSDREnhancement)
        #expect(VideoColorInfo.read(from: buffer).transfer == "unsupported:\(kCVImageBufferTransferFunction_SMPTE_240M_1995)")
    }
    @Test func newHDRNamesAndUnknownMetadataAreDeclinedBeforeFrameDelivery() async {
        for transfer in ["pq", "hlg", "future-transfer"] {
            let source = DecodedFrameSource(); let stream = source.stream()
            let h = DecodedFrame.Header(session: "s", w: 64, h: 64, format: "NV12",
                planes: [.init(offset: 0, stride: 64), .init(offset: 4096, stride: 64)], seq: 1, ts: 0,
                colorSpace: .init(primaries: "bt2020", transfer: transfer, matrix: "bt2020-ncl", fullRange: false))
            source.accept(.init(header: h, payload: Data(repeating: 128, count: 6144))); source.finish()
            var iterator = stream.makeAsyncIterator()
            let frame = await iterator.next()
            #expect(frame == nil)
        }
    }
}
