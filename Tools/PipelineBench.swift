//
//  PipelineBench.swift
//
//  ModelBench times the Core ML prediction and nothing else. That is not what a
//  frame costs. The pipeline works in 420v and the model wants RGB, so every
//  frame is converted on the way in and converted back on the way out - and the
//  way out is at 4x, which is where the real money goes: 640x360 in means
//  2560x1440 out.
//
//  This measures those two conversions at each shipping size, so the frame
//  budget in LearnedUpscaler can be checked against the whole cost rather than
//  the model alone.
//
//  swiftc -O -framework VideoToolbox Tools/PipelineBench.swift -o .build/pipelinebench
//  .build/pipelinebench
//

import CoreVideo
import Foundation
import VideoToolbox

let variants = [(256, 144), (320, 180), (432, 240), (480, 270), (640, 360)]
let scale = 4

func makeBuffer(_ width: Int, _ height: Int, _ format: OSType) -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, width, height, format,
                        [kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
                         kCVPixelBufferMetalCompatibilityKey as String: true] as CFDictionary,
                        &buffer)
    return buffer!
}

func time(_ iterations: Int, _ body: () -> Void) -> Double {
    for _ in 0..<5 { body() }              // warm the session and the pools
    var samples: [Double] = []
    for _ in 0..<iterations {
        let start = DispatchTime.now().uptimeNanoseconds
        body()
        samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
    }
    samples.sort()
    return samples[samples.count / 2]
}

var session: VTPixelTransferSession?
VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session)
guard let session else { fatalError("no transfer session") }

let video = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
let rgb = kCVPixelFormatType_32BGRA

print("input          in ms     out ms   total ms")
for (width, height) in variants {
    let sourceYUV = makeBuffer(width, height, video)
    let modelIn = makeBuffer(width, height, rgb)
    let modelOut = makeBuffer(width * scale, height * scale, rgb)
    let resultYUV = makeBuffer(width * scale, height * scale, video)

    let into = time(30) { VTPixelTransferSessionTransferImage(session, from: sourceYUV, to: modelIn) }
    let outOf = time(30) { VTPixelTransferSessionTransferImage(session, from: modelOut, to: resultYUV) }

    print(String(format: "%-12@ %10.2f %10.2f %10.2f",
                 "\(width)x\(height)" as NSString, into, outOf, into + outOf))
}
