//
//  MetalTileCompositor.swift
//  Lucid
//
//  Region copies between IOSurface-backed pixel buffers using Metal blit
//  encoders: one command buffer per batch, no shaders, no Core Image, and no
//  format conversion. Works plane-wise so 420v (biplanar YUV) buffers are
//  copied exactly like BGRA ones.
//

import CoreVideo
import Foundation
import Metal

final class MetalTileCompositor: @unchecked Sendable {
    struct Copy {
        let source: CVPixelBuffer
        let sourceRect: CGRect
        let destination: CVPixelBuffer
        let destinationOrigin: CGPoint
    }

    enum Failure: Error {
        case noDevice
        case textureCache
        case texture(plane: Int)
        case formatMismatch
    }

    let device: MTLDevice
    private let queue: MTLCommandQueue
    private var textureCache: CVMetalTextureCache?

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw Failure.noDevice
        }
        self.device = device
        self.queue = queue
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache) == kCVReturnSuccess, let cache else {
            throw Failure.textureCache
        }
        textureCache = cache
    }

    /// Performs all copies in one command buffer and waits for completion.
    func perform(_ copies: [Copy]) throws {
        guard !copies.isEmpty, let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder()
        else { return }
        var retained: [CVMetalTexture] = []
        retained.reserveCapacity(copies.count * 4)

        for copy in copies {
            let format = CVPixelBufferGetPixelFormatType(copy.source)
            guard format == CVPixelBufferGetPixelFormatType(copy.destination) else { throw Failure.formatMismatch }
            let planeCount = max(1, CVPixelBufferGetPlaneCount(copy.source))
            let sourceWidth = CVPixelBufferGetWidth(copy.source)
            let sourceHeight = CVPixelBufferGetHeight(copy.source)
            for plane in 0..<planeCount {
                let planar = CVPixelBufferIsPlanar(copy.source)
                let planeWidth = planar ? CVPixelBufferGetWidthOfPlane(copy.source, plane) : sourceWidth
                let planeHeight = planar ? CVPixelBufferGetHeightOfPlane(copy.source, plane) : sourceHeight
                let scaleX = Double(planeWidth) / Double(sourceWidth)
                let scaleY = Double(planeHeight) / Double(sourceHeight)
                let pixelFormat = Self.metalFormat(for: format, plane: plane)

                let (sourceTexture, sourceRef) = try makeTexture(copy.source, plane: plane, format: pixelFormat)
                let (destinationTexture, destinationRef) = try makeTexture(copy.destination, plane: plane, format: pixelFormat)
                retained.append(sourceRef)
                retained.append(destinationRef)

                let originX = Int((copy.sourceRect.minX * scaleX).rounded())
                let originY = Int((copy.sourceRect.minY * scaleY).rounded())
                let width = min(Int((copy.sourceRect.width * scaleX).rounded()), sourceTexture.width - originX)
                let height = min(Int((copy.sourceRect.height * scaleY).rounded()), sourceTexture.height - originY)
                let destinationX = Int((copy.destinationOrigin.x * scaleX).rounded())
                let destinationY = Int((copy.destinationOrigin.y * scaleY).rounded())
                let clippedWidth = min(width, destinationTexture.width - destinationX)
                let clippedHeight = min(height, destinationTexture.height - destinationY)
                guard clippedWidth > 0, clippedHeight > 0 else { continue }

                blit.copy(
                    from: sourceTexture,
                    sourceSlice: 0,
                    sourceLevel: 0,
                    sourceOrigin: MTLOrigin(x: originX, y: originY, z: 0),
                    sourceSize: MTLSize(width: clippedWidth, height: clippedHeight, depth: 1),
                    to: destinationTexture,
                    destinationSlice: 0,
                    destinationLevel: 0,
                    destinationOrigin: MTLOrigin(x: destinationX, y: destinationY, z: 0)
                )
            }
        }
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        withExtendedLifetime(retained) {}
    }

    private func makeTexture(_ buffer: CVPixelBuffer, plane: Int, format: MTLPixelFormat) throws -> (MTLTexture, CVMetalTexture) {
        guard let textureCache else { throw Failure.textureCache }
        let planar = CVPixelBufferIsPlanar(buffer)
        let width = planar ? CVPixelBufferGetWidthOfPlane(buffer, plane) : CVPixelBufferGetWidth(buffer)
        let height = planar ? CVPixelBufferGetHeightOfPlane(buffer, plane) : CVPixelBufferGetHeight(buffer)
        var textureRef: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, buffer, nil, format, width, height, plane, &textureRef
        )
        guard status == kCVReturnSuccess, let textureRef, let texture = CVMetalTextureGetTexture(textureRef) else {
            throw Failure.texture(plane: plane)
        }
        return (texture, textureRef)
    }

    static func metalFormat(for pixelFormat: OSType, plane: Int) -> MTLPixelFormat {
        switch pixelFormat {
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
            return plane == 0 ? .r8Unorm : .rg8Unorm
        case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
            return plane == 0 ? .r16Unorm : .rg16Unorm
        default:
            return .bgra8Unorm
        }
    }
}
