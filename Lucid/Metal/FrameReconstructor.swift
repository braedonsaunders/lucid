import CoreVideo
import Foundation
import Metal

/// The reconstruction stage of the pipeline: one frame in the pipeline's own
/// 4:2:0 format in, one frame `scale` times larger in the same format out.
/// Everything before it (preprocess) and after it (detail) is unchanged by
/// which reconstructor is loaded, which is what lets the same harness measure
/// a learned graph and Apple's temporal scaler through identical stages.
protocol FrameReconstructor: AnyObject, Sendable {
    var scale: Int { get }
    var inputWidth: Int { get }
    var inputHeight: Int { get }
    var outputWidth: Int { get }
    var outputHeight: Int { get }
    /// Spatial radius the detail stage should treat as nominal for this output.
    var detailReferenceRadius: Int { get }
    func upscale(_ source: CVPixelBuffer) throws -> CVPixelBuffer
}

extension LearnedUpscaler: FrameReconstructor {}

extension LearnedUpscaler {
    /// Builds whichever reconstructor the current model selection names.
    /// Learned graphs come from the bundle (or the harness's package); the
    /// MetalFX stem is a pseudo-model that needs no package at all.
    static func makeReconstructor(width: Int, height: Int) throws -> any FrameReconstructor {
        if MetalFXTemporalUpscaler.isSelected {
            return try MetalFXTemporalUpscaler(width: width, height: height)
        }
        return try LearnedUpscaler(width: width, height: height)
    }
}
