import CoreVideo
import Foundation

/// WebCodecs names on the wire; Core Video attachments at the native boundary.
struct VideoColorInfo: Codable, Sendable, Equatable {
    var primaries: String?
    var transfer: String?
    var matrix: String?
    var fullRange: Bool?
    static let rec709 = VideoColorInfo(primaries: "bt709", transfer: "bt709", matrix: "bt709", fullRange: false)
    var isHDR: Bool { transfer == "smpte2084" || transfer == "arib-std-b67" }

    func apply(to buffer: CVPixelBuffer) {
        let p: CFString = primaries == "bt2020" ? kCVImageBufferColorPrimaries_ITU_R_2020
            : primaries == "smpte170m" ? kCVImageBufferColorPrimaries_SMPTE_C
            : primaries == "bt470bg" ? kCVImageBufferColorPrimaries_EBU_3213 : kCVImageBufferColorPrimaries_ITU_R_709_2
        let t: CFString = transfer == "iec61966-2-1" ? kCVImageBufferTransferFunction_sRGB : kCVImageBufferTransferFunction_ITU_R_709_2
        let m: CFString = matrix == "bt2020-ncl" ? kCVImageBufferYCbCrMatrix_ITU_R_2020
            : ["smpte170m", "bt470bg"].contains(matrix ?? "") ? kCVImageBufferYCbCrMatrix_ITU_R_601_4 : kCVImageBufferYCbCrMatrix_ITU_R_709_2
        for (key, value) in [(kCVImageBufferColorPrimariesKey, p), (kCVImageBufferTransferFunctionKey, t), (kCVImageBufferYCbCrMatrixKey, m)] {
            CVBufferSetAttachment(buffer, key, value, .shouldPropagate)
        }
    }

    static func read(from buffer: CVPixelBuffer) -> VideoColorInfo {
        func tag(_ key: CFString) -> String { CVBufferCopyAttachment(buffer, key, nil) as? String ?? "" }
        let p = tag(kCVImageBufferColorPrimariesKey), t = tag(kCVImageBufferTransferFunctionKey), m = tag(kCVImageBufferYCbCrMatrixKey)
        return VideoColorInfo(
            primaries: p == kCVImageBufferColorPrimaries_ITU_R_2020 as String ? "bt2020" : p == kCVImageBufferColorPrimaries_SMPTE_C as String ? "smpte170m" : p == kCVImageBufferColorPrimaries_EBU_3213 as String ? "bt470bg" : "bt709",
            transfer: t == kCVImageBufferTransferFunction_sRGB as String ? "iec61966-2-1" : "bt709",
            matrix: m == kCVImageBufferYCbCrMatrix_ITU_R_601_4 as String ? "smpte170m" : m == kCVImageBufferYCbCrMatrix_ITU_R_2020 as String ? "bt2020-ncl" : "bt709",
            fullRange: CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
    }
}
