import CoreVideo
import Foundation

/// WebCodecs names on the wire; Core Video attachments at the native boundary.
struct VideoColorInfo: Codable, Sendable, Equatable {
    var primaries: String?
    var transfer: String?
    var matrix: String?
    var fullRange: Bool?
    static let rec709 = VideoColorInfo(primaries: "bt709", transfer: "bt709", matrix: "bt709", fullRange: false)
    var isHDR: Bool { transferName == "pq" || transferName == "hlg" }

    // Keep both directions in one table. Unknown metadata must not become 709:
    // that could admit HDR as SDR or skip a required gamut conversion.
    private static let primaryTags: [String: String] = [
        "bt709": kCVImageBufferColorPrimaries_ITU_R_709_2 as String,
        "bt470bg": kCVImageBufferColorPrimaries_EBU_3213 as String,
        "smpte170m": kCVImageBufferColorPrimaries_SMPTE_C as String,
        "bt2020": kCVImageBufferColorPrimaries_ITU_R_2020 as String,
        "smpte432": kCVImageBufferColorPrimaries_P3_D65 as String,
    ]
    private static let transferTags: [String: String] = [
        "bt709": kCVImageBufferTransferFunction_ITU_R_709_2 as String,
        "iec61966-2-1": kCVImageBufferTransferFunction_sRGB as String,
        "linear": kCVImageBufferTransferFunction_Linear as String,
        "pq": kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ as String,
        "hlg": kCVImageBufferTransferFunction_ITU_R_2100_HLG as String,
    ]
    private static let matrixTags: [String: String] = [
        "bt709": kCVImageBufferYCbCrMatrix_ITU_R_709_2 as String,
        "smpte170m": kCVImageBufferYCbCrMatrix_ITU_R_601_4 as String,
        "bt2020-ncl": kCVImageBufferYCbCrMatrix_ITU_R_2020 as String,
    ]
    private var transferName: String {
        switch transfer {
        case "smpte170m": return "bt709"
        case "smpte2084": return "pq"
        case "arib-std-b67": return "hlg"
        default: return transfer ?? "bt709"
        }
    }
    private var matrixName: String { matrix == "bt470bg" ? "smpte170m" : matrix ?? "bt709" }
    var hasSupportedMetadata: Bool {
        Self.primaryTags[primaries ?? "bt709"] != nil && Self.transferTags[transferName] != nil
            && (matrixName == "rgb" || Self.matrixTags[matrixName] != nil)
    }
    var supportsSDREnhancement: Bool { hasSupportedMetadata && !isHDR }

    /// Missing fields use the browser path's SDR default; explicit unknowns fail.
    /// RGB has no YCbCr matrix, so clear any old attachment on a pooled buffer.
    @discardableResult
    func apply(to buffer: CVPixelBuffer) -> Bool {
        guard hasSupportedMetadata,
              let p = Self.primaryTags[primaries ?? "bt709"],
              let t = Self.transferTags[transferName] else { return false }
        CVBufferSetAttachment(buffer, kCVImageBufferColorPrimariesKey, p as CFString, .shouldPropagate)
        CVBufferSetAttachment(buffer, kCVImageBufferTransferFunctionKey, t as CFString, .shouldPropagate)
        if let m = Self.matrixTags[matrixName] {
            CVBufferSetAttachment(buffer, kCVImageBufferYCbCrMatrixKey, m as CFString, .shouldPropagate)
        } else {
            CVBufferRemoveAttachment(buffer, kCVImageBufferYCbCrMatrixKey)
        }
        return true
    }

    static func read(from buffer: CVPixelBuffer) -> VideoColorInfo {
        func name(_ key: CFString, tags: [String: String]) -> String? {
            guard let tag = CVBufferCopyAttachment(buffer, key, nil) as? String else { return nil }
            // 2020 SDR and 709 transfer functions are equivalent in Core Video.
            if key == kCVImageBufferTransferFunctionKey,
               tag == kCVImageBufferTransferFunction_ITU_R_2020 as String { return "bt709" }
            return tags.first { $0.value == tag }?.key ?? "unsupported:\(tag)"
        }
        let format = CVPixelBufferGetPixelFormatType(buffer)
        let rgb = [kCVPixelFormatType_32BGRA, kCVPixelFormatType_32RGBA,
                   kCVPixelFormatType_32ARGB, kCVPixelFormatType_64RGBAHalf].contains(format)
        let full = rgb || [kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                           kCVPixelFormatType_420YpCbCr8PlanarFullRange,
                           kCVPixelFormatType_420YpCbCr10BiPlanarFullRange].contains(format)
        return VideoColorInfo(
            primaries: name(kCVImageBufferColorPrimariesKey, tags: primaryTags),
            transfer: name(kCVImageBufferTransferFunctionKey, tags: transferTags),
            matrix: rgb ? "rgb" : name(kCVImageBufferYCbCrMatrixKey, tags: matrixTags),
            fullRange: full)
    }
}
