//
//  EngineKind.swift
//  Lucid
//
//  Which reconstruction engine a session runs.
//
//  Frames arrive from the page at the video's own decoded resolution. Measured
//  against a 1080p reference on compressed source, fine-band correlation with
//  the truth - which separates recovered detail from invented detail:
//
//      SPAN on the Neural Engine     0.221
//      lanczos anchor                0.180
//      Apple's scaler                0.174
//      Apple's scaler + our detail   0.152
//
//  So the learned engine is used wherever it fits the frame budget, and Apple's
//  scaler is the fallback for sources too large for it.
//

enum EngineKind: String, Codable, Sendable, CaseIterable {
    case apple          // Apple's scaler alone, for comparison
    case lucid          // Apple's scaler plus Lucid's grade and detail stages
    case learned        // SPAN on the Neural Engine, Apple's scaler as fallback

    var usesDetail: Bool { self != .apple }
    var rescales: Bool { true }
    /// Whether to try the Core ML upscaler before falling back to Apple's.
    var usesLearned: Bool { self == .learned }

    var label: String {
        switch self {
        case .apple: return "Apple scaler only"
        case .lucid: return "Apple scaler + Lucid"
        case .learned: return "Neural Engine"
        }
    }
}
