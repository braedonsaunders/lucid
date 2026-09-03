//
//  EngineKind.swift
//  Lucid
//
//  Which reconstruction engine a session runs.
//
//  Frames arrive from the page at the video's own decoded resolution and are
//  reconstructed by Apple's scaler, which is both the fastest option measured
//  and the one that looks best. Lucid's own stages then do what the scaler
//  deliberately does not: put back contrast, black level and colour that
//  compression flattened, and sharpen at the right scale.
//
//  (Re-homed here from NeuralUpscaler.swift when the learned-engine class was
//  removed. The Neural Engine student remains a documented future step in
//  PLAN-INVISIBLE-PRESENTATION.md; there is intentionally no neural case
//  until a distilled model ships.)
//

enum EngineKind: String, Codable, Sendable, CaseIterable {
    case apple          // Apple's scaler alone, for comparison
    case lucid          // Apple's scaler plus Lucid's grade and detail stages

    var usesDetail: Bool { self == .lucid }
    var rescales: Bool { true }

    var label: String {
        switch self {
        case .apple: return "Apple scaler only"
        case .lucid: return "Lucid"
        }
    }
}
