//
//  EngineKind.swift
//  Lucid
//
//  The shipping pipeline is SPAN on the Neural Engine. The other two cases
//  exist for the offline bench (`--bench`, LUCID_SHOOT=1) and are not
//  constructed on a normal launch.
//

import Foundation

enum EngineKind: String, Codable, Sendable, CaseIterable {
    case apple
    case lucid
    case learned

    static let shipping: [EngineKind] = [.learned]

    /// Apple's tiled scaler and the comparison shooter. Off unless the process
    /// was launched as a bench.
    static var comparisonPathEnabled: Bool {
        let env = ProcessInfo.processInfo.environment
        return env["LUCID_SHOOT"] == "1"
            || env["LUCID_DEBUG"] == "1"
            || CommandLine.arguments.contains("--bench")
    }

    var usesDetail: Bool { self != .apple }
    var rescales: Bool { true }
    var usesLearned: Bool { self == .learned }

    var label: String {
        switch self {
        case .apple: return "Apple scaler only"
        case .lucid: return "Apple scaler + Lucid"
        case .learned: return "Lucid"
        }
    }
}
