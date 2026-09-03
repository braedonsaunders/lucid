//
//  AppState.swift
//  Lucid
//

import Foundation
import SwiftUI

@MainActor
@Observable
final class AppState {
    /// Master switch. Off hides everything and stops capture.
    var enabled: Bool = UserDefaults.standard.object(forKey: "enhancementEnabled") as? Bool ?? true {
        didSet { UserDefaults.standard.set(enabled, forKey: "enhancementEnabled") }
    }
    var isEnhancing = false
    var statusLine = "Waiting for browser video"
    var statsLine = ""
    var lastError: String?
    var connectedBrowsers: Set<String> = []
}
