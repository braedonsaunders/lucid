import Foundation
import Observation
import ServiceManagement

/// macOS owns the preference. Always read it back, including changes made in
/// System Settings, rather than keeping a separate UserDefaults switch.
@MainActor
@Observable
final class LoginItemController {
    private(set) var status: SMAppService.Status = .notRegistered
    private(set) var errorMessage: String?
    private let readStatus: () -> SMAppService.Status
    private let register: () throws -> Void
    private let unregister: () throws -> Void
    private let openSettings: () -> Void

    init(readStatus: @escaping () -> SMAppService.Status = { SMAppService.mainApp.status },
         register: @escaping () throws -> Void = { try SMAppService.mainApp.register() },
         unregister: @escaping () throws -> Void = { try SMAppService.mainApp.unregister() },
         openSettings: @escaping () -> Void = { SMAppService.openSystemSettingsLoginItems() }) {
        self.readStatus = readStatus
        self.register = register
        self.unregister = unregister
        self.openSettings = openSettings
        refresh()
    }

    var isEnabled: Bool { status == .enabled }
    var requiresApproval: Bool { status == .requiresApproval }

    func refresh() { status = readStatus() }

    func setEnabled(_ enabled: Bool) {
        refresh()
        errorMessage = nil
        do {
            if enabled {
                if requiresApproval {
                    openSettings()
                } else if !isEnabled {
                    try register()
                }
            } else if status == .enabled || status == .requiresApproval {
                try unregister()
            }
        } catch {
            errorMessage = "Couldn’t change launch at login: \(error.localizedDescription)"
        }
        refresh()
    }

    func showSettings() { openSettings() }
}
