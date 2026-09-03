//
//  LucidApp.swift
//  Lucid
//

import AppKit
import SwiftUI

/// Serves the same menu from the Dock icon that the menu bar shows.
@MainActor
final class LucidAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDockMenu(_ sender: NSApplication) -> NSMenu? {
        AppCoordinator.shared.dockMenu()
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
}

@main
enum LucidMain {
    static func main() {
        // Offline engine comparison; must run before any app state is created.
        if CommandLine.arguments.contains("--bench") {
            if #available(macOS 26.0, *) {
                Task.detached { await EngineBench.run() }
                RunLoop.main.run()
            }
            exit(2)
        }
        LucidApp.main()
    }
}

struct LucidApp: App {
    @State private var coordinator = AppCoordinator.shared
    @NSApplicationDelegateAdaptor(LucidAppDelegate.self) private var delegate

    init() {
        if ProcessInfo.processInfo.environment["LUCID_DEBUG"] == "1" { setvbuf(stdout, nil, _IOLBF, 0) }
        // A Dock icon is what makes a Dock menu possible at all. Set this back
        // to .accessory for a menu-bar-only build.
        NSApp.setActivationPolicy(.regular)
    }

    var body: some Scene {
        Settings {
            VStack(alignment: .leading, spacing: 8) {
                Text("Lucid enhances browser video automatically.")
                Text("There are no settings. Use the menu bar item to pause it.")
                    .foregroundStyle(.secondary)
            }
            .padding(24)
        }
    }
}
