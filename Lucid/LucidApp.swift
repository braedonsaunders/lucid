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
        if CommandLine.arguments.contains("--pipeline-ms") {
            if #available(macOS 26.0, *) {
                Task.detached { await PipelineTiming.run() }
                RunLoop.main.run()
            }
            exit(2)
        }
        if CommandLine.arguments.contains("--delivery-ms") {
            Task.detached { DeliveryTiming.run() }
            RunLoop.main.run()
        }
        LucidApp.main()
    }
}

struct LucidApp: App {
    @State private var coordinator = AppCoordinator.shared
    @NSApplicationDelegateAdaptor(LucidAppDelegate.self) private var delegate

    init() {
        if ProcessInfo.processInfo.environment["LUCID_DEBUG"] == "1" { setvbuf(stdout, nil, _IOLBF, 0) }
        // Menu bar only. A Dock icon would let us offer a Dock menu, but it
        // also makes Lucid a regular app, and activating a regular app while
        // you are watching something full screen throws you out of that Space.
        // Being unobtrusive matters more here than having a second menu.
        NSApp.setActivationPolicy(.accessory)
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
