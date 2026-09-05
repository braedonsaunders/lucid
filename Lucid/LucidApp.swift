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
        if let index = CommandLine.arguments.firstIndex(of: "--panel-snapshot"), CommandLine.arguments.count > index + 1 {
            let app = NSApplication.shared
            app.setActivationPolicy(.prohibited)
            let model = ControlPanelModel()
            model.enhancing = true; model.connected = true
            model.status = "Browser video · 360p → 1440p"
            model.stats = "Native enhancement · Apple silicon"
            app.appearance = NSAppearance(named: .darkAqua)
            let view = NSHostingView(rootView: ControlPanel(model: model).environment(\.colorScheme, .dark))
            let size = NSSize(width: 360, height: 430)
            let window = NSWindow(contentRect: NSRect(origin: NSPoint(x: -10000, y: -10000), size: size), styleMask: .borderless, backing: .buffered, defer: false)
            window.contentView = view
            view.frame = NSRect(origin: .zero, size: size)
            window.orderFrontRegardless()
            RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.2))
            view.layoutSubtreeIfNeeded()
            if let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) {
                view.cacheDisplay(in: view.bounds, to: bitmap)
                window.orderOut(nil)
                if let data = bitmap.representation(using: .png, properties: [:]) {
                    do { try data.write(to: URL(fileURLWithPath: CommandLine.arguments[index + 1])); exit(0) }
                    catch { print(error); exit(1) }
                }
            }
            exit(1)
        }
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
