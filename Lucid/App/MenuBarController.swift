//
//  MenuBarController.swift
//  Lucid
//
//  The whole of Lucid's interface. Two controls - on/off and a strength - plus
//  a line saying what it is doing. The same menu is served from the menu bar
//  and from the Dock icon, because a person should not have to remember which
//  one this app happens to live in.
//

import AppKit
import Foundation
import SwiftUI

@MainActor
final class MenuBarController: NSObject, NSMenuDelegate, NSPopoverDelegate {
    private var statusItem: NSStatusItem?
    /// The menu bar item opens a panel rather than a menu. A menu dismisses
    /// itself the moment you choose anything, which makes it useless for
    /// adjusting a picture while you watch it.
    private var popover: NSPopover?
    /// A transient popover is dismissed by any click outside it, including the
    /// click on the very item that opens it. Without noting when that happened,
    /// clicking the icon to close the panel would close and immediately reopen
    /// it, so it could never be dismissed that way.
    private var panelClosedAt: Date = .distantPast
    let panel = ControlPanelModel()
    private let appState: AppState
    /// One menu per host. An NSMenu cannot be attached in two places at once.
    private var menus: [NSMenu] = []
    var onToggleEnabled: ((Bool) -> Void)?
    var onOpenTestPage: (() -> Void)?
    var onStrengthChanged: ((EnhancementSession.Tuning.Strength) -> Void)?

    /// Persisted so the app comes back the way it was left.
    private(set) var strength: EnhancementSession.Tuning.Strength {
        get {
            EnhancementSession.Tuning.Strength(
                rawValue: UserDefaults.standard.string(forKey: "strength") ?? ""
            ) ?? .standard
        }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: "strength") }
    }

    init(appState: AppState) {
        self.appState = appState
        super.init()
        setup()
    }

    private func setup() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        item.button?.image = NSImage(systemSymbolName: "camera.aperture", accessibilityDescription: "Lucid")
        item.button?.image?.isTemplate = true
        item.button?.target = self
        item.button?.action = #selector(statusItemClicked)
        item.button?.sendAction(on: [.leftMouseUp, .rightMouseUp])
        statusItem = item

        panel.onEnabledChange = { [weak self] on in
            guard let self else { return }
            appState.enabled = on
            onToggleEnabled?(on)
            refresh()
        }
        panel.onStrengthChange = { [weak self] option in
            guard let self else { return }
            strength = option
            onStrengthChanged?(option)
            panel.tuning = EnhancementSession.tuning
            refresh()
        }
        panel.onTuningChange = { [weak self] tuning in
            self?.onTuningChanged?(tuning)
        }
        panel.onOpenLab = { [weak self] in self?.onOpenTestPage?() }
        panel.onReset = { [weak self] in
            guard let self else { return }
            onResetTuning?()
            panel.tuning = EnhancementSession.tuning
        }

        let popover = NSPopover()
        popover.delegate = self
        popover.behavior = .transient
        popover.animates = true
        popover.contentViewController = NSHostingController(rootView: ControlPanel(model: panel))
        self.popover = popover

        refresh()
    }

    var onTuningChanged: ((EnhancementSession.Tuning) -> Void)?
    var onResetTuning: (() -> Void)?

    @objc private func statusItemClicked() {
        guard let button = statusItem?.button else { return }
        let event = NSApp.currentEvent
        if AppCoordinator.debugLogging {
            print("   🖱 status item clicked: event=\(event?.type.rawValue.description ?? "nil") popoverShown=\(popover?.isShown == true)")
        }
        let wantsMenu = event?.type == .rightMouseUp
            || event?.modifierFlags.contains(.control) == true

        if wantsMenu {
            // Pop the menu directly. Assigning statusItem.menu hands click
            // handling to AppKit permanently, which left the item dead to
            // every click afterwards.
            let menu = makeMenu(retained: false)
            menu.popUp(positioning: nil,
                       at: NSPoint(x: 0, y: button.bounds.height + 5),
                       in: button)
            return
        }

        if popover?.isShown == true {
            popover?.performClose(nil)
            return
        }
        if Date().timeIntervalSince(panelClosedAt) < 0.3 { return }
        showPanel(from: button)
    }

    private func showPanel(from button: NSStatusBarButton) {
        guard let popover else { return }
        syncPanel()
        // Deliberately does NOT activate the app. Lucid is used over full-screen
        // video, and activating pulls you out of that Space - the menu bar just
        // hides again and the panel never appears. The popover is shown without
        // taking activation, and closes itself on the next click elsewhere.
        popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
        popover.contentViewController?.view.window?.makeKey()
        if AppCoordinator.debugLogging {
            print("   🖱 panel shown=\(popover.isShown) window=\(popover.contentViewController?.view.window != nil)")
        }
    }

    private func syncPanel() {
        panel.enabled = appState.enabled
        panel.strength = strength
        panel.status = appState.statusLine
        panel.stats = appState.statsLine
        panel.enhancing = appState.isEnhancing
        panel.tuning = EnhancementSession.tuning
    }

    /// Builds a fresh copy of the menu. Called once for the menu bar and again
    /// each time the Dock asks for one.
    func makeMenu(retained: Bool = true) -> NSMenu {
        let menu = NSMenu()
        menu.delegate = self
        menu.autoenablesItems = false

        let status = NSMenuItem(title: appState.statusLine, action: nil, keyEquivalent: "")
        status.isEnabled = false
        status.tag = Tag.status.rawValue
        menu.addItem(status)

        let stats = NSMenuItem(title: appState.statsLine, action: nil, keyEquivalent: "")
        stats.isEnabled = false
        stats.tag = Tag.stats.rawValue
        menu.addItem(stats)

        menu.addItem(.separator())

        let enable = NSMenuItem(title: "Enhance Browser Video", action: #selector(toggleEnabled), keyEquivalent: "")
        enable.target = self
        enable.tag = Tag.enable.rawValue
        menu.addItem(enable)

        let quality = NSMenuItem(title: "Quality", action: nil, keyEquivalent: "")
        let submenu = NSMenu()
        submenu.autoenablesItems = false
        for option in EnhancementSession.Tuning.Strength.allCases {
            let entry = NSMenuItem(title: option.label, action: #selector(pickStrength(_:)), keyEquivalent: "")
            entry.target = self
            entry.representedObject = option.rawValue
            entry.toolTip = option.detail
            entry.tag = Tag.strengthBase.rawValue
            submenu.addItem(entry)
        }
        quality.submenu = submenu
        menu.addItem(quality)

        menu.addItem(.separator())
        let lab = NSMenuItem(title: "Open Test Lab…", action: #selector(openTestPage), keyEquivalent: "t")
        lab.target = self
        menu.addItem(lab)

        menu.addItem(.separator())
        // Which build this is. Worth having in the menu rather than only in
        // Get Info: this app is replaced often, and "is the thing running the
        // thing I just built" is otherwise answered by guesswork.
        let bundle = Bundle.main.infoDictionary
        let short = bundle?["CFBundleShortVersionString"] as? String ?? "?"
        let version = NSMenuItem(title: "Lucid \(short)", action: nil, keyEquivalent: "")
        version.isEnabled = false
        menu.addItem(version)

        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Quit Lucid", action: #selector(quit), keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)

        if retained { menus.append(menu) }
        update(menu)
        return menu
    }

    private enum Tag: Int {
        case status = 1, stats = 2, enable = 3, strengthBase = 4
    }

    func refresh() {
        for menu in menus { update(menu) }
        if popover?.isShown == true {
            panel.status = appState.statusLine
            panel.stats = appState.statsLine
            panel.enhancing = appState.isEnhancing
            panel.enabled = appState.enabled
        }
        statusItem?.button?.appearsDisabled = !appState.enabled
        let symbol = appState.isEnhancing ? "camera.aperture" : "circle.dotted"
        statusItem?.button?.image = NSImage(systemSymbolName: symbol, accessibilityDescription: "Lucid")
        statusItem?.button?.image?.isTemplate = true
    }

    private func update(_ menu: NSMenu) {
        menu.item(withTag: Tag.status.rawValue)?.title = appState.statusLine
        if let stats = menu.item(withTag: Tag.stats.rawValue) {
            stats.title = appState.statsLine
            stats.isHidden = appState.statsLine.isEmpty
        }
        menu.item(withTag: Tag.enable.rawValue)?.state = appState.enabled ? .on : .off
        let current = strength
        for entry in menu.items.compactMap(\.submenu).flatMap(\.items)
        where entry.tag == Tag.strengthBase.rawValue {
            entry.state = (entry.representedObject as? String) == current.rawValue ? .on : .off
        }
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        update(menu)
    }

    func menuDidClose(_ menu: NSMenu) {
        menus.removeAll { $0 === menu }
    }

    nonisolated func popoverDidClose(_ notification: Notification) {
        Task { @MainActor in self.panelClosedAt = Date() }
    }

    @objc private func toggleEnabled() {
        appState.enabled.toggle()
        onToggleEnabled?(appState.enabled)
        refresh()
    }

    @objc private func pickStrength(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let option = EnhancementSession.Tuning.Strength(rawValue: raw) else { return }
        strength = option
        onStrengthChanged?(option)
        refresh()
    }

    @objc private func openTestPage() {
        onOpenTestPage?()
    }

    @objc private func quit() {
        NSApplication.shared.terminate(nil)
    }
}
