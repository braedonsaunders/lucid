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

@MainActor
final class MenuBarController: NSObject, NSMenuDelegate {
    private var statusItem: NSStatusItem?
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
        item.button?.image = NSImage(systemSymbolName: "sparkles.tv", accessibilityDescription: "Lucid")
        item.button?.image?.isTemplate = true
        item.menu = makeMenu()
        statusItem = item
        refresh()
    }

    /// Builds a fresh copy of the menu. Called once for the menu bar and again
    /// each time the Dock asks for one.
    func makeMenu() -> NSMenu {
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
        let quit = NSMenuItem(title: "Quit Lucid", action: #selector(quit), keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)

        menus.append(menu)
        update(menu)
        return menu
    }

    private enum Tag: Int {
        case status = 1, stats = 2, enable = 3, strengthBase = 4
    }

    func refresh() {
        for menu in menus { update(menu) }
        statusItem?.button?.appearsDisabled = !appState.enabled
        let symbol = appState.isEnhancing ? "sparkles.tv.fill" : "sparkles.tv"
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
