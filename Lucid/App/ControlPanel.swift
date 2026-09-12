//
//  ControlPanel.swift
//  Lucid
//
//  The panel behind the menu bar item. A menu closes the moment you pick
//  something, which is useless for adjusting a picture you are watching, so
//  this is a popover instead: it stays open, and every control takes effect on
//  the next frame.
//

import SwiftUI

@MainActor
@Observable
final class ControlPanelModel {
    let loginItem = LoginItemController()
    var enabled: Bool = true
    var strength: EnhancementSession.Tuning.Strength = .standard
    var tuning = EnhancementSession.tuning
    var status: String = ""
    var stats: String = ""
    var enhancing: Bool = false
    var comparing = false
    var connected = false
    var onCompare: ((Bool) -> Void)?
    func compare(_ original: Bool) { comparing = original; onCompare?(original) }

    /// Applied on every change, so a slider moves the picture as it is dragged.
    var onTuningChange: ((EnhancementSession.Tuning) -> Void)?
    var onEnabledChange: ((Bool) -> Void)?
    var onStrengthChange: ((EnhancementSession.Tuning.Strength) -> Void)?
    var onOpenLab: (() -> Void)?
    var onReset: (() -> Void)?

    func push() { onTuningChange?(tuning) }

    /// A binding that writes straight through to the running pipeline.
    func value(_ path: WritableKeyPath<EnhancementSession.Tuning, Float>) -> Binding<Double> {
        Binding(
            get: { Double(self.tuning[keyPath: path]) },
            set: { self.tuning[keyPath: path] = Float($0); self.push() }
        )
    }

    func flag(_ path: WritableKeyPath<EnhancementSession.Tuning, Float>) -> Binding<Bool> {
        Binding(
            get: { self.tuning[keyPath: path] > 0.5 },
            set: { self.tuning[keyPath: path] = $0 ? 1 : 0; self.push() }
        )
    }
}

struct ControlPanel: View {
    @Bindable var model: ControlPanelModel
    @State private var showAdjustments = false
    @State private var showStages = false

    private let accent = Color(red: 0.28, green: 0.86, blue: 0.75)
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    activity
                    quality
                    compareControl
                    startup
                    if !model.connected { connectionHelp }
                    section("Picture adjustments", isOpen: $showAdjustments) { adjustments }
                    if AppCoordinator.debugLogging { section("Developer controls", isOpen: $showStages) { stages } }
                }.padding(20)
            }.frame(maxHeight: 470)
            Divider().opacity(0.35)
            footer
        }
        .frame(width: 360)
        .background(.regularMaterial)
        .tint(accent)
        .onDisappear { model.compare(false) }
    }

    private var header: some View {
        HStack(spacing: 13) {
            Image(systemName: "camera.aperture")
                .font(.system(size: 32, weight: .ultraLight))
                .foregroundStyle(accent)
                .symbolEffect(.pulse, options: .repeating, isActive: model.enhancing)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text("Lucid").font(.system(size: 24, weight: .semibold, design: .rounded))
                Text("A clearer kind of watching.")
                    .font(.system(size: 11)).foregroundStyle(.secondary)
            }
            Spacer()
            Toggle("Enhance browser video", isOn: Binding(
                get: { model.enabled },
                set: { model.enabled = $0; model.onEnabledChange?($0) }
            )).toggleStyle(.switch).labelsHidden()
        }
        .padding(20)
        .background(LinearGradient(colors: [accent.opacity(0.12), .clear], startPoint: .topLeading, endPoint: .bottomTrailing))
    }

    private var activity: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 6) {
                Circle().fill(model.enhancing ? accent : Color.secondary).frame(width: 5, height: 5)
                Text(model.enhancing ? "ENHANCING LIVE" : model.enabled ? "READY WHEN YOU ARE" : "PAUSED")
                    .font(.system(size: 9, weight: .semibold)).tracking(1.5)
                    .foregroundStyle(model.enhancing ? accent : Color.secondary)
            }
            Text(model.status.isEmpty ? "Play a browser video to begin" : model.status)
                .font(.system(size: 13, weight: .medium)).lineLimit(2)
            if !model.stats.isEmpty {
                Text(model.stats).font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary).lineLimit(2)
            } else {
                Text("Restores compressed video locally on your Mac.")
                    .font(.system(size: 11)).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 13))
        .overlay(RoundedRectangle(cornerRadius: 13).strokeBorder(.primary.opacity(0.055)))
    }

    private var compareControl: some View {
        HStack(spacing: 8) {
            Image(systemName: "rectangle.lefthalf.inset.filled")
            Text(model.comparing ? "Original video" : "Hold to see original")
                .font(.system(size: 12, weight: .medium))
            Spacer()
            Image(systemName: "hand.draw").foregroundStyle(.secondary)
        }
        .padding(12)
        .background(model.comparing ? accent.opacity(0.18) : Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 10))
        .contentShape(RoundedRectangle(cornerRadius: 10))
        .gesture(DragGesture(minimumDistance: 0).onChanged { _ in model.compare(true) }.onEnded { _ in model.compare(false) })
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(model.comparing ? "Show enhanced video" : "Compare original video")
        .accessibilityAddTraits(.isButton)
        .accessibilityAction { model.compare(!model.comparing) }
        .disabled(!model.enhancing)
        .opacity(model.enhancing ? 1 : 0.45)
    }

    private var connectionHelp: some View {
        VStack(alignment: .leading, spacing: 7) {
            Label("Connect your browser", systemImage: "puzzlepiece.extension")
                .font(.system(size: 12, weight: .medium))
            Text("Enable the Lucid companion in Chrome, Edge, or Safari, then play a 144p–480p video. Enhancement starts automatically.")
                .font(.system(size: 11)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Button("Open companion setup") {
                if let url = Bundle.main.url(forResource: "CompanionSetup", withExtension: "html") { NSWorkspace.shared.open(url) }
            }.buttonStyle(.link).font(.system(size: 11))
        }
    }

    private var startup: some View {
        VStack(alignment: .leading, spacing: 6) {
            Toggle("Launch at login", isOn: Binding(
                get: { model.loginItem.isEnabled },
                set: { model.loginItem.setEnabled($0) }
            )).toggleStyle(.switch).font(.system(size: 12))
            Text("Keeps Lucid ready in the menu bar when you sign in.")
                .font(.system(size: 11)).foregroundStyle(.secondary)
            if model.loginItem.requiresApproval {
                Text("Allow Lucid in macOS Login Items to finish enabling startup.")
                    .font(.system(size: 11)).foregroundStyle(.secondary)
                Button("Open Login Items…") { model.loginItem.showSettings() }
                    .buttonStyle(.link).font(.system(size: 11))
            }
            if let error = model.loginItem.errorMessage {
                Text(error).font(.system(size: 11)).foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - Quality

    private var quality: some View {
        VStack(alignment: .leading, spacing: 6) {
            label("Quality")
            Picker("", selection: Binding(
                get: { model.strength },
                set: { model.strength = $0; model.onStrengthChange?($0) }
            )) {
                ForEach(EnhancementSession.Tuning.Strength.allCases, id: \.self) { option in
                    Text(option.label).tag(option)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            Text(model.strength.detail).font(.system(size: 11)).foregroundStyle(.secondary)

        }
    }

    // MARK: - Adjustments

    private var adjustments: some View {
        VStack(spacing: 9) {
            slider("Sharpen",  model.value(\.sharpness),  0, 1.6,  "How hard edges are pulled up")
            slider("Detail",   model.value(\.fine),       0, 2.0,  "Gain on fine texture")
            slider("Deblock",  model.value(\.sourceDeblock), 0, 0.08, "Smooths compression blocking before scaling")
            Divider().padding(.vertical, 2)
            slider("Black",    model.value(\.blackPoint), 0, 0.10, "Puts crushed blacks back where they belong")
            slider("Contrast", model.value(\.contrast),   0, 0.50, "S-curve strength")
            slider("Colour",   model.value(\.saturation), 0.8, 1.4, "Saturation, applied in Oklab when that stage is on")
        }
    }

    // MARK: - Stages

    private var stages: some View {
        VStack(alignment: .leading, spacing: 7) {
            stage("Chroma siting", model.flag(\.stageSiting),
                  "4:2:0 chroma is left-sited; without this colour lands half a pixel off")
            stage("Motion alignment", model.flag(\.stageMotion), "Reprojects history and rejects unreliable motion")
            stage("Temporal", model.flag(\.stageTaa),
                  "Steadies compression noise between frames")
            stage("Oklab colour", model.flag(\.stageOklab),
                  "Saturation that does not also change brightness")
            stage("Deblocking filter", model.flag(\.stageLoopFilter),
                  "The H.264 loop filter, run without the bitstream")
            stage("Dering", model.flag(\.stageCdef),
                  "AV1's CDEF, for mosquito noise around edges")
            stage("Debanding", model.flag(\.stageDeband),
                  "Breaks up banding in skies and fades, and adds fine grain")
        }
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: 10) {
            Button("Reset") { model.onReset?() }
                .buttonStyle(.link)
            if AppCoordinator.debugLogging {
                Button("Test Lab") { model.onOpenLab?() }.buttonStyle(.link)
            }
            Spacer()
            Button("Quit") { NSApplication.shared.terminate(nil) }
                .buttonStyle(.link)
                .foregroundStyle(.secondary)
        }
        .font(.system(size: 11))
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
    }

    // MARK: - Pieces

    private func label(_ text: String) -> some View {
        Text(text.uppercased())
            .font(.system(size: 9.5, weight: .semibold))
            .kerning(0.6)
            .foregroundStyle(.secondary)
    }

    private func section<Content: View>(
        _ title: String, isOpen: Binding<Bool>, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.easeOut(duration: 0.16)) { isOpen.wrappedValue.toggle() }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 8, weight: .bold))
                        .rotationEffect(.degrees(isOpen.wrappedValue ? 90 : 0))
                    label(title)
                    Spacer()
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if isOpen.wrappedValue { content() }
        }
    }

    private func slider(_ name: String, _ value: Binding<Double>,
                        _ low: Double, _ high: Double, _ help: String) -> some View {
        HStack(spacing: 8) {
            Text(name)
                .font(.system(size: 11))
                .frame(width: 58, alignment: .leading)
            Slider(value: value, in: low...high)
                .controlSize(.small)
            Text(format(value.wrappedValue))
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 34, alignment: .trailing)
        }
        .help(help)
    }

    private func stage(_ name: String, _ on: Binding<Bool>, _ help: String) -> some View {
        Toggle(isOn: on) {
            Text(name).font(.system(size: 11))
        }
        .toggleStyle(.checkbox)
        .help(help)
    }

    private func format(_ value: Double) -> String {
        value >= 1 ? String(format: "%.2f", value) : String(format: "%.3f", value)
    }
}
