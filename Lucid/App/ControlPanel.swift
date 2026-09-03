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
    var enabled: Bool = true
    var strength: EnhancementSession.Tuning.Strength = .standard
    var tuning = EnhancementSession.tuning
    var status: String = ""
    var stats: String = ""
    var enhancing: Bool = false

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
    @State private var showAdjustments = true
    @State private var showStages = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    quality
                    section("Adjustments", isOpen: $showAdjustments) { adjustments }
                    section("Stages", isOpen: $showStages) { stages }
                }
                .padding(14)
            }
            .frame(maxHeight: 430)
            Divider()
            footer
        }
        .frame(width: 320)
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 10) {
            ZStack {
                Circle()
                    .strokeBorder(Color.primary.opacity(0.35), lineWidth: 1.6)
                    .frame(width: 22, height: 22)
                Circle()
                    .fill(model.enhancing ? Color.accentColor : Color.secondary.opacity(0.45))
                    .frame(width: 7, height: 7)
            }
            VStack(alignment: .leading, spacing: 1) {
                Text("Lucid").font(.system(size: 13, weight: .semibold))
                Text(model.status.isEmpty ? "Idle" : model.status)
                    .font(.system(size: 10.5))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer(minLength: 6)
            Toggle("", isOn: Binding(
                get: { model.enabled },
                set: { model.enabled = $0; model.onEnabledChange?($0) }
            ))
            .toggleStyle(.switch)
            .labelsHidden()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
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
            if !model.stats.isEmpty {
                Text(model.stats)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
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
            Button("Test Lab") { model.onOpenLab?() }
                .buttonStyle(.link)
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
