//
//  ShadowLayerWindow.swift
//  Lucid
//
//  The presentation surface: a borderless, click-through, shadowless window
//  that is placed exactly over the browser's video box and stacked directly
//  above the browser window. It is excluded from screen sharing, Mission
//  Control and window cycling, and keeps its alpha just below 1.0 so the
//  browser never sees itself as occluded (both Chrome and Safari suspend
//  rendering when fully occluded, which would starve capture).
//

import AppKit
import QuartzCore

final class ShadowLayerWindow: NSWindow {
    /// Below 1.0 so WindowServer and the browser treat the window as non-occluding.
    static let visibleAlpha: CGFloat = 0.995

    private let hostView: ShadowLayerView

    init(presenter: FramePresenter) {
        hostView = ShadowLayerView(presenterLayer: presenter.layer)
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 16, height: 9),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        level = .normal
        isOpaque = false
        backgroundColor = .clear
        hasShadow = false
        ignoresMouseEvents = true
        isExcludedFromWindowsMenu = true
        isReleasedWhenClosed = false
        animationBehavior = .none
        // Excluded from screenshots and screen sharing unless a developer opts in.
        sharingType = ProcessInfo.processInfo.environment["LUCID_CAPTURABLE"] == "1" ? .readOnly : .none
        alphaValue = 0
        collectionBehavior = [.moveToActiveSpace, .fullScreenAuxiliary, .transient, .ignoresCycle, .stationary]
        contentView = hostView
    }

    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    /// AppKit normally keeps windows below the menu bar and on screen; the
    /// shadow layer must sit exactly where the video is, wherever that is.
    override func constrainFrameRect(_ frameRect: NSRect, to screen: NSScreen?) -> NSRect { frameRect }

    func setVisible(_ visible: Bool) {
        let target = visible ? Self.visibleAlpha : 0
        if alphaValue != target { alphaValue = target }
    }

    /// Cut holes for page elements (controls, captions) so they stay live.
    /// Rects are in this window's content coordinates (AppKit, bottom-left origin).
    func setCutouts(_ rects: [NSRect], cornerRadius: CGFloat) {
        hostView.apply(cutouts: rects, cornerRadius: cornerRadius)
    }

    /// Screenshots exclude the overlay by default; the comparison harness
    /// needs it included.
    func setCapturable(_ capturable: Bool) {
        sharingType = capturable ? .readOnly : .none
    }

    func updateBackingScale(_ scale: CGFloat) {
        hostView.updateContentsScale(scale)
    }

    /// Places the enhanced image inside the window. It may extend past the
    /// window bounds (the captured margin) and is clipped.
    func setImageFrame(_ frame: NSRect) {
        hostView.setImageFrame(frame)
    }
}

final class ShadowLayerView: NSView {
    private let presenterLayer: CALayer
    private var maskLayer: CAShapeLayer?

    init(presenterLayer: CALayer) {
        self.presenterLayer = presenterLayer
        super.init(frame: .zero)
        wantsLayer = true
        layerContentsRedrawPolicy = .never
        layer?.masksToBounds = true
        layer?.backgroundColor = nil
        presenterLayer.frame = bounds
        layer?.addSublayer(presenterLayer)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { nil }

    override var isFlipped: Bool { false }

    private var imageFrame: NSRect?

    override func layout() {
        super.layout()
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        presenterLayer.frame = imageFrame ?? bounds
        maskLayer?.frame = presenterLayer.bounds
        CATransaction.commit()
    }

    func setImageFrame(_ frame: NSRect) {
        guard imageFrame != frame else { return }
        imageFrame = frame
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        presenterLayer.frame = frame
        maskLayer?.frame = presenterLayer.bounds
        CATransaction.commit()
    }

    func updateContentsScale(_ scale: CGFloat) {
        layer?.contentsScale = scale
        presenterLayer.contentsScale = scale
    }

    func apply(cutouts: [NSRect], cornerRadius: CGFloat) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        defer { CATransaction.commit() }
        if cutouts.isEmpty && cornerRadius <= 0 {
            if maskLayer != nil {
                presenterLayer.mask = nil
                maskLayer = nil
            }
            return
        }
        let mask = maskLayer ?? {
            let shape = CAShapeLayer()
            shape.fillRule = .evenOdd
            shape.fillColor = NSColor.black.cgColor
            maskLayer = shape
            presenterLayer.mask = shape
            return shape
        }()
        let imageBounds = presenterLayer.bounds
        let offset = CGPoint(x: presenterLayer.frame.minX, y: presenterLayer.frame.minY)
        mask.frame = imageBounds
        let path = CGMutablePath()
        if cornerRadius > 0 {
            path.addRoundedRect(in: imageBounds, cornerWidth: cornerRadius, cornerHeight: cornerRadius)
        } else {
            path.addRect(imageBounds)
        }
        for rect in cutouts {
            let shifted = rect.offsetBy(dx: -offset.x, dy: -offset.y)
            let clipped = shifted.intersection(imageBounds)
            guard !clipped.isNull, clipped.width > 0, clipped.height > 0 else { continue }
            path.addRect(clipped)
        }
        mask.path = path
    }
}
