//
//  ScreenCapturePermission.swift
//  Lucid
//
//  Screen Recording permission for the overlay / ScreenCaptureKit fallback.
//  The decoded-frame path (page canvas via the extension iframe) needs no
//  system permission and must not prompt at launch. The coordinator only
//  asks when that path has not produced frames and capture is about to start.
//

import ScreenCaptureKit

/// Point of contact for Screen Recording permission.
enum ScreenCapturePermission {
    /// Whether a capture stream may be started right now. No prompt.
    static var isGranted: Bool {
        CGPreflightScreenCaptureAccess()
    }

    /// Brings up the system prompt when access is missing. Returns the state
    /// after the user answers. Safe to call when access is already granted.
    @discardableResult
    static func request() async -> Bool {
        guard !isGranted else { return true }
        return CGRequestScreenCaptureAccess()
    }
}
