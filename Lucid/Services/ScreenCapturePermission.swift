//
//  ScreenCapturePermission.swift
//  Lucid
//
//  Screen Recording permission state for the overlay presentation path. The
//  decoded-frame path (page canvas) needs no system permission; the
//  ScreenCaptureKit fallback that feeds the shadow-layer window cannot start
//  without it, so the coordinator checks before opening a session and prompts
//  once at launch.
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
