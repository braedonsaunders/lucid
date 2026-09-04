//
//  AppCoordinator.swift
//  Lucid
//
//  Owns the browser bridge and at most one enhancement session. A session is
//  started automatically when a browser reports a playing video that is worth
//  enhancing, follows it while it moves, and stops when it goes away.
//

import AppKit
import CoreMedia
import Foundation

@MainActor
@Observable
final class AppCoordinator {
    static let shared = AppCoordinator()

    let appState = AppState()
    private var menuBar: MenuBarController?
    private var bridge: BrowserBridgeServer?
    fileprivate var session: EnhancementSession?
    private var reports: [String: (report: BrowserVideoReport, received: ContinuousClock.Instant)] = [:]
    private var sweepTimer: Timer?
    private var loggedSessions: Set<String> = []
    /// When the running session's page first went quiet, so a brief gap does
    /// not tear the pipeline down.
    private var sessionMissingSince: ContinuousClock.Instant?
    /// There is one pipeline, and it is `.learned`. This stays a variable only
    /// so the offline comparison shooter can step through the other paths to
    /// produce its stills; nothing in the shipping app changes it, and it is
    /// deliberately not remembered between launches.
    var engine: EngineKind = .learned

    private var latencySeconds: Double {
        let stored = UserDefaults.standard.double(forKey: "latencySeconds")
        return stored == 0 ? 0.040 : stored
    }

    static let staleReportAge: Duration = .seconds(3)
    static let debugLogging = ProcessInfo.processInfo.environment["LUCID_DEBUG"] == "1"

    private init() {
        let menuBar = MenuBarController(appState: appState)
        menuBar.onToggleEnabled = { [weak self] enabled in
            guard let self else { return }
            if !enabled { self.stopSession(reason: "Paused") } else { self.evaluate() }
            self.broadcastStatus()
        }
        menuBar.onOpenTestPage = { [weak self] in self?.openTestPage() }
        // Live edits from the control panel take effect on the next frame.
        menuBar.onTuningChanged = { [weak self] tuning in
            EnhancementSession.tuning = tuning
            TiledVideoToolboxUpscaler.chromaSitingLeft = tuning.stageSiting > 0.5
            self?.session?.reloadTuning()
            self?.broadcastStatus()
        }
        menuBar.onResetTuning = { [weak self] in
            EnhancementSession.tuning = EnhancementSession.Tuning.load()
            self?.session?.reloadTuning()
            self?.broadcastStatus()
        }
        menuBar.onStrengthChanged = { [weak self] strength in
            guard let self else { return }
            var tuning = EnhancementSession.tuning
            tuning.apply(strength)
            EnhancementSession.tuning = tuning
            self.session?.reloadTuning()
            self.broadcastStatus()
        }
        self.menuBar = menuBar
        var startup = EnhancementSession.tuning
        startup.apply(menuBar.strength)
        EnhancementSession.tuning = startup
        startBridge()
        let timer = Timer(timeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.sweepStaleReports()
                if self?.session == nil { self?.broadcastStatus() }
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        sweepTimer = timer
    }

    // MARK: - Bridge

    /// Pushes an enhanced frame out to the pages.
    fileprivate func sendEnhanced(_ packet: Data, session: String) {
        bridge?.send(binary: packet, to: session)
    }

    private func startBridge() {
        let bridge = BrowserBridgeServer(
            onReport: { report in
                Task { @MainActor in AppCoordinator.shared.handle(report) }
            },
            onDisconnect: { sessions in
                Task { @MainActor in AppCoordinator.shared.handleDisconnect(sessions) }
            },
            onControl: { control in
                Task { @MainActor in AppCoordinator.shared.handleControl(control) }
            }
        )
        // Decoded frames straight from the page, at the video's own resolution.
        bridge.onFrame = { frame in
            Task { @MainActor in
                let coordinator = AppCoordinator.shared
                if AppCoordinator.debugLogging, coordinator.framesSeen == 0 {
                    print("   🎞 first decoded frame: session \(frame.header.session.prefix(8)) \(frame.header.w)x\(frame.header.h) \(frame.header.format), app session \(coordinator.session?.id.prefix(8) ?? "none")")
                }
                coordinator.framesSeen += 1
                coordinator.noteFrameArrived()
                coordinator.session?.accept(frame)
            }
        }
        do {
            try bridge.start()
            self.bridge = bridge
        } catch {
            appState.lastError = "Bridge failed to start: \(error.localizedDescription)"
            appState.statusLine = "Bridge unavailable"
            menuBar?.refresh()
        }
    }

    private func handle(_ report: BrowserVideoReport) {
        appState.connectedBrowsers.insert(report.browser)
        switch report.type {
        case .hello:
            return
        case .gone:
            if Self.debugLogging, reports[report.session] != nil {
                print("   👋 page says gone: \(report.session.prefix(8))")
            }
            reports.removeValue(forKey: report.session)
        case .video:
            reports[report.session] = (report, .now)
            if let session, session.id == report.session { session.setPageRenders(report.draws == true) }
            if loggedSessions.insert(report.session).inserted, let video = report.video {
                print("   📺 \(report.browser) session \(report.session.prefix(8)): \"\(report.title.prefix(60))\" video \(video.iw)x\(video.ih) box \(Int(video.rect.w))x\(Int(video.rect.h))@\(Int(video.rect.x)),\(Int(video.rect.y)) dpr \(report.dpr) window \(Int(report.screenX)),\(Int(report.screenY)) outer \(Int(report.outerWidth))x\(Int(report.outerHeight)) inner \(Int(report.innerWidth))x\(Int(report.innerHeight)) paused=\(video.paused) enhanceable=\(Self.isEnhanceable(report))")
            }
        }
        evaluate()
    }

    private var lastStats = PipelineStats()

    fileprivate var framesSeen = 0
    private var frameArrivals: [ContinuousClock.Instant] = []
    fileprivate var arrivalRate = 0

    /// Frames per second arriving from the page, so a shortfall can be pinned
    /// on the page or on the pipeline rather than guessed at.
    fileprivate func noteFrameArrived() {
        let now = ContinuousClock.now
        frameArrivals.append(now)
        frameArrivals.removeAll { now - $0 > .seconds(1) }
        arrivalRate = frameArrivals.count
    }
    private var shooter: ComparisonShooter?

    /// Captures the same frame once per engine. The page should pause the video
    /// first so every shot is the identical frame.
    private func shoot(into name: String) {
        guard shooter == nil, let session else { return }
        let box = session.videoScreenRect
        guard box.width > 4, box.height > 4 else { return }
        let shooter = ComparisonShooter()
        self.shooter = shooter
        let directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/shots/\(name)")
        let previous = engine
        Task { @MainActor in
            _ = await shooter.run(
                box: box,
                browserWindowID: session.windowID,
                overlayWindowID: { [weak self] in self?.session?.overlayWindowID ?? 0 },
                directory: directory,
                engines: EngineKind.allCases,
                setEngine: { [weak self] kind in
                    guard let self else { return }
                    self.engine = kind
                    self.session?.setEngine(kind)
                    // A paused video presents no new frames; ask for one so the
                    // overlay re-renders with the engine just selected.
                    self.bridge?.broadcast(BridgeNudge())
                    try? await Task.sleep(for: .milliseconds(120))
                    self.bridge?.broadcast(BridgeNudge())
                },
                setOverlayVisible: { [weak self] visible in
                    self?.session?.setCapturable(visible)
                    self?.session?.setOverlayHidden(!visible)
                }
            )
            self.session?.setCapturable(false)
            self.session?.setOverlayHidden(false)
            self.engine = previous
            self.session?.setEngine(previous)
            self.shooter = nil
            self.broadcastStatus()
        }
    }

    private func handleControl(_ control: BridgeControl) {
        if let enabled = control.enabled, enabled != appState.enabled {
            appState.enabled = enabled
            if !enabled { stopSession(reason: "Paused") } else { evaluate() }
        }
        if let changes = control.tuning, !changes.isEmpty {
            var t = EnhancementSession.tuning
            for (key, value) in changes {
                switch key {
                case "sharpness": t.sharpness = value
                case "fine": t.fine = value
                case "deblock": t.deblock = value
                case "sourceDeblock": t.sourceDeblock = value
                case "temporal": t.temporal = value
                case "blackPoint": t.blackPoint = value
                case "whitePoint": t.whitePoint = value
                case "contrast": t.contrast = value
                case "saturation": t.saturation = value
                case "micro": t.micro = value
                case "lobeScale": t.lobeScale = value
                case "mid": t.mid = value
                case "presharpen": t.presharpen = value
                case "adaptive": t.adaptive = value
                case "skinProtect": t.skinProtect = value
                case "taaFeedback": t.taaFeedback = value
                case "taaGamma": t.taaGamma = value
                case "grain": t.grain = value
                case "debandThreshold": t.debandThreshold = value
                case "cdefSecondary": t.cdefSecondary = value
                case "cdefPrimary": t.cdefPrimary = value
                case "loopFilterQuant": t.loopFilterQuant = value
                case "stageCdef": t.stageCdef = value
                case "stageTaa": t.stageTaa = value
                case "stageLoopFilter": t.stageLoopFilter = value
                case "stageOklab": t.stageOklab = value
                case "stageDeband": t.stageDeband = value
                case "stageSiting": t.stageSiting = value
                default: break
                }
            }
            EnhancementSession.tuning = t
            TiledVideoToolboxUpscaler.chromaSitingLeft = t.stageSiting > 0.5
            session?.reloadTuning()
        }
        if control.resetTuning == true {
            EnhancementSession.tuning = EnhancementSession.Tuning.load()
            session?.reloadTuning()
        }
        if let name = control.shoot, !name.isEmpty {
            // The shoot control drives Screen Recording to write browser
            // stills to disk. Auth stops a website; this stops a same-user
            // process that stole the token from turning that TCC grant into
            // a screenshot service. Bench tools opt in with LUCID_SHOOT=1.
            let allowed = Self.debugLogging
                || ProcessInfo.processInfo.environment["LUCID_SHOOT"] == "1"
            if allowed {
                shoot(into: name)
            } else {
                print("   🔒 shoot refused — set LUCID_SHOOT=1 to allow comparison stills")
            }
        }
        if let latency = control.latency, latency > 0 {
            UserDefaults.standard.set(latency, forKey: "latencySeconds")
            session?.presenter.latency = CMTime(seconds: latency, preferredTimescale: 1_000_000_000)
        }
        menuBar?.refresh()
        broadcastStatus()
    }

    /// A fresh menu for the Dock; AppKit asks each time it is opened.
    func dockMenu() -> NSMenu? { menuBar?.makeMenu(retained: false) }

    private func broadcastStatus() {
        bridge?.broadcast(BridgeStatus(
            enabled: appState.enabled,
            enhancing: appState.isEnhancing,
            engine: engine.rawValue,
            engines: (EngineKind.comparisonPathEnabled ? EngineKind.allCases : EngineKind.shipping).map(\.rawValue),
            engineLabels: (EngineKind.comparisonPathEnabled ? EngineKind.allCases : EngineKind.shipping).map(\.label),
            tuning: EnhancementSession.tuningDictionary,
            status: appState.statusLine,
            stats: appState.statsLine,
            latency: latencySeconds,
            sourceFPS: lastStats.sourceFPS,
            outputFPS: lastStats.outputFPS,
            processingMilliseconds: lastStats.processingMilliseconds,
            tileCount: lastStats.tileCount,
            outputWidth: Int(lastStats.outputSize.width),
            outputHeight: Int(lastStats.outputSize.height),
            error: appState.lastError
        ))
    }

    private func handleDisconnect(_ sessions: Set<String>) {
        for id in sessions { reports.removeValue(forKey: id) }
        evaluate()
    }

    private func sweepStaleReports() {
        let now = ContinuousClock.now
        let stale = reports.filter { now - $0.value.received > Self.staleReportAge }.map(\.key)
        guard !stale.isEmpty else { return }
        if Self.debugLogging { print("   🧹 stale reports swept: \(stale.map { $0.prefix(8) })") }
        for id in stale { reports.removeValue(forKey: id) }
        evaluate()
    }

    // MARK: - Session policy

    private func evaluate() {
        guard appState.enabled, !stopping else { return }

        if let session {
            if let current = reports[session.id]?.report, Self.isEnhanceable(current, keeping: true) {
                sessionMissingSince = nil
                session.update(report: current)
                return
            }
            // A page that goes quiet for a moment - a reconfiguration, a
            // layout settle - should not cost a pipeline teardown and rebuild.
            if reports[session.id] == nil {
                if sessionMissingSince == nil { sessionMissingSince = .now }
                if let since = sessionMissingSince, ContinuousClock.now - since < .milliseconds(1500) { return }
            }
            if Self.debugLogging, let r = reports[session.id]?.report {
                let v = r.video
                print("   🔎 session dropped: visible=\(r.visible) type=\(r.type) ended=\(v?.ended ?? true) pip=\(v?.pip ?? true) iw=\(v?.iw ?? 0) ih=\(v?.ih ?? 0) rect=\(Int(v?.rect.w ?? 0))x\(Int(v?.rect.h ?? 0)) dpr=\(r.dpr)")
            } else if Self.debugLogging {
                print("   🔎 session dropped: no report for \(session.id.prefix(8))")
            }
            sessionMissingSince = nil
            stopSession(reason: "Video ended")
        }

        let candidates = reports.values.map(\.report).filter { Self.isEnhanceable($0) }
        guard let best = candidates.max(by: { area($0) < area($1) }) else {
            if appState.statusLine != "Waiting for browser video" {
                appState.statusLine = reports.isEmpty ? "Waiting for browser video" : "Video not enhanceable"
                menuBar?.refresh()
            }
            return
        }
        startSession(for: best)
    }

    private func area(_ report: BrowserVideoReport) -> Double {
        guard let video = report.video else { return 0 }
        return video.rect.w * video.rect.h
    }

    /// Only playing, visible, sub-1080p video that is displayed larger than its
    /// decoded size benefits from 2× reconstruction. Everything else is left alone.
    static func isEnhanceable(_ report: BrowserVideoReport, keeping: Bool = false) -> Bool {
        guard report.type == .video, report.visible, let video = report.video else { return false }
        guard !video.ended, !video.pip else { return false }
        // The window Lucid works in, and the reasons for each end of it.
        //
        // The floor is low on purpose: 144p is the case that needs help most,
        // and the old 320x180 floor rejected it outright. Below about 128 wide
        // there is not enough left to reconstruct from.
        //
        // The ceiling is not a number chosen here. It is wherever the learned
        // upscaler stops fitting a frame - see LearnedUpscaler.variants - and
        // today that is 640x360. Above it Lucid does nothing rather than
        // reaching for a weaker upscaler, because the weaker upscaler measured
        // worse than leaving the frame alone. This is also the point of
        // diminishing returns: the larger the source, the less there is to
        // recover, which is why RTX Video Super Resolution has a window too.
        guard video.iw >= 128, video.ih >= 72 else { return false }
        guard LearnedUpscaler.supports(width: video.iw, height: video.ih) else { return false }
        guard video.rect.w >= 200, video.rect.h >= 100 else { return false }
        let physicalWidth = video.rect.w * report.dpr
        // Hysteresis: start above 1.15×, but keep a running session down to 1.0×.
        return physicalWidth > Double(video.iw) * (keeping ? 1.0 : 1.15)
    }

    private func startSession(for report: BrowserVideoReport) {
        guard session == nil else { return }
        let snapshot = WindowSnapshot.capture()
        guard let window = WindowTracker.matchWindow(for: report, in: snapshot) else {
            if appState.statusLine != "Browser window not found" {
                print("   🔍 No on-screen \(report.browser) window matches \"\(report.title)\" at \(Int(report.screenX)),\(Int(report.screenY)) \(Int(report.outerWidth))x\(Int(report.outerHeight))")
            }
            appState.statusLine = "Browser window not found"
            menuBar?.refresh()
            return
        }
        do {
            let session = try EnhancementSession(
                report: report, window: window, latencySeconds: latencySeconds, engine: engine,
                onStats: { stats in
                    Task { @MainActor in AppCoordinator.shared.updateStats(stats) }
                },
                onEnded: { [weak self] reason in self?.sessionEnded(reason) }
            )
            self.session = session
            print("   🎯 Session \(report.session.prefix(8)): \(report.browser) window [\(window.id)] video \(report.video!.iw)x\(report.video!.ih) at \(report.video!.rect)")
            appState.isEnhancing = true
            // 4×, not 2×: the tiled 2×-per-pass scaler was replaced by SPAN and
            // this is the line users actually read.
            appState.statusLine = "Enhancing \(report.browser.capitalized) video \(report.video!.iw)×\(report.video!.ih) → 4×"
            appState.lastError = nil
            menuBar?.refresh()
            broadcastStatus()
            Task { await session.start() }
        } catch {
            print("   ❌ Session \(report.session.prefix(8)) could not start: \(error)")
            appState.statusLine = "Cannot enhance this video"
            appState.lastError = "\(error)"
            menuBar?.refresh()
        }
    }

    /// True while a previous session releases its capture and VideoToolbox
    /// sessions; the next session only starts once that has finished.
    private var stopping = false

    private func stopSession(reason: String) {
        guard let session else { return }
        print("   ⏹ Session \(session.id.prefix(8)) stopped: \(reason)")
        self.session = nil
        appState.isEnhancing = false
        appState.statusLine = reason
        appState.statsLine = ""
        menuBar?.refresh()
        lastStats = PipelineStats()
        broadcastStatus()
        stopping = true
        Task { @MainActor [weak self] in
            await session.stop()
            self?.stopping = false
            self?.evaluate()
        }
    }

    private func sessionEnded(_ reason: String) {
        stopSession(reason: reason)
        evaluate()
    }

    private func updateStats(_ stats: PipelineStats) {
        appState.statsLine = String(
            format: "%.0f → %.0f fps · %.1f ms · %d tile%@ · %.0f×%.0f",
            stats.sourceFPS, stats.outputFPS, stats.processingMilliseconds, stats.tileCount,
            stats.tileCount == 1 ? "" : "s", stats.outputSize.width, stats.outputSize.height
        )
        if let error = stats.lastError { appState.lastError = error }
        if AppCoordinator.debugLogging {
            print("   📊 page \(arrivalRate)/s · \(appState.statsLine)\(stats.lastError.map { " · ⚠️ \($0)" } ?? "")")
        }
        lastStats = stats
        menuBar?.refresh()
        broadcastStatus()
    }

    /// The lab page lives in the source tree next to the built app
    /// (`.build/DerivedData/Build/Products/<config>/`), or wherever
    /// the `testPagePath` default points.
    private func openTestPage() {
        var candidates: [URL] = []
        if let custom = UserDefaults.standard.string(forKey: "testPagePath") {
            candidates.append(URL(fileURLWithPath: custom))
        }
        var root = Bundle.main.bundleURL
        for _ in 0..<6 {
            root.deleteLastPathComponent()
            candidates.append(root.appendingPathComponent("TestSite/lab.html"))
        }
        if let page = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) {
            NSWorkspace.shared.open(page)
        } else {
            appState.lastError = "Test page not found; set the testPagePath default to TestSite/lab.html"
            menuBar?.refresh()
        }
    }

    /// Set when the user declines Screen Recording this launch. Decoded-frame
    /// sessions still start; the capture fallback will not keep re-prompting.
    fileprivate var screenCaptureDenied = false

    fileprivate func noteScreenCaptureDenied() {
        screenCaptureDenied = true
        appState.lastError = "Grant Screen Recording to Lucid in System Settings → Privacy & Security."
        menuBar?.refresh()
    }
}

/// One enhanced video: capture → pipeline → shadow layer, glued to one browser window.
///
/// The capture stream is created once and lives as long as the session. macOS
/// shows a capture indicator on the captured window whenever a stream starts,
/// so restarting it for every size or engine change made that indicator
/// strobe. Instead the stream is reconfigured in place and the processing
/// stages are rebuilt lazily when the frame size or engine changes.
@MainActor
final class EnhancementSession {
    let id: String
    let windowID: CGWindowID
    let pid: pid_t
    let presenter: FramePresenter
    private(set) var engine: EngineKind
    private let window: ShadowLayerWindow
    private let controller: ShadowLayerController
    private let capture: CaptureSession
    /// Frames handed over by the page, at the video's own resolution. Preferred
    /// over reading the screen, which only ever shows an already-stretched copy.
    private let decoded = DecodedFrameSource()
    private let sender = EnhancedFrameSender()
    private var sentFrames = 0
    /// True while the page is drawing the result itself. The overlay window is
    /// only used when the page cannot: then it is the fallback, not the plan.
    private(set) var pageRenders = false
    private var usingDecodedFrames = false
    private var decodedTask: Task<Void, Never>?
    private var lastDecodedFrame = ContinuousClock.now
    private var screenCaptureRunning = false
    private var rejectedFrames = 0
    private let pipeline: EnhancementPipeline
    private let backingScale: CGFloat
    private var report: BrowserVideoReport
    private var stopped = false
    private var captureRunning = false
    private var pendingRect: CGRect?
    private var resizeTask: Task<Void, Never>?
    private let onEnded: (String) -> Void

    enum Failure: Error { case noVideo, tooSmall, comparisonEngineDisabled }

    /// How many chained 2× passes to run. Picks the power of two nearest the
    /// stretch, not the next one up: with the detail gains normalised across
    /// radii an oversized render is no longer harmful, only wasteful (a 5x
    /// stretch rendered at 8x draws ~2.7x the pixels the box needs), so
    /// rendering past the box buys nothing. This is a performance and cost
    /// fix, not a quality one: the output lands within half a pass of the
    /// box either way, and downscaling a slightly smaller image stays sharp.
    nonisolated static func stageCount(for report: BrowserVideoReport) -> Int {
        guard let video = report.video, video.iw > 0 else { return 1 }
        let stretch = video.rect.w * report.dpr / Double(video.iw)
        return min(max(Int(log2(max(stretch, 1)).rounded()), 1), 3)
    }

    /// Physical width of the video box. The sender uses this as the box, not
    /// as a hard output size: NV12 within 1.5× of the box is sent whole
    /// (plane copy) because downscaling costs milliseconds and discards
    /// detail. Bounded so a full-screen 4K box cannot ask for more than the
    /// bridge can carry at frame rate.
    nonisolated static func deliveryWidth(for report: BrowserVideoReport) -> Int {
        if let forced = ProcessInfo.processInfo.environment["LUCID_MAXW"].flatMap(Int.init) { return forced }
        guard let video = report.video else { return 1280 }
        let physical = Int((video.rect.w * report.dpr).rounded(.up))
        return min(max(physical, 640), 3840)
    }

    /// Detail reconstruction strength and scale.
    /// Everything the tuner is allowed to change, in one place. Loaded from a
    /// JSON file when `LUCID_TUNING` points at one, then overridden by
    /// environment variables, so a search can sweep without rebuilding.
    struct Tuning: Codable, Sendable {
        // Settled by hand against real clips and then checked against ground
        // truth. Mirrors Tools/tuning.json, which the offline tuner writes.
        // Off by default. Measured against a plain Lanczos anchor, the
        // sharpening stage cost 0.021 of fine-band correlation on its own and
        // turned the scaler's invented foliage texture into something crunchy.
        // Strong turns it back on for anyone who wants it.
        var sharpness: Float = 0.0
        // 0.3, measured. On the bicubic-trained model any detail gain
        // amplified invented texture and cost fine-band correlation; the model
        // trained on real codec degradation gives a clean enough base that a
        // little gain now recovers rather than exaggerates. 0.5 overshoots.
        var fine: Float = 0.3
        // The post-upscale deblock is flat to four decimals across its whole
        // range now and every non-zero value is very slightly worse, so it is
        // off: the model removes blocking itself.
        var deblock: Float = 0.0
        // Off. The model is trained on real codec degradation and removes
        // blocking itself, so cleaning the input first only softens what it is
        // about to reconstruct from. Perceptual ablation on the L1 ch32u
        // (bbb-360p-350k): turning this off wins DISTS +0.0097, LPIPS +0.0102
        // and detail energy +0.0150, with no column disagreeing. The GAN
        // checkpoint at 4000 is if anything more so (DISTS +0.0087, and
        // LPIPS improved +0.0091 → +0.0148). It was wrong on ch28 too,
        // just less so.
        var sourceDeblock: Float = 0.0
        var temporal: Float = 0.50
        var blackPoint: Float = 0.020
        var whitePoint: Float = 0.975
        // 0.2: buys fine-band correlation (0.2866 -> 0.2886) and spends PSNR
        // (26.39 -> 25.71). A deliberate trade, chosen by eye, not measured -
        // the reference has no grade so no metric can pick this.
        var contrast: Float = 0.200
        // Exactly 1.0 skips the Oklab pass entirely, which is worth 1.0 ms of
        // the frame. Saturation measured fidelity-neutral across its whole
        // range, so this is a millisecond for nothing.
        var saturation: Float = 1.000
        /// 2 runs the scaler twice, 4 runs it once at its native 4x factor.
        var scalerFactor: Float = 2
        var micro: Float = 0.0
        // Half the detail scale: measured against ground truth this halves
        // coarse-band overshoot, which is what reads as halos.
        var lobeScale: Float = 0.3
        var mid: Float = 0.0
        var presharpen: Float = 0.0
        /// Lets the deblocker read the frame and set its own strength.
        var adaptive: Float = 1.0
        // Stage toggles, 0 or 1, so each can be judged on its own.
        var stageSiting: Float = 1
        // On. The stage adds the same high-frequency energy regardless of
        // the model underneath (+0.0833 L1 vs +0.0836 GAN on bbb-360p-350k),
        // and that work still wins DISTS +0.0133 and BRISQUE +17.7 on the
        // GAN checkpoint at 4000. Smaller than the L1 table (DISTS +0.0175,
        // LPIPS +0.0196) because the better model needs less help — LPIPS
        // almost vanishes (+0.0010) while DISTS still credits the texture.
        // Costs nothing measurable. CDEF and the loop filter stay off.
        var stageDeband: Float = 1
        var stageOklab: Float = 1
        var stageLoopFilter: Float = 0
        var stageTaa: Float = 1
        var stageCdef: Float = 0
        var loopFilterQuant: Float = 32
        var cdefPrimary: Float = 4
        var cdefSecondary: Float = 2
        var debandThreshold: Float = 0.008
        var grain: Float = 0.010
        var taaGamma: Float = 1.25
        // The gentlest setting the control offers: enough to steady the image
        // without the history dominating it.
        var taaFeedback: Float = 0.50
        var skinProtect: Float = 1.0

        init() {}

        /// Synthesised decoding treats every key as required, so a file written
        /// before a knob existed would throw and silently drop the whole set
        /// back to defaults. Each key is optional instead.
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            func f(_ key: CodingKeys, _ fallback: Float) throws -> Float {
                try c.decodeIfPresent(Float.self, forKey: key) ?? fallback
            }
            sharpness = try f(.sharpness, sharpness)
            fine = try f(.fine, fine)
            deblock = try f(.deblock, deblock)
            sourceDeblock = try f(.sourceDeblock, sourceDeblock)
            temporal = try f(.temporal, temporal)
            blackPoint = try f(.blackPoint, blackPoint)
            whitePoint = try f(.whitePoint, whitePoint)
            contrast = try f(.contrast, contrast)
            saturation = try f(.saturation, saturation)
            scalerFactor = try f(.scalerFactor, scalerFactor)
            micro = try f(.micro, micro)
            lobeScale = try f(.lobeScale, lobeScale)
            mid = try f(.mid, mid)
            presharpen = try f(.presharpen, presharpen)
            adaptive = try f(.adaptive, adaptive)
            skinProtect = try f(.skinProtect, skinProtect)
            taaFeedback = try f(.taaFeedback, taaFeedback)
            taaGamma = try f(.taaGamma, taaGamma)
            grain = try f(.grain, grain)
            debandThreshold = try f(.debandThreshold, debandThreshold)
            cdefSecondary = try f(.cdefSecondary, cdefSecondary)
            cdefPrimary = try f(.cdefPrimary, cdefPrimary)
            loopFilterQuant = try f(.loopFilterQuant, loopFilterQuant)
            stageCdef = try f(.stageCdef, stageCdef)
            stageTaa = try f(.stageTaa, stageTaa)
            stageLoopFilter = try f(.stageLoopFilter, stageLoopFilter)
            stageOklab = try f(.stageOklab, stageOklab)
            stageDeband = try f(.stageDeband, stageDeband)
            stageSiting = try f(.stageSiting, stageSiting)
        }

        /// The whole product, as far as someone using it is concerned: one
        /// control with four positions. Everything else in this struct is a
        /// development knob and is not exposed outside the test page.
        enum Strength: String, CaseIterable, Sendable {
            case off, subtle, standard, strong
            var label: String {
                switch self {
                case .off: return "Off"
                case .subtle: return "Subtle"
                case .standard: return "Standard"
                case .strong: return "Strong"
                }
            }
            var detail: String {
                switch self {
                case .off: return "Upscaling only, no grade"
                case .subtle: return "Barely there"
                case .standard: return "Recommended"
                case .strong: return "For very soft sources"
                }
            }
        }

        /// Applies a strength without disturbing the stage switches, which are
        /// not something a person using this should have to think about.
        mutating func apply(_ strength: Strength) {
            // The ladder is built on `sharpness`, not `fine`, and that is the
            // whole point of it rather than a detail.
            //
            // Both controls add high-frequency detail, but not the same way.
            // `sharpness` drives a contrast-adaptive lobe whose per-pixel
            // strength is set by how close that pixel already sits to the local
            // extremes, and which cannot leave the range of the pixels it was
            // computed from. `fine` is a straight gain on a band: no
            // adaptation, no per-pixel limit, clamped only at the very end.
            //
            // Every arm used to pin sharpness to 0 and express strength purely
            // through `fine` - the self-limiting mechanism switched off and the
            // unlimited one carrying all the detail. That came directly from
            // measuring on band correlation, which rewards matching the
            // reference's band energy (a flat gain does that well) and punishes
            // an adaptive lobe's deviations. Measured perceptually the order
            // reverses, and the edge-overshoot guard says the swept
            // configuration rings 27% LESS than what shipped while carrying
            // more detail. Putting the lift back through the mechanism built to
            // carry it safely is one move, not two.
            switch strength {
            case .off:
                sharpness = 0; fine = 0; blackPoint = 0; whitePoint = 1
                contrast = 0; saturation = 1; sourceDeblock = 0; micro = 0
            case .subtle:
                sharpness = 0.40; fine = 0; blackPoint = 0.020; whitePoint = 0.990
                contrast = 0.10; saturation = 1.0; sourceDeblock = 0; micro = 0
            case .standard:
                // Swept on the adversarially fine-tuned model against the
                // perceptual scoreboard. micro is held at 0 deliberately: the
                // sweep wanted 0.5, it buys 0.0007 of DISTS - noise - and adds
                // 20% to the overshoot, and when the primary metric cannot
                // separate two settings the tiebreak is risk.
                sharpness = 0.75; fine = 0; blackPoint = 0.020; whitePoint = 0.990
                contrast = 0.20; saturation = 1.0; sourceDeblock = 0; micro = 0
            case .strong:
                sharpness = 1.00; fine = 0; blackPoint = 0.030; whitePoint = 0.985
                contrast = 0.30; saturation = 1.0; sourceDeblock = 0; micro = 0
            }
        }

        static func load() -> Tuning {
            var t = Tuning()
            let env = ProcessInfo.processInfo.environment
            if let path = env["LUCID_TUNING"], let data = FileManager.default.contents(atPath: path),
               let loaded = try? JSONDecoder().decode(Tuning.self, from: data) { t = loaded }
            func f(_ key: String, _ current: Float) -> Float { env[key].flatMap(Float.init) ?? current }
            t.sharpness = f("LUCID_SHARP", t.sharpness)
            t.fine = f("LUCID_FINE", t.fine)
            t.deblock = f("LUCID_DEBLOCK", t.deblock)
            t.sourceDeblock = f("LUCID_SRC_DEBLOCK", t.sourceDeblock)
            t.temporal = f("LUCID_TEMPORAL", t.temporal)
            t.blackPoint = f("LUCID_BLACK", t.blackPoint)
            t.whitePoint = f("LUCID_WHITE", t.whitePoint)
            t.contrast = f("LUCID_CONTRAST", t.contrast)
            t.saturation = f("LUCID_SAT", t.saturation)
            t.scalerFactor = f("LUCID_SCALE", t.scalerFactor)
            t.micro = f("LUCID_MICRO", t.micro)
            t.lobeScale = f("LUCID_LOBE", t.lobeScale)
            t.mid = f("LUCID_MID", t.mid)
            t.presharpen = f("LUCID_PRESHARP", t.presharpen)
            t.adaptive = f("LUCID_ADAPTIVE", t.adaptive)
            t.skinProtect = f("LUCID_SKINPROTECT", t.skinProtect)
            t.taaFeedback = f("LUCID_TAAFEEDBACK", t.taaFeedback)
            t.taaGamma = f("LUCID_TAAGAMMA", t.taaGamma)
            t.grain = f("LUCID_GRAIN", t.grain)
            t.debandThreshold = f("LUCID_DEBANDTHRESHOLD", t.debandThreshold)
            t.cdefSecondary = f("LUCID_CDEFSECONDARY", t.cdefSecondary)
            t.cdefPrimary = f("LUCID_CDEFPRIMARY", t.cdefPrimary)
            t.loopFilterQuant = f("LUCID_LOOPFILTERQUANT", t.loopFilterQuant)
            t.stageCdef = f("LUCID_STAGECDEF", t.stageCdef)
            t.stageTaa = f("LUCID_STAGETAA", t.stageTaa)
            t.stageLoopFilter = f("LUCID_STAGELOOPFILTER", t.stageLoopFilter)
            t.stageOklab = f("LUCID_STAGEOKLAB", t.stageOklab)
            t.stageDeband = f("LUCID_STAGEDEBAND", t.stageDeband)
            t.stageSiting = f("LUCID_STAGESITING", t.stageSiting)
            return t
        }
    }

    nonisolated(unsafe) static var tuning = Tuning.load()

    /// The current values, for the page's controls.
    static var tuningDictionary: [String: Float] {
        let t = tuning
        return ["sharpness": t.sharpness, "fine": t.fine, "deblock": t.deblock,
                "sourceDeblock": t.sourceDeblock, "temporal": t.temporal,
                "blackPoint": t.blackPoint, "whitePoint": t.whitePoint,
                "contrast": t.contrast, "saturation": t.saturation,
                "micro": t.micro, "lobeScale": t.lobeScale, "mid": t.mid,
                "presharpen": t.presharpen, "adaptive": t.adaptive,
                "stageSiting": t.stageSiting, "stageDeband": t.stageDeband, "stageOklab": t.stageOklab,
                "stageLoopFilter": t.stageLoopFilter, "stageTaa": t.stageTaa, "stageCdef": t.stageCdef,
                "loopFilterQuant": t.loopFilterQuant, "cdefPrimary": t.cdefPrimary, "cdefSecondary": t.cdefSecondary,
                "debandThreshold": t.debandThreshold, "grain": t.grain, "taaGamma": t.taaGamma,
                "taaFeedback": t.taaFeedback, "skinProtect": t.skinProtect]
    }

    /// Advances every rebuild so the stochastic stages do not stand still.
    nonisolated(unsafe) static var frameCounter: UInt32 = 0

    nonisolated static func detailSettings(for report: BrowserVideoReport, outputScale: Double = 1) -> DetailSettings {
        guard let video = report.video, video.iw > 0 else { return .off }
        let stretch = video.rect.w * report.dpr / Double(video.iw)
        let strength = Float(min(max((stretch - 1.2) / 2.0, 0), 1))
        let t = tuning
        return DetailSettings(
            sharpness: t.sharpness,
            fine: t.fine * (0.7 + 0.3 * strength),
            micro: t.micro,
            lobeScale: t.lobeScale,
            mid: t.mid,
            flatThreshold: 0.004,
            edgeThreshold: 0.030,
            deblock: t.deblock,
            sourceDeblock: t.sourceDeblock,
            sourceDeblockRadius: 1.6,
            presharpen: t.presharpen,
            adaptive: t.adaptive,
            temporal: t.temporal,
            motionLow: 0.02,
            motionHigh: 0.08,
            radius: max(1, Int(outputScale.rounded())),
            blackPoint: t.blackPoint,
            whitePoint: t.whitePoint,
            contrast: t.contrast,
            saturation: t.saturation,
            stageLoopFilter: t.stageLoopFilter > 0.5,
            stageCdef: t.stageCdef > 0.5,
            stageDeband: t.stageDeband > 0.5,
            stageTaa: t.stageTaa > 0.5,
            stageOklab: t.stageOklab > 0.5,
            loopFilterQuant: t.loopFilterQuant,
            cdefPrimary: t.cdefPrimary,
            cdefSecondary: t.cdefSecondary,
            debandThreshold: t.debandThreshold,
            grain: t.grain,
            taaGamma: t.taaGamma,
            taaFeedback: t.taaFeedback,
            skinProtect: t.skinProtect,
            frame: Float(frameCounter)
        )
    }

    /// Decoded video pixels per window point for the current engine.
    private func pixelsPerPoint(_ report: BrowserVideoReport, box: CGRect) -> CGFloat {
        guard let video = report.video else { return backingScale }
        // Capturing at the decoded size means ScreenCaptureKit downsamples the
        // browser's already-upscaled rendering; the native engine captures at
        // display resolution instead, so nothing is resampled on the way in.
        return engine.rescales ? CGFloat(video.iw) / box.width : backingScale
    }

    private func captureSize(for rect: CGRect, scale: CGFloat) -> CGSize {
        CGSize(width: (rect.width * scale).rounded().evenDown, height: (rect.height * scale).rounded().evenDown)
    }

    init(
        report: BrowserVideoReport,
        window browserWindow: WindowInfo,
        latencySeconds: Double,
        engine: EngineKind,
        onStats: @escaping @Sendable (PipelineStats) -> Void,
        onEnded: @escaping (String) -> Void
    ) throws {
        guard let video = report.video else { throw Failure.noVideo }
        id = report.session
        windowID = browserWindow.id
        pid = browserWindow.pid
        self.engine = engine
        self.report = report
        self.onEnded = onEnded

        backingScale = WindowTracker.screen(for: browserWindow.bounds)?.backingScaleFactor ?? 2
        let box = report.windowLocalRect(video.rect, backingScale: backingScale)
        guard box.width >= 2, box.height >= 2 else { throw Failure.tooSmall }

        presenter = FramePresenter(latencySeconds: latencySeconds)
        window = ShadowLayerWindow(presenter: presenter)
        controller = ShadowLayerController(
            window: window, presenter: presenter,
            windowID: browserWindow.id, pid: browserWindow.pid, report: report
        )
        // The learned engine has a fixed input size, so it takes the box exactly.

        let rect = controller.captureRect(for: box, in: browserWindow.bounds.size)
        let scale = engine.rescales ? CGFloat(video.iw) / box.width : backingScale
        let size = CGSize(width: (rect.width * scale).rounded().evenDown, height: (rect.height * scale).rounded().evenDown)

        capture = CaptureSession(configuration: .init(
            windowID: browserWindow.id,
            sourceRect: rect,
            outputSize: size,
            pixelFormat: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            frameRate: 60
        ))
        let controllerRef = controller
        pipeline = EnhancementPipeline(
            factory: Self.makeFactory(engine: engine, report: report),
            presenter: presenter,
            onFirstFrame: { Task { @MainActor in controllerRef.hasContent = true } },
            onStats: onStats
        )

        decoded.setContentRect(box)
        // Send frames at the size the box actually occupies on screen. Capping
        // below that silently undoes the detail stages.
        sender.maximumWidth = Self.deliveryWidth(for: report)
        // Hand every finished frame back to the page. Drawing it there puts it
        // under the page's own scroll, clipping and stacking, which no overlay
        // window can match.
        let sessionID = report.session
        let handler: @Sendable (CVPixelBuffer) -> Void = { [weak self] buffer in
            guard let self else { return }
            Task { @MainActor in
                // CSP-restricted sites cannot receive the in-page canvas frames
                // (ArrayBuffers do not survive the service-worker JSON hop), so
                // the overlay window presents there instead. Packing and
                // queueing ~10MB frames nobody reads is pure waste and feeds
                // the nw_write_request_list_remove_head crash: skip it.
                guard self.pageRenders else { return }
                self.sentFrames += 1
                guard let packet = self.sender.packet(for: buffer, sequence: self.sentFrames, session: sessionID) else {
                    if AppCoordinator.debugLogging, self.sentFrames % 60 == 1 { print("   ⚠️ could not pack enhanced frame") }
                    return
                }
                if AppCoordinator.debugLogging, self.sentFrames % 120 == 1 {
                    print("   🖼 sent enhanced frame \(self.sentFrames), \(packet.count / 1024) kB")
                }
                AppCoordinator.shared.sendEnhanced(packet, session: sessionID)
            }
        }
        Task { await pipeline.setEnhancedFrameHandler(handler) }

        controller.onCaptureRectChanged = { [weak self] rect in self?.follow(rect) }
        controller.onResized = { [weak self] in self?.scheduleResize() }
        controller.onWindowLost = { [weak self] in self?.end("Browser window closed") }
    }

    /// Builds the stage factory for an engine and the current report. Runs on
    /// the pipeline actor whenever the frame size changes.
    nonisolated private static func makeFactory(engine: EngineKind, report: BrowserVideoReport) -> EnhancementPipeline.StageFactory {
        let video = report.video
        let stretch = (video.map { $0.rect.w * report.dpr / Double(max($0.iw, 1)) }) ?? 1
        let wanted = stageCount(for: report)
        return { width, height in
            let compositor = try MetalTileCompositor()
            let kind = engine
            let learned: LearnedUpscaler?
            var upscaler: TiledVideoToolboxUpscaler?
            var built = 1
            if kind.usesLearned {
                // Not `try?`: a session only starts for sizes the table covers,
                // so a missing model is an error, not a silent downgrade.
                learned = try LearnedUpscaler(width: width, height: height)
            } else {
                guard EngineKind.comparisonPathEnabled else {
                    throw EnhancementSession.Failure.comparisonEngineDisabled
                }
                learned = nil
                let factor = Int(tuning.scalerFactor.rounded())
                let first = try TiledVideoToolboxUpscaler(
                    width: width, height: height, compositor: compositor, preferredScale: factor
                )
                var tail = first
                while built < wanted, factor < 4 {
                    guard let next = try? TiledVideoToolboxUpscaler(
                        width: tail.outputWidth, height: tail.outputHeight, compositor: compositor, preferredScale: factor
                    ),
                          first.totalTileCount + next.layout.tileCount <= 12 else { break }
                    tail.nextStage = next
                    tail = next
                    built += 1
                }
                upscaler = first
            }
            // SPAN is a fixed 4x, so the detail stage that follows works at
            // that scale rather than the tiled pass count.
            let outputScale = learned != nil ? 4.0 : (kind.rescales ? pow(2, Double(built)) : stretch)
            let settings = detailSettings(for: report, outputScale: outputScale)
            let detail = kind.usesDetail ? try? DetailEnhancer(device: compositor.device, settings: settings) : nil
            let label: String
            if let learned {
                label = "\(kind.label): SPAN 4× → \(learned.outputWidth)x\(learned.outputHeight), input \(width)x\(height)"
            } else if let upscaler {
                label = "\(kind.label): \(Int(pow(2.0, Double(built))))× in \(built) pass\(built == 1 ? "" : "es") → \(upscaler.outputWidth)x\(upscaler.outputHeight), \(upscaler.totalTileCount) tiles, input \(width)x\(height), radius \(settings.radius)"
            } else {
                label = "\(kind.label): 1:1 at \(width)x\(height), radius \(settings.radius)"
            }
            return PipelineStages(
                upscaler: upscaler, learned: learned, detail: detail,
                inputWidth: width, inputHeight: height,
                preprocessRadius: kind.rescales ? 1 : max(1, Int(stretch.rounded())),
                label: label
            )
        }
    }

    func start() async {
        // Give the page a moment to start sending decoded frames. Only if it
        // cannot (cross-origin video, no WebCodecs) do we read the screen.
        controller.start()
        decodedTask = Task { @MainActor [weak self] in
            guard let self else { return }
            await self.pipeline.run(self.decoded.stream())
        }
        // Wait properly for the page to start sending: it only begins once the
        // video is actually playing, which can be a second or more after the
        // first report arrives.
        for _ in 0..<24 {
            if stopped { return }
            if decoded.frameCount > 0 {
                usingDecodedFrames = true
                print("   🎞 decoded frames from the page at \(Int(decoded.lastSize.width))x\(Int(decoded.lastSize.height)) — no screen capture")
                return
            }
            try? await Task.sleep(for: .milliseconds(250))
        }
        print("   ⏳ no decoded frames after 6s")
        await startScreenCapture()
    }

    private func startScreenCapture() async {
        guard !screenCaptureRunning else { return }
        if !ScreenCapturePermission.isGranted {
            if AppCoordinator.shared.screenCaptureDenied {
                end("Screen Recording permission required")
                return
            }
            let granted = await ScreenCapturePermission.request()
            if !granted {
                AppCoordinator.shared.noteScreenCaptureDenied()
                end("Screen Recording permission required")
                return
            }
        }
        do {
            let stream = try await capture.start()
            captureRunning = true
            screenCaptureRunning = true
            decodedTask?.cancel()
            await pipeline.run(stream)
            if let rect = pendingRect { pendingRect = nil; follow(rect) }
            print("   🖥 falling back to screen capture")
        } catch {
            end("Capture failed: \(error.localizedDescription)")
        }
    }

    /// A decoded frame arrived from the page.
    func accept(_ frame: DecodedFrame) {
        guard frame.header.session == id else {
            if AppCoordinator.debugLogging, rejectedFrames == 0 {
                print("   ⚠️ frame session \(frame.header.session.prefix(8)) != app session \(id.prefix(8))")
            }
            rejectedFrames += 1
            return
        }
        lastDecodedFrame = .now
        decoded.accept(frame)
        if !usingDecodedFrames, !screenCaptureRunning { usingDecodedFrames = true }
    }

    /// Rebuilds the stages so a settings change takes effect on the next frame.
    func reloadTuning() {
        let report = self.report
        Task {
            await pipeline.updateDetail { current in
                var updated = EnhancementSession.detailSettings(for: report)
                // Radius and the stage layout were fixed when the pipeline was
                // built; only the tunable parts change here.
                updated.radius = current.radius
                return updated
            }
        }
    }

    var videoScreenRect: CGRect { controller.videoScreenRect }
    var overlayWindowID: CGWindowID { CGWindowID(window.windowNumber) }
    func setCapturable(_ capturable: Bool) { controller.setCapturable(capturable) }

    /// The page reports whether it is drawing the frames itself.
    func setPageRenders(_ renders: Bool) {
        guard renders != pageRenders else { return }
        pageRenders = renders
        controller.setForcedHidden(renders)
        print(renders ? "   🖼 the page is drawing the enhanced frames" : "   🪟 falling back to the overlay window")
    }
    func setOverlayHidden(_ hidden: Bool) { controller.setForcedHidden(hidden) }

    /// Switches engine without touching the capture stream.
    func setEngine(_ kind: EngineKind) {
        guard kind != engine else { return }
        engine = kind
        Task { await pipeline.reconfigure(Self.makeFactory(engine: kind, report: report)) }
        scheduleResize()
    }

    func update(report: BrowserVideoReport) {
        let previous = self.report
        self.report = report
        controller.update(report: report)
        if let video = report.video {
            decoded.setContentRect(report.windowLocalRect(video.rect, backingScale: backingScale))
            sender.maximumWidth = Self.deliveryWidth(for: report)
        }
        // A quality switch changes the decoded size, which changes the capture
        // size for the scaling engines; handled like a resize, in place.
        if let a = previous.video, let b = report.video, a.iw != b.iw || a.ih != b.ih {
            Task { await pipeline.reconfigure(Self.makeFactory(engine: engine, report: report)) }
            scheduleResize()
        }
    }

    /// Reapplies the capture rect and output size after the box settles.
    private func scheduleResize() {
        resizeTask?.cancel()
        resizeTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            guard let self, !Task.isCancelled, !self.stopped else { return }
            self.follow(self.controller.captureRect)
        }
    }

    /// Points the stream at a rect, resizing the output to match the engine's
    /// pixel scale. The stream keeps running.
    private func follow(_ rect: CGRect) {
        guard captureRunning else { pendingRect = rect; return }
        guard rect.width >= 2, rect.height >= 2, let video = report.video else { return }
        let box = report.windowLocalRect(video.rect, backingScale: backingScale)
        let scale = engine.rescales ? CGFloat(video.iw) / max(box.width, 1) : backingScale
        var configuration = capture.configuration
        configuration.sourceRect = rect
        configuration.outputSize = captureSize(for: rect, scale: scale)
        Task { @MainActor in
            do {
                try await capture.update(configuration)
                if AppCoordinator.debugLogging {
                    print("   📐 capture \(Int(rect.minX)),\(Int(rect.minY)) \(Int(rect.width))x\(Int(rect.height)) → \(Int(configuration.outputSize.width))x\(Int(configuration.outputSize.height))")
                }
            } catch {
                print("   ⚠️ capture update failed: \(error)")
            }
        }
    }

    private func end(_ reason: String) {
        guard !stopped else { return }
        onEnded(reason)
    }

    func stop() async {
        guard !stopped else { return }
        stopped = true
        decodedTask?.cancel()
        decoded.finish()
        resizeTask?.cancel()
        controller.stop()
        await pipeline.stop()
        await capture.stop()
        window.orderOut(nil)
    }
}

private extension CGFloat {
    /// Largest even integer not greater than self; 420v needs even dimensions.
    var evenDown: CGFloat { (self / 2).rounded(.down) * 2 }
}
