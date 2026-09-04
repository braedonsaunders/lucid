//
//  BrowserBridgeServer.swift
//  Lucid
//
//  Loopback WebSocket server the browser companions connect to. Extension
//  background scripts (Chrome, Edge, Safari) and pages in direct mode send
//  `BrowserVideoReport` JSON messages. The server is the only channel between
//  the browser and the native app; nothing is ever written into the page.
//
//  Every connection must send `{"type":"hello","token":"..."}` before
//  reports, attach, control, or frames. The token is issued on
//  http://127.0.0.1:47812/token to allowlisted Origins only. A website
//  that opens this socket without the token is dropped.
//

import Foundation
import Network

final class BrowserBridgeServer: @unchecked Sendable {
    static let defaultPort: UInt16 = 47811

    private let port: UInt16
    private let token: String
    private var tokenIssuer: BridgeTokenIssuer?
    private let queue = DispatchQueue(label: "com.lucid.bridge", qos: .userInteractive)
    private var listener: NWListener?
    private var connections: [ObjectIdentifier: NWConnection] = [:]
    private var authenticated: Set<ObjectIdentifier> = []
    private var authTimeouts: [ObjectIdentifier: DispatchWorkItem] = [:]
    private let decoder = JSONDecoder()
    private let onReport: @Sendable (BrowserVideoReport) -> Void
    /// Decoded frames straight from the browser, at the video's own resolution.
    var onFrame: (@Sendable (DecodedFrame) -> Void)?
    private let onDisconnect: @Sendable (Set<String>) -> Void
    private let onControl: @Sendable (BridgeControl) -> Void
    private var sessionsByConnection: [ObjectIdentifier: Set<String>] = [:]
    /// Connections that said "attach" - they exist to receive frames and they
    /// read them. A connection that merely sent a report also gets listed in
    /// `sessionsByConnection`, but it may be a socket that only carries reports
    /// and never reads binary at all, so an attached connection always wins.
    private var attachedByConnection: [ObjectIdentifier: Set<String>] = [:]
    /// Frame sends still outstanding per connection. An enhanced frame is
    /// megabytes; queueing them faster than the socket drains grows
    /// Network.framework's write list without bound, which shows up first as
    /// rising latency and then as a trap inside nw_write_request_list_remove_head.
    /// A stale frame is worth less than a live one, so we drop instead of queue.
    private var framesInFlight: [ObjectIdentifier: Int] = [:]
    private static let maximumFramesInFlight = 1
    /// Frames dropped for backpressure, so the shortfall is visible rather than
    /// looking like the pipeline stalled.
    private(set) var droppedFrames = 0
    /// Frames actually handed to a socket, and when this window started. The
    /// pipeline's own fps counts frames it produced, which is not the same as
    /// frames the page received - and when the two disagree the picture stops
    /// changing while every number in the app still looks healthy.
    private var deliveredFrames = 0
    private var deliveredBytes = 0
    private var offeredFrames = 0
    private var unmatchedFrames = 0
    private var lastFrameWidth = 0
    private var lastFrameHeight = 0
    private var windowStart = Date()

    init(
        port: UInt16 = BrowserBridgeServer.defaultPort,
        onReport: @escaping @Sendable (BrowserVideoReport) -> Void,
        onDisconnect: @escaping @Sendable (Set<String>) -> Void,
        onControl: @escaping @Sendable (BridgeControl) -> Void = { _ in }
    ) {
        self.port = port
        self.token = BridgeAuth.issue()
        self.onReport = onReport
        self.onDisconnect = onDisconnect
        self.onControl = onControl
    }

    /// Sends a JSON status object to every connected browser (test/lab pages
    /// display it; the extension ignores it).
    func broadcast(_ status: BridgeStatus) {
        queue.async { [self] in
            guard !connections.isEmpty, let data = try? JSONEncoder().encode(status) else { return }
            let metadata = NWProtocolWebSocket.Metadata(opcode: .text)
            let context = NWConnection.ContentContext(identifier: "status", metadata: [metadata])
            for (id, connection) in connections where connection.state == .ready && authenticated.contains(id) {
                connection.send(content: data, contentContext: context, isComplete: true, completion: .contentProcessed { _ in })
            }
        }
    }

    /// Sends an enhanced frame to the one page that owns the session it was
    /// made for. Broadcasting it was the original design and it is why the app
    /// kept trapping inside Network.framework: a 16 MB frame was being queued
    /// onto every open connection - other tabs, the extension's service worker,
    /// anything - none of which ever read it, so their write queues grew without
    /// limit until the framework gave up.
    func send(binary data: Data, to session: String) {
        let context = NWConnection.ContentContext(
            identifier: "frame", metadata: [NWProtocolWebSocket.Metadata(opcode: .binary)]
        )
        queue.async { [weak self] in
            guard let self else { return }
            // Counted before the guards, not after. A frame that reaches no
            // connection at all is the failure that looks exactly like a
            // healthy pipeline: everything upstream reports 30fps and the page
            // never receives a pixel.
            self.offeredFrames += 1
            if let size = Self.luceSize(data) {
                self.lastFrameWidth = size.0
                self.lastFrameHeight = size.1
            }
            // Prefer connections that asked for this session's frames. Falling
            // back to every connection that merely mentioned the session sends
            // megabytes to sockets that never read them - which looks like
            // perfect delivery and draws nothing.
            let attached = self.connections.filter {
                $0.value.state == .ready && self.authenticated.contains($0.key)
                    && self.attachedByConnection[$0.key]?.contains(session) == true
            }
            let targets = attached.isEmpty
                ? self.connections.filter {
                    $0.value.state == .ready && self.authenticated.contains($0.key)
                        && self.sessionsByConnection[$0.key]?.contains(session) == true
                }
                : attached
            // How many connections this frame was addressed to, which is what
            // `targets` already is. This was a `var` that nothing incremented,
            // so every frame counted as unmatched and the delivery log's "NO
            // CONNECTION" column was always equal to the offered count - a
            // lying counter inside the diagnostic built to catch lying counters.
            let matched = targets.count
            for (id, connection) in targets {
                guard self.framesInFlight[id, default: 0] < Self.maximumFramesInFlight else {
                    self.droppedFrames += 1
                    continue
                }
                self.framesInFlight[id, default: 0] += 1
                self.deliveredFrames += 1
                self.deliveredBytes += data.count
                connection.send(
                    content: data, contentContext: context, isComplete: true,
                    completion: .contentProcessed { [weak self] error in
                        guard let self else { return }
                        self.queue.async {
                            self.framesInFlight[id] = max(0, (self.framesInFlight[id] ?? 1) - 1)
                            // A connection that has started failing must stop
                            // being written to, not be retried every frame.
                            if error != nil { connection.cancel(); self.drop(id) }
                        }
                    }
                )
            }
            if matched == 0 { self.unmatchedFrames += 1 }
            self.reportDelivery()
        }
    }

    /// Once a second, what the page is actually receiving. Called on `queue`,
    /// which owns all three counters.
    private func reportDelivery() {
        let elapsed = Date().timeIntervalSince(windowStart)
        guard elapsed >= 1, offeredFrames > 0 else { return }
        let megabytes = Double(deliveredBytes) / 1_048_576
        let perFrame = deliveredFrames > 0 ? megabytes / Double(deliveredFrames) : 0
        print(String(format: "   📤 offered %.0f fps · delivered %.0f · dropped %d · NO CONNECTION %d · %dx%d · %.1f MB/frame · %.0f MB/s",
                     Double(offeredFrames) / elapsed, Double(deliveredFrames) / elapsed,
                     droppedFrames, unmatchedFrames, lastFrameWidth, lastFrameHeight,
                     perFrame, megabytes / elapsed))
        deliveredFrames = 0; deliveredBytes = 0; droppedFrames = 0
        offeredFrames = 0; unmatchedFrames = 0; windowStart = Date()
    }

    private static func luceSize(_ data: Data) -> (Int, Int)? {
        guard data.count >= 8 else { return nil }
        let magic = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self).bigEndian }
        guard magic == 0x4C554345 else { return nil }
        let headerLength = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self).bigEndian })
        guard data.count >= 8 + headerLength,
              let json = try? JSONSerialization.jsonObject(with: data.subdata(in: 8..<(8 + headerLength))) as? [String: Any],
              let width = json["w"] as? Int, let height = json["h"] as? Int
        else { return nil }
        return (width, height)
    }

    /// Sends any encodable message to every connected page.
    func broadcast<T: Encodable & Sendable>(_ message: T) {
        guard let data = try? JSONEncoder().encode(message) else { return }
        let context = NWConnection.ContentContext(
            identifier: "text", metadata: [NWProtocolWebSocket.Metadata(opcode: .text)]
        )
        queue.async { [weak self] in
            guard let self else { return }
            for (id, connection) in self.connections where self.authenticated.contains(id) {
                connection.send(content: data, contentContext: context, isComplete: true, completion: .contentProcessed { _ in })
            }
        }
    }

    func start() throws {
        let issuer = BridgeTokenIssuer(token: token)
        try issuer.start()
        tokenIssuer = issuer

        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        parameters.requiredLocalEndpoint = NWEndpoint.hostPort(
            host: "127.0.0.1",
            port: NWEndpoint.Port(rawValue: port)!
        )
        let webSocket = NWProtocolWebSocket.Options()
        webSocket.autoReplyPing = true
        webSocket.maximumMessageSize = 48 << 20   // headroom; 4K NV12 is ~12 MB, 3456 RGBA was 25
        parameters.defaultProtocolStack.applicationProtocols.insert(webSocket, at: 0)

        let listener = try NWListener(using: parameters)
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready: print("   🔌 Browser bridge listening on ws://127.0.0.1:\(BrowserBridgeServer.defaultPort)")
            case .failed(let error): print("   ❌ Browser bridge failed: \(error)")
            default: break
            }
        }
        listener.start(queue: queue)
        self.listener = listener
    }

    func stop() {
        tokenIssuer?.stop()
        tokenIssuer = nil
        listener?.cancel()
        listener = nil
        for work in authTimeouts.values { work.cancel() }
        authTimeouts.removeAll()
        authenticated.removeAll()
        for connection in connections.values { connection.cancel() }
        connections.removeAll()
    }

    private func accept(_ connection: NWConnection) {
        let key = ObjectIdentifier(connection)
        connections[key] = connection
        let timeout = DispatchWorkItem { [weak self] in
            guard let self, !self.authenticated.contains(key) else { return }
            print("   🔒 Bridge: dropped unauthenticated connection")
            connection.cancel()
            self.drop(key)
        }
        authTimeouts[key] = timeout
        queue.asyncAfter(deadline: .now() + 2, execute: timeout)
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled:
                self?.queue.async { self?.drop(key) }
            default:
                break
            }
        }
        connection.start(queue: queue)
        receive(on: connection, key: key)
    }

    private func drop(_ key: ObjectIdentifier) {
        authTimeouts[key]?.cancel()
        authTimeouts.removeValue(forKey: key)
        authenticated.remove(key)
        guard connections.removeValue(forKey: key) != nil else { return }
        framesInFlight.removeValue(forKey: key)
        attachedByConnection.removeValue(forKey: key)
        print("   🔌 Bridge: browser disconnected (\(connections.count) open)")
        if let sessions = sessionsByConnection.removeValue(forKey: key), !sessions.isEmpty {
            onDisconnect(sessions)
        }
    }

    private func authenticate(_ data: Data, key: ObjectIdentifier) {
        let isFrame = data.count > 8 && data.withUnsafeBytes({ $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self).bigEndian }) == 0x4c554346
        if !isFrame,
           let probe = try? decoder.decode(MessageProbe.self, from: data),
           probe.type == "hello",
           BridgeAuth.tokensEqual(token, probe.token) {
            authenticated.insert(key)
            authTimeouts[key]?.cancel()
            authTimeouts.removeValue(forKey: key)
            print("   🔌 Bridge: browser authenticated (\(authenticated.count) open)")
            return
        }
        print("   🔒 Bridge: rejected handshake")
        connections[key]?.cancel()
        drop(key)
    }

    private func receive(on connection: NWConnection, key: ObjectIdentifier) {
        connection.receiveMessage { [weak self] data, context, _, error in
            guard let self else { return }
            if let data, !data.isEmpty {
                self.handle(data, key: key)
            }
            if error != nil {
                self.queue.async { self.drop(key) }
                return
            }
            // `isFinal` is set on every complete WebSocket message, not just
            // the last one on the connection. Treating it as a close tore the
            // connection down after each frame, so the page reconnected in a
            // loop and never stayed around long enough to be broadcast to.
            if let metadata = context?.protocolMetadata(definition: NWProtocolWebSocket.definition)
                as? NWProtocolWebSocket.Metadata, metadata.opcode == .close {
                self.queue.async { self.drop(key) }
                return
            }
            self.receive(on: connection, key: key)
        }
    }

    /// Binary frame packet: 'LUCF', big-endian header length, JSON header,
    /// then the planes exactly as the browser laid them out.
    private func handleBinary(_ data: Data) {
        guard data.count > 8 else { return }
        let magic = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self).bigEndian }
        guard magic == 0x4c554346 else { return }
        let headerLength = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self).bigEndian })
        guard data.count >= 8 + headerLength else { return }
        let headerData = data.subdata(in: 8..<(8 + headerLength))
        guard let header = try? decoder.decode(DecodedFrame.Header.self, from: headerData) else { return }
        let payload = data.subdata(in: (8 + headerLength)..<data.count)
        onFrame?(DecodedFrame(header: header, payload: payload))
    }

    private func handle(_ data: Data, key: ObjectIdentifier) {
        if !authenticated.contains(key) {
            authenticate(data, key: key)
            return
        }
        if data.count > 8, data.withUnsafeBytes({ $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self).bigEndian }) == 0x4c554346 {
            handleBinary(data)
            return
        }
        if let probe = try? decoder.decode(MessageProbe.self, from: data) {
            if probe.type == "control" {
                if let control = try? decoder.decode(BridgeControl.self, from: data) { onControl(control) }
                return
            }
            // The drawing surface runs at the extension's origin in its own
            // iframe and holds its own socket, so it has to say which video's
            // frames belong to it. Nothing else identifies that connection.
            if probe.type == "attach", let session = probe.session, !session.isEmpty {
                sessionsByConnection[key, default: []].insert(session)
                attachedByConnection[key, default: []].insert(session)
                if AppCoordinator.debugLogging {
                    print("   🖼 surface attached for session \(session.prefix(8))")
                }
                return
            }
        }
        do {
            let report = try decoder.decode(BrowserVideoReport.self, from: data)
            sessionsByConnection[key, default: []].insert(report.session)
            onReport(report)
        } catch {
            if let text = String(data: data, encoding: .utf8) {
                print("   ⚠️ Bridge: undecodable message (\(error.localizedDescription)): \(text.prefix(200))")
            }
        }
    }
}


private struct MessageProbe: Decodable {
    let type: String
    let session: String?
    let token: String?
}

/// Commands a page or tool may send: `{"type":"control", ...}`.
struct BridgeControl: Codable, Sendable {
    var enabled: Bool?
    /// Latency budget in seconds.
    var latency: Double?
    /// Which reconstruction engine to use.
    var engine: String?
    /// Capture one frame per engine into this folder under ~/Documents/Code/lucid/.build/shots.
    /// Ignored unless the app was launched with LUCID_SHOOT=1 or LUCID_DEBUG=1.
    var shoot: String?
    /// Live enhancement changes: only the keys present are applied.
    var tuning: [String: Float]?
    var resetTuning: Bool?
}

/// Asks any connected page to re-send the current frame.
struct BridgeNudge: Codable, Sendable { var type = "nudge" }

/// One decoded video frame as the browser produced it.
struct DecodedFrame: @unchecked Sendable {
    struct Plane: Codable, Sendable {
        var offset: Int
        var stride: Int
    }
    struct Header: Codable, Sendable {
        var session: String
        var w: Int
        var h: Int
        var format: String
        var planes: [Plane]
        var seq: Int
        var ts: Double
    }
    var header: Header
    var payload: Data
}

/// What the app broadcasts once a second and on every change.
struct BridgeStatus: Codable, Sendable {
    var type = "status"
    var enabled: Bool
    var enhancing: Bool
    var engine: String = ""
    var engines: [String] = EngineKind.shipping.map(\.rawValue)
    var engineLabels: [String] = EngineKind.shipping.map(\.label)
    var tuning: [String: Float] = [:]
    var status: String
    var stats: String
    var latency: Double
    var sourceFPS: Double
    var outputFPS: Double
    var processingMilliseconds: Double
    var tileCount: Int
    var outputWidth: Int
    var outputHeight: Int
    var error: String?
}
