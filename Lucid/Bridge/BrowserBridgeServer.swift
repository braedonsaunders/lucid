//
//  BrowserBridgeServer.swift
//  Lucid
//
//  Loopback WebSocket server the browser companions connect to. Extension
//  background scripts (Chrome, Edge, Safari) and pages in direct mode send
//  `BrowserVideoReport` JSON messages. The server is the only channel between
//  the browser and the native app; nothing is ever written into the page.
//

import Foundation
import Network

final class BrowserBridgeServer: @unchecked Sendable {
    static let defaultPort: UInt16 = 47811

    private let port: UInt16
    private let queue = DispatchQueue(label: "com.lucid.bridge", qos: .userInteractive)
    private var listener: NWListener?
    private var connections: [ObjectIdentifier: NWConnection] = [:]
    private let decoder = JSONDecoder()
    private let onReport: @Sendable (BrowserVideoReport) -> Void
    /// Decoded frames straight from the browser, at the video's own resolution.
    var onFrame: (@Sendable (DecodedFrame) -> Void)?
    private let onDisconnect: @Sendable (Set<String>) -> Void
    private let onControl: @Sendable (BridgeControl) -> Void
    private var sessionsByConnection: [ObjectIdentifier: Set<String>] = [:]
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

    init(
        port: UInt16 = BrowserBridgeServer.defaultPort,
        onReport: @escaping @Sendable (BrowserVideoReport) -> Void,
        onDisconnect: @escaping @Sendable (Set<String>) -> Void,
        onControl: @escaping @Sendable (BridgeControl) -> Void = { _ in }
    ) {
        self.port = port
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
            for connection in connections.values where connection.state == .ready {
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
            for (id, connection) in self.connections where connection.state == .ready {
                guard self.sessionsByConnection[id]?.contains(session) == true else { continue }
                guard self.framesInFlight[id, default: 0] < Self.maximumFramesInFlight else {
                    self.droppedFrames += 1
                    continue
                }
                self.framesInFlight[id, default: 0] += 1
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
        }
    }

    /// Sends any encodable message to every connected page.
    func broadcast<T: Encodable & Sendable>(_ message: T) {
        guard let data = try? JSONEncoder().encode(message) else { return }
        let context = NWConnection.ContentContext(
            identifier: "text", metadata: [NWProtocolWebSocket.Metadata(opcode: .text)]
        )
        queue.async { [weak self] in
            guard let self else { return }
            for connection in self.connections.values {
                connection.send(content: data, contentContext: context, isComplete: true, completion: .contentProcessed { _ in })
            }
        }
    }

    func start() throws {
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        parameters.requiredLocalEndpoint = NWEndpoint.hostPort(
            host: "127.0.0.1",
            port: NWEndpoint.Port(rawValue: port)!
        )
        let webSocket = NWProtocolWebSocket.Options()
        webSocket.autoReplyPing = true
        webSocket.maximumMessageSize = 48 << 20   // a 4K enhanced frame is 33 MB of RGBA
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
        listener?.cancel()
        listener = nil
        for connection in connections.values { connection.cancel() }
        connections.removeAll()
    }

    private func accept(_ connection: NWConnection) {
        let key = ObjectIdentifier(connection)
        connections[key] = connection
        print("   🔌 Bridge: browser connected (\(connections.count) open)")
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
        guard connections.removeValue(forKey: key) != nil else { return }
        framesInFlight.removeValue(forKey: key)
        print("   🔌 Bridge: browser disconnected (\(connections.count) open)")
        if let sessions = sessionsByConnection.removeValue(forKey: key), !sessions.isEmpty {
            onDisconnect(sessions)
        }
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
        if data.count > 8, data.withUnsafeBytes({ $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self).bigEndian }) == 0x4c554346 {
            handleBinary(data)
            return
        }
        if let probe = try? decoder.decode(MessageProbe.self, from: data), probe.type == "control" {
            if let control = try? decoder.decode(BridgeControl.self, from: data) { onControl(control) }
            return
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
}

/// Commands a page or tool may send: `{"type":"control", ...}`.
struct BridgeControl: Codable, Sendable {
    var enabled: Bool?
    /// Latency budget in seconds.
    var latency: Double?
    /// Which reconstruction engine to use.
    var engine: String?
    /// Capture one frame per engine into this folder under ~/Documents/Code/lucid/.build/shots.
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
    var engines: [String] = EngineKind.allCases.map(\.rawValue)
    var engineLabels: [String] = EngineKind.allCases.map(\.label)
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
