//
//  BridgeAuth.swift
//  Lucid
//
//  Per-launch token for the loopback bridge. A webpage can open
//  ws://127.0.0.1:47811; without this, it could steal or inject frames.
//  The extension cannot read a 0600 file, so a tiny HTTP issuer on
//  127.0.0.1:47812 hands the token only to allowlisted Origins (the
//  extension, and loopback lab pages). A https origin gets 403 and no
//  CORS reflection. Missing Origin is allowed: that is the service
//  worker / node case, and a same-user process can read the 0600 file
//  anyway. Origin "null" is not allowed — open the lab through its
//  local server, not as file://.
//

import Foundation
import Network
import Security

enum BridgeAuth {
    static let tokenHTTPPort: UInt16 = 47812

    /// 32 random bytes as 64 hex chars. `LUCID_BRIDGE_TOKEN` overrides for tests.
    static func issue() -> String {
        let token: String
        if let env = ProcessInfo.processInfo.environment["LUCID_BRIDGE_TOKEN"], !env.isEmpty {
            token = env
        } else {
            token = generate()
        }
        do {
            try write(token)
        } catch {
            print("   ⚠️ could not write bridge.token: \(error)")
        }
        return token
    }

    static func generate() -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        if SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess {
            return bytes.map { String(format: "%02x", $0) }.joined()
        }
        let fallback = (UUID().uuidString + UUID().uuidString)
            .replacingOccurrences(of: "-", with: "")
            .lowercased()
        return String(fallback.prefix(64))
    }

    static func tokensEqual(_ expected: String, _ provided: String?) -> Bool {
        guard let provided else { return false }
        let a = Array(expected.utf8)
        let b = Array(provided.utf8)
        guard a.count == b.count, !a.isEmpty else { return false }
        var acc: UInt8 = 0
        for i in 0..<a.count { acc |= a[i] ^ b[i] }
        return acc == 0
    }

    /// Who may read the token. A missing Origin is the same-user case.
    static func allowsOrigin(_ origin: String?) -> Bool {
        guard let origin, !origin.isEmpty else { return true }
        if origin == "null" { return false }
        if origin.hasPrefix("chrome-extension://") { return true }
        if origin.hasPrefix("moz-extension://") { return true }
        if origin.hasPrefix("safari-web-extension://") { return true }
        guard let url = URL(string: origin), let host = url.host else { return false }
        return host == "127.0.0.1" || host == "localhost" || host == "::1"
    }

    private static func write(_ token: String) throws {
        let fm = FileManager.default
        let bundle = Bundle.main.bundleIdentifier ?? "com.braedonsaunders.lucid"
        let dir = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(bundle, isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("bridge.token")
        try Data(token.utf8).write(to: url, options: .atomic)
        try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }
}

/// GET /token on 127.0.0.1:47812. One request, then the connection closes.
final class BridgeTokenIssuer: @unchecked Sendable {
    private let token: String
    private let port: UInt16
    private let queue = DispatchQueue(label: "com.lucid.bridge.token", qos: .utility)
    private var listener: NWListener?

    init(token: String, port: UInt16 = BridgeAuth.tokenHTTPPort) {
        self.token = token
        self.port = port
    }

    func start() throws {
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        parameters.requiredLocalEndpoint = NWEndpoint.hostPort(
            host: "127.0.0.1",
            port: NWEndpoint.Port(rawValue: port)!
        )
        let listener = try NWListener(using: parameters)
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready:
                print("   🔐 Bridge token issuer on http://127.0.0.1:\(BridgeAuth.tokenHTTPPort)/token")
            case .failed(let error):
                print("   ❌ Bridge token issuer failed: \(error)")
            default:
                break
            }
        }
        listener.start(queue: queue)
        self.listener = listener
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        receive(on: connection, buffer: Data())
    }

    private func receive(on connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 8192) { [weak self] data, _, _, error in
            guard let self else { return }
            if let error {
                connection.cancel()
                if Self.isIgnorable(error) { return }
                return
            }
            var next = buffer
            if let data { next.append(data) }
            if next.count > 16_384 {
                self.respond(on: connection, status: 413, origin: nil, body: "")
                return
            }
            guard let text = String(data: next, encoding: .utf8),
                  text.contains("\r\n\r\n") || text.contains("\n\n") else {
                self.receive(on: connection, buffer: next)
                return
            }
            self.handleHTTP(text, on: connection)
        }
    }

    private func handleHTTP(_ text: String, on connection: NWConnection) {
        let lines = text.split(whereSeparator: \.isNewline).map {
            $0.trimmingCharacters(in: CharacterSet(charactersIn: "\r"))
        }
        guard let request = lines.first else {
            respond(on: connection, status: 400, origin: nil, body: "")
            return
        }
        let parts = request.split(separator: " ")
        let method = parts.first.map { String($0).uppercased() } ?? ""
        let rawPath = parts.dropFirst().first.map(String.init) ?? ""
        let path = rawPath.split(separator: "?").first.map(String.init) ?? rawPath

        var origin: String?
        for line in lines.dropFirst() {
            guard !line.isEmpty else { break }
            let lower = line.lowercased()
            if lower.hasPrefix("origin:") {
                origin = String(line.dropFirst(7)).trimmingCharacters(in: .whitespaces)
            }
        }

        let allowed = BridgeAuth.allowsOrigin(origin)
        if method == "OPTIONS" && path == "/token" {
            respond(on: connection, status: allowed ? 204 : 403, origin: allowed ? origin : nil, body: "")
            return
        }
        guard method == "GET", path == "/token" else {
            respond(on: connection, status: 404, origin: allowed ? origin : nil, body: "")
            return
        }
        guard allowed else {
            respond(on: connection, status: 403, origin: nil, body: "")
            return
        }
        respond(on: connection, status: 200, origin: origin, body: token, contentType: "text/plain; charset=utf-8")
    }

    private func respond(
        on connection: NWConnection,
        status: Int,
        origin: String?,
        body: String,
        contentType: String = "text/plain"
    ) {
        let reason: String
        switch status {
        case 200: reason = "OK"
        case 204: reason = "No Content"
        case 400: reason = "Bad Request"
        case 403: reason = "Forbidden"
        case 404: reason = "Not Found"
        case 413: reason = "Payload Too Large"
        default: reason = "Error"
        }
        var lines = [
            "HTTP/1.1 \(status) \(reason)",
            "Content-Type: \(contentType)",
            "Content-Length: \(body.utf8.count)",
            "Cache-Control: no-store",
            "Connection: close"
        ]
        if let origin, !origin.isEmpty, BridgeAuth.allowsOrigin(origin) {
            lines.append("Access-Control-Allow-Origin: \(origin)")
            lines.append("Access-Control-Allow-Methods: GET, OPTIONS")
            lines.append("Vary: Origin")
        }
        lines.append("")
        lines.append(body)
        let data = Data(lines.joined(separator: "\r\n").utf8)
        connection.send(content: data, contentContext: .defaultMessage, isComplete: true, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }

    private static func isIgnorable(_ error: Error) -> Bool {
        let ns = error as NSError
        return ns.domain == NWError.errorDomain
    }
}
