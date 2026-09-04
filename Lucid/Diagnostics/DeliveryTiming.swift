//
//  DeliveryTiming.swift
//  Lucid
//
//  How much the page-bound path costs now that 480p actually runs.
//  Enhanced frames go back as NV12 (4:2:0), sized to the on-screen box
//  (EnhancedFrameSender.maximumWidth). RGBA at 3456x1920 was 25.31 MB
//  and dropped a third of its frames; NV12 is 9.49 MB. This does not
//  bind the shipping bridge (47811 / 47812).
//
//    Lucid --delivery-ms
//    Lucid --delivery-ms 3456 1920 3456 60
//

import CoreVideo
import Foundation
import Network

enum DeliveryTiming {
    static func run() {
        setvbuf(stdout, nil, _IOLBF, 0)
        let args = CommandLine.arguments
        let index = args.firstIndex(of: "--delivery-ms")
        let rest = index.map { Array(args.dropFirst($0 + 1)) } ?? []
        if rest.count >= 3, let w = Int(rest[0]), let h = Int(rest[1]), let maxW = Int(rest[2]) {
            let count = rest.count >= 4 ? max(Int(rest[3]) ?? 60, 1) : 60
            measure(sourceWidth: w, sourceHeight: h, maximumWidth: maxW, count: count)
        } else {
            // Reconstructed sizes against box widths. Within 1.5× the sender
            // now keeps the full frame (2560 box of 3456 is 1.35×).
            let cases = [
                (2560, 1440, 1280),
                (2560, 1440, 1920),
                (2560, 1440, 2560),
                (3456, 1920, 1280),
                (3456, 1920, 1920),
                (3456, 1920, 2560),
                (3456, 1920, 3456),
            ]
            let count = rest.first.flatMap(Int.init).map { max($0, 1) } ?? 60
            for item in cases {
                measure(sourceWidth: item.0, sourceHeight: item.1, maximumWidth: item.2, count: count)
                print()
            }
        }
        exit(0)
    }

    private static func measure(sourceWidth: Int, sourceHeight: Int, maximumWidth: Int, count: Int) {
        guard let buffer = pixelBuffer(width: sourceWidth, height: sourceHeight) else {
            print("delivery-ms could not allocate \(sourceWidth)x\(sourceHeight)")
            return
        }
        let sender = EnhancedFrameSender()
        sender.maximumWidth = maximumWidth
        guard let packet = sender.packet(for: buffer, sequence: 1, session: "delivery-ms") else {
            print("delivery-ms could not pack \(sourceWidth)x\(sourceHeight) maxW=\(maximumWidth)")
            return
        }
        let (outW, outH) = luceSize(packet) ?? (0, 0)
        let megabytes = Double(packet.count) / 1_048_576

        let warmup = 8
        var pack: [Double] = []
        pack.reserveCapacity(count)
        for i in 0..<(warmup + count) {
            let started = ContinuousClock.now
            guard sender.packet(for: buffer, sequence: i, session: "delivery-ms") != nil else { continue }
            if i >= warmup { pack.append((ContinuousClock.now - started).milliseconds) }
        }

        print(String(
            format: "delivery-ms pack  %dx%d → %dx%d  %.2f MB  n=%d  mean %.2f  p50 %.2f  p95 %.2f  max %.2f",
            sourceWidth, sourceHeight, outW, outH, megabytes, pack.count,
            mean(pack), percentile(pack, 0.50), percentile(pack, 0.95), pack.max() ?? 0
        ))

        do {
            let paced = try Loopback.run(packet: packet, count: count, paced: true)
            print(String(
                format: "delivery-ms loop  30fps  offered %d  delivered %d  dropped %d  rx %d  %.0f MB/s  copy %.2f ms",
                paced.offered, paced.delivered, paced.dropped, paced.received,
                paced.megabytesPerSecond, paced.copyMilliseconds
            ))
            let burst = try Loopback.run(packet: packet, count: count, paced: false)
            print(String(
                format: "delivery-ms burst unpaced offered %d  delivered %d  dropped %d  rx %d  %.0f MB/s  %.1f fps",
                burst.offered, burst.delivered, burst.dropped, burst.received,
                burst.megabytesPerSecond, burst.receivedFPS
            ))
        } catch {
            print("delivery-ms loopback failed: \(error)")
        }
    }

    private static func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        let attributes: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferBytesPerRowAlignmentKey as String: 64,
        ]
        guard CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            attributes as CFDictionary, &buffer
        ) == kCVReturnSuccess, let buffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        // Not zeros: a compressible frame would flatter a path that does not
        // compress, and would hide copies that skip empty pages.
        if let y = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) {
            var tone: UInt32 = 0x5A5A5A5A
            memset_pattern4(y, &tone, CVPixelBufferGetBytesPerRowOfPlane(buffer, 0) * CVPixelBufferGetHeightOfPlane(buffer, 0))
        }
        if let uv = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) {
            var chroma: UInt32 = 0x80FF80FF
            memset_pattern4(uv, &chroma, CVPixelBufferGetBytesPerRowOfPlane(buffer, 1) * CVPixelBufferGetHeightOfPlane(buffer, 1))
        }
        return buffer
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

    private static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func percentile(_ values: [Double], _ p: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let i = min(sorted.count - 1, max(0, Int((Double(sorted.count - 1) * p).rounded())))
        return sorted[i]
    }
}

/// Private loopback pair on 127.0.0.1:47911. Same in-flight=1 drop policy
/// as BrowserBridgeServer. TCP rather than WebSocket: the payload is 3–25 MB,
/// so framing is noise, and this must not touch the shipping bridge.
private final class Loopback: @unchecked Sendable {
    struct Result {
        var offered = 0
        var delivered = 0
        var dropped = 0
        var received = 0
        var megabytesPerSecond = 0.0
        var receivedFPS = 0.0
        var copyMilliseconds = 0.0
    }

    static func run(packet: Data, count: Int, paced: Bool) throws -> Result {
        try Loopback(packet: packet).run(count: count, paced: paced)
    }

    private let packet: Data
    private let lock = NSLock()
    private var inFlight = 0
    private var offered = 0
    private var delivered = 0
    private var dropped = 0
    private var received = 0
    private var copyNanos: [UInt64] = []
    private var connection: NWConnection?
    private var listener: NWListener?

    init(packet: Data) { self.packet = packet }

    func run(count: Int, paced: Bool) throws -> Result {
        let parameters = Self.tcpParameters()
        parameters.allowLocalEndpointReuse = true
        parameters.requiredLocalEndpoint = NWEndpoint.hostPort(
            host: "127.0.0.1",
            port: 47911
        )
        let listener = try NWListener(using: parameters)
        self.listener = listener
        let ready = DispatchSemaphore(value: 0)
        var readyError: Error?
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready: ready.signal()
            case .failed(let error): readyError = error; ready.signal()
            default: break
            }
        }
        listener.newConnectionHandler = { [weak self] incoming in
            self?.serve(incoming)
        }
        listener.start(queue: DispatchQueue(label: "delivery-ms.server", qos: .userInteractive))
        guard ready.wait(timeout: .now() + 3) == .success else {
            throw NSError(domain: "delivery-ms", code: 1, userInfo: [NSLocalizedDescriptionKey: "listener did not become ready"])
        }
        if let readyError { throw readyError }
        guard let port = listener.port else { throw NSError(domain: "delivery-ms", code: 1, userInfo: [NSLocalizedDescriptionKey: "no port"]) }

        let client = NWConnection(
            host: NWEndpoint.Host("127.0.0.1"),
            port: port,
            using: Self.tcpParameters()
        )
        let clientReady = DispatchSemaphore(value: 0)
        var clientFailed: Error?
        client.stateUpdateHandler = { state in
            switch state {
            case .ready: clientReady.signal()
            case .failed(let error): clientFailed = error; clientReady.signal()
            default: break
            }
        }
        client.start(queue: DispatchQueue(label: "delivery-ms.client", qos: .userInteractive))
        guard clientReady.wait(timeout: .now() + 3) == .success else {
            throw NSError(domain: "delivery-ms", code: 2, userInfo: [NSLocalizedDescriptionKey: "client did not connect"])
        }
        if let clientFailed { throw clientFailed }
        receive(on: client)

        // The server accepts asynchronously; wait until send() has a target.
        let attachDeadline = Date().addingTimeInterval(2)
        while Date() < attachDeadline {
            lock.lock(); let has = connection != nil; lock.unlock()
            if has { break }
            Thread.sleep(forTimeInterval: 0.01)
        }
        lock.lock(); let hasConnection = connection != nil; lock.unlock()
        guard hasConnection else { throw NSError(domain: "delivery-ms", code: 2, userInfo: [NSLocalizedDescriptionKey: "client did not attach"]) }

        let started = ContinuousClock.now
        for i in 0..<count {
            if paced {
                if i > 0 {
                    let target = started + .nanoseconds(Int64(i) * 33_333_333)
                    let now = ContinuousClock.now
                    if target > now {
                        let ns = (target - now).components
                        let sleepNs = ns.seconds * 1_000_000_000 + ns.attoseconds / 1_000_000_000
                        if sleepNs > 0 { nanosleep_ns(sleepNs) }
                    }
                }
            } else {
                while true {
                    lock.lock(); let busy = inFlight > 0; lock.unlock()
                    if !busy { break }
                    Thread.sleep(forTimeInterval: 0.0005)
                }
            }
            offer()
        }
        let drainDeadline = Date().addingTimeInterval(2)
        while Date() < drainDeadline {
            lock.lock(); let busy = inFlight > 0 || received < delivered; lock.unlock()
            if !busy { break }
            Thread.sleep(forTimeInterval: 0.005)
        }
        let elapsed = max((ContinuousClock.now - started).milliseconds / 1000, 0.001)

        client.cancel()
        listener.cancel()
        Thread.sleep(forTimeInterval: 0.05)

        lock.lock()
        defer { lock.unlock() }
        let copies = copyNanos.map { Double($0) / 1_000_000 }
        return Result(
            offered: offered,
            delivered: delivered,
            dropped: dropped,
            received: received,
            megabytesPerSecond: (Double(delivered) * Double(packet.count) / 1_048_576) / elapsed,
            receivedFPS: Double(received) / elapsed,
            copyMilliseconds: copies.isEmpty ? 0 : copies.reduce(0, +) / Double(copies.count)
        )
    }

    private func serve(_ incoming: NWConnection) {
        incoming.stateUpdateHandler = { [weak self] state in
            if case .failed = state { incoming.cancel() }
            if case .cancelled = state { self?.lock.lock(); self?.connection = nil; self?.lock.unlock() }
        }
        incoming.start(queue: DispatchQueue(label: "delivery-ms.peer", qos: .userInteractive))
        lock.lock()
        connection = incoming
        lock.unlock()
    }

    private func offer() {
        lock.lock()
        offered += 1
        guard let connection, inFlight < 1 else {
            dropped += 1
            lock.unlock()
            return
        }
        inFlight += 1
        delivered += 1
        lock.unlock()
        connection.send(content: packet, contentContext: .defaultMessage, isComplete: true, completion: .contentProcessed { [weak self] _ in
            guard let self else { return }
            self.lock.lock()
            self.inFlight = max(0, self.inFlight - 1)
            self.lock.unlock()
        })
    }

    private func receive(on client: NWConnection) {
        receive(on: client, wanted: packet.count, collected: Data())
    }

    private func receive(on client: NWConnection, wanted: Int, collected: Data) {
        client.receive(minimumIncompleteLength: 1, maximumLength: wanted - collected.count) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            if error != nil || (isComplete && (data == nil || data?.isEmpty == true) && collected.isEmpty) { return }
            var next = collected
            if let data { next.append(data) }
            if next.count >= wanted {
                var scratch = [UInt8](repeating: 0, count: wanted)
                let t0 = DispatchTime.now().uptimeNanoseconds
                next.copyBytes(to: &scratch, count: wanted)
                let dt = DispatchTime.now().uptimeNanoseconds &- t0
                self.lock.lock()
                self.received += 1
                self.copyNanos.append(dt)
                self.lock.unlock()
                self.receive(on: client, wanted: wanted, collected: Data(next.dropFirst(wanted)))
                return
            }
            if error != nil { return }
            self.receive(on: client, wanted: wanted, collected: next)
        }
    }

    private static func tcpParameters() -> NWParameters {
        let parameters = NWParameters.tcp
        parameters.requiredInterfaceType = .loopback
        return parameters
    }
}

private func nanosleep_ns(_ nanoseconds: Int64) {
    var req = timespec(tv_sec: time_t(nanoseconds / 1_000_000_000), tv_nsec: Int(nanoseconds % 1_000_000_000))
    nanosleep(&req, nil)
}
