import Foundation

struct PresentationAck: Codable, Sendable {
    var session: String
    var seq: Int
    var latencyMilliseconds: Double
}

struct PresentationMetrics: Sendable {
    var session = ""
    var framesPerSecond: Double = 0
    var medianMilliseconds: Double = 0
    var p95Milliseconds: Double = 0
    var p99Milliseconds: Double = 0
    var samples = 0
}

/// Browser capture → canvas draw submission, not physical screen scanout.
struct PresentationWindow {
    private var started: ContinuousClock.Instant
    init(now: ContinuousClock.Instant = .now) { started = now }
    private var latencies: [Double] = []
    private var lastSequence = -1

    mutating func record(_ ack: PresentationAck, now: ContinuousClock.Instant = .now) -> PresentationMetrics? {
        guard ack.seq > lastSequence, ack.latencyMilliseconds.isFinite,
              (0...2000).contains(ack.latencyMilliseconds) else { return nil }
        lastSequence = ack.seq
        latencies.append(ack.latencyMilliseconds)
        let elapsed = (now - started).milliseconds / 1000
        guard elapsed >= 1 else { return nil }
        let sorted = latencies.sorted()
        func percentile(_ p: Double) -> Double { sorted[min(sorted.count - 1, max(0, Int(ceil(Double(sorted.count) * p)) - 1))] }
        let metrics = PresentationMetrics(session: ack.session, framesPerSecond: Double(sorted.count) / elapsed,
            medianMilliseconds: percentile(0.5), p95Milliseconds: percentile(0.95), p99Milliseconds: percentile(0.99), samples: sorted.count)
        started = now; latencies.removeAll(keepingCapacity: true)
        return metrics
    }
}
