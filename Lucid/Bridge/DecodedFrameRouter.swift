import Foundation

/// One pending decoded frame across the bridge/CPU boundary. All pixel copies
/// and color conversion run here, never in an unbounded series of main-actor tasks.
final class DecodedFrameRouter: @unchecked Sendable {
    private let lock = NSLock()
    private let queue = DispatchQueue(label: "com.lucid.decode", qos: .userInitiated)
    private var target: (String, DecodedFrameSource)?
    private var pending: DecodedFrame?
    private var scheduled = false

    func install(session: String?, source: DecodedFrameSource?) {
        lock.lock(); defer { lock.unlock() }
        target = session.flatMap { id in source.map { (id, $0) } }
        pending = nil
    }

    func accept(_ frame: DecodedFrame) {
        lock.lock()
        guard target?.0 == frame.header.session else { lock.unlock(); return }
        pending = frame
        let start = !scheduled
        scheduled = true
        lock.unlock()
        if start { queue.async { [self] in drain() } }
    }

    private func drain() {
        while true {
            lock.lock()
            guard let frame = pending, let target, target.0 == frame.header.session else {
                scheduled = false; lock.unlock(); return
            }
            pending = nil
            lock.unlock()
            target.1.accept(frame)
        }
    }
}
