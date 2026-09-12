import CoreML
import CoreVideo
import Foundation
import Metal

/// What this machine can actually do, measured on this machine, once.
///
/// `LearnedUpscaler.variants` is a table of milliseconds measured on one M4 Pro,
/// checked against a fixed budget. Every Mac that is not that Mac reads its own
/// capability off someone else's measurement: slower silicon selects rungs it
/// cannot sustain and drops frames, faster silicon leaves budget unspent and
/// under-enhances. A lookup keyed on chip name would not fix it - M1 to M1 Max
/// is roughly 4x in GPU throughput under one name, and every unreleased chip is
/// a cache miss that falls back to a guess. So nothing here is keyed on a chip,
/// a generation, or a core count.
///
/// **It also does not model the cost curve, because the curve is not modelable
/// from a couple of points.** The obvious design - time two sizes, fit
/// `ms = a + b * pixels`, predict the rest - was built first and rejected by
/// its own test. Per-kilopixel cost across the shipping ladder falls
/// monotonically from 0.0787 ms at 256x144 to 0.0228 ms at 1280x720, a 3.4x
/// spread: small rungs under-occupy the GPU and large ones amortise fixed
/// costs. A line fitted to the bottom two rungs predicts 38.6 ms at 1280x720
/// against 21.0 measured; fitted to the extremes it predicts 6.9 ms at 640x360
/// against 10.7. Any two-point model is wrong somewhere, and the places it is
/// wrong are the rungs.
///
/// So there is no model. Every rung is timed and the number is kept. That costs
/// a few seconds once, extrapolates nothing, and is automatically right for
/// rungs and silicon that do not exist yet.
///
/// **Measured once, at install.** The record is written to Application Support
/// on the first launch that finds no valid one, which for a Mac app is as close
/// to an install hook as exists. It is explicitly not per launch: the answer
/// does not change between launches, so paying seconds for it every time would
/// be a visible stall for a number already on disk.
struct DeviceCalibration: Codable, Equatable, Sendable {
    /// Measured whole-call milliseconds, keyed by `"<width>x<height>"`.
    let measurements: [String: Double]
    /// What this calibration describes. A mismatch means remeasure; it is never
    /// consulted to *predict* anything.
    let signature: String
    let measuredAt: Date

    static func key(width: Int, height: Int) -> String { "\(width)x\(height)" }

    /// Measured cost for a size, or nil if this size was never timed - in which
    /// case the caller keeps using the bootstrap table rather than guessing.
    func milliseconds(width: Int, height: Int) -> Double? {
        measurements[Self.key(width: width, height: height)]
    }

    // MARK: - Identity

    /// Everything that can change what a millisecond is worth, and nothing else.
    ///
    /// The GPU name appears here as a *cache key*, never as a speed lookup - an
    /// unrecognised name is not a problem, it is simply a different key that
    /// triggers one measurement. New hardware, an OS upgrade that changes the
    /// compiler or scheduler, a new app build, or a different model family all
    /// invalidate the record and remeasure.
    static func signature(modelStem: String) -> String {
        let gpu = MTLCreateSystemDefaultDevice()?.name ?? "unknown-gpu"
        let memory = MTLCreateSystemDefaultDevice().map { String($0.recommendedMaxWorkingSetSize) } ?? "0"
        let version = ProcessInfo.processInfo.operatingSystemVersion
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "0"
        return "\(gpu)|\(memory)|\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)|\(build)|\(modelStem)"
    }

    // MARK: - Persistence

    static func storageURL() -> URL? {
        guard let root = FileManager.default.urls(for: .applicationSupportDirectory,
                                                  in: .userDomainMask).first else { return nil }
        let directory = root.appendingPathComponent("Lucid", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent("device-calibration.json")
    }

    /// The stored calibration, if it is for this machine, this OS, this build
    /// and this model family. Anything else returns nil so the caller remeasures.
    static func stored(modelStem: String) -> DeviceCalibration? {
        guard let url = storageURL(), let data = try? Data(contentsOf: url),
              let record = try? JSONDecoder().decode(DeviceCalibration.self, from: data),
              record.signature == signature(modelStem: modelStem),
              record.isPlausible else { return nil }
        return record
    }

    func save() {
        guard let url = DeviceCalibration.storageURL(),
              let data = try? JSONEncoder().encode(self) else { return }
        try? data.write(to: url, options: .atomic)
    }

    /// A calibration that came back nonsensical - from thermal throttling, a
    /// contended GPU, or a timing fluke - is worse than none, because it
    /// persists and is believed. An hour per frame is not a slow Mac, it is a
    /// broken measurement.
    ///
    /// The monotonicity check is not theoretical. The first real calibration
    /// this code produced ran while the test suite was saturating the machine
    /// and recorded 4.89 ms at 480x270 against 6.40 ms at 432x240 - a *cheaper*
    /// result for 25% more pixels through the same network, which cannot
    /// happen. Range checks alone accepted it. Cost must not fall as input
    /// pixels rise; `tolerance` leaves room for ordinary timing jitter without
    /// admitting an inversion that size.
    var isPlausible: Bool {
        guard !measurements.isEmpty,
              measurements.values.allSatisfy({ $0 > 0 && $0 < 10_000 }) else { return false }
        let tolerance = 0.9
        var ceilingSoFar = 0.0
        for (_, milliseconds) in sortedByPixelCount {
            if milliseconds < ceilingSoFar * tolerance { return false }
            ceilingSoFar = max(ceilingSoFar, milliseconds)
        }
        return true
    }

    /// Measurements ordered by input pixel count, skipping any key that is not
    /// a `WxH` pair rather than trusting the file on disk to be well formed.
    private var sortedByPixelCount: [(pixels: Int, Double)] {
        measurements.compactMap { key, milliseconds in
            let parts = key.split(separator: "x")
            guard parts.count == 2, let width = Int(parts[0]), let height = Int(parts[1]),
                  width > 0, height > 0 else { return nil }
            return (width * height, milliseconds)
        }
        .sorted { $0.pixels < $1.pixels }
    }

    // MARK: - Measurement

    /// Times each size and keeps what it measured. No fitting, no extrapolation.
    ///
    /// A size that throws is left out rather than failing the whole
    /// calibration: one rung whose model will not load should not cost the
    /// machine its measurements for the other six.
    static func measure(sizes: [(width: Int, height: Int)], modelStem: String,
                        time: (Int, Int) throws -> Double) -> DeviceCalibration? {
        var measurements: [String: Double] = [:]
        for size in sizes {
            guard let milliseconds = try? time(size.width, size.height),
                  milliseconds.isFinite, milliseconds > 0 else { continue }
            measurements[key(width: size.width, height: size.height)] = milliseconds
        }
        let calibration = DeviceCalibration(measurements: measurements,
                                            signature: signature(modelStem: modelStem),
                                            measuredAt: Date())
        return calibration.isPlausible ? calibration : nil
    }

    /// Run the measurement if - and only if - this machine has no valid record.
    ///
    /// In practice that is the first launch after install or upgrade. Off the
    /// main thread, and a failure is silent: nothing is written, the bootstrap
    /// table still works, and the next launch retries.
    /// `persist` is injectable so tests can exercise the decision without
    /// writing a real record into the user's Application Support directory -
    /// which the first version of these tests did, leaving a junk calibration
    /// on the developer's machine.
    @discardableResult
    static func calibrateIfNeeded(modelStem: String,
                                  sizes: [(width: Int, height: Int)],
                                  queue: DispatchQueue = .global(qos: .utility),
                                  persist: @escaping @Sendable (DeviceCalibration) -> Void = { $0.save() },
                                  time: @escaping @Sendable (Int, Int) throws -> Double) -> Bool {
        guard !sizes.isEmpty, stored(modelStem: modelStem) == nil else { return false }
        let requested = sizes.map { (width: $0.width, height: $0.height) }
        queue.async {
            if let calibration = measure(sizes: requested, modelStem: modelStem, time: time) {
                persist(calibration)
            }
        }
        return true
    }
}
