import Foundation
import Testing

@testable import Lucid

/// The calibration exists so that no chip name, generation or core count is
/// ever consulted to predict speed. These tests are mostly about that property
/// holding on machines that do not exist yet.
@Suite(.serialized)
struct DeviceCalibrationTests {

    private func calibration(_ pairs: [(Int, Int, Double)]) -> DeviceCalibration {
        var measurements: [String: Double] = [:]
        for (width, height, milliseconds) in pairs {
            measurements[DeviceCalibration.key(width: width, height: height)] = milliseconds
        }
        return DeviceCalibration(measurements: measurements, signature: "test", measuredAt: Date())
    }

    /// Roughly four times slower than the reference machine - the M1 end of the
    /// range - but named nowhere and derived from nothing.
    private var slowMachine: DeviceCalibration {
        calibration(LearnedUpscaler.variants.map { ($0.width, $0.height, $0.milliseconds * 4) })
    }

    private var fastMachine: DeviceCalibration {
        calibration(LearnedUpscaler.variants.map { ($0.width, $0.height, $0.milliseconds / 2.5) })
    }

    @Test func measuresEveryRungRatherThanFittingACurve() throws {
        // The rejected design fitted ms = a + b*pixels from two points. Per
        // kilopixel the real ladder runs 0.0787 down to 0.0228 ms - a 3.4x
        // spread - so any such fit is wrong at the rungs. Measuring is the fix.
        var timed: [String] = []
        let fitted = try #require(DeviceCalibration.measure(
            sizes: LearnedUpscaler.calibrationSizes, modelStem: "test") { width, height in
                timed.append("\(width)x\(height)")
                return LearnedUpscaler.variants
                    .first { $0.width == width && $0.height == height }!.milliseconds
            })
        #expect(timed.count == LearnedUpscaler.variants.count)
        for variant in LearnedUpscaler.variants {
            #expect(fitted.milliseconds(width: variant.width, height: variant.height) == variant.milliseconds)
        }
    }

    @Test func reportsNothingForSizesItNeverTimed() {
        let partial = calibration([(640, 360, 12.0)])
        #expect(partial.milliseconds(width: 640, height: 360) == 12.0)
        // Not a guess, not an interpolation - nil, so the caller falls back.
        #expect(partial.milliseconds(width: 1280, height: 720) == nil)
    }

    @Test func aSlowMachineDeclinesRungsItCannotSustain() {
        LearnedUpscaler.calibrationOverride = slowMachine
        defer { LearnedUpscaler.calibrationOverride = nil }
        // The bootstrap table says 640x360 costs 10.7 ms and is always eligible.
        // Here it costs 42.8 ms, so a 30 fps budget must refuse it...
        #expect(LearnedUpscaler.costMilliseconds(width: 640, height: 360) > 33.0)
        #expect(LearnedUpscaler.variant(width: 640, height: 360, budget: 33.0) == nil)
        // ...while a small source still finds a rung. Slow hardware should lose
        // coverage gradually, not be written off entirely.
        #expect(LearnedUpscaler.variant(width: 256, height: 144, budget: 33.0) != nil)
    }

    @Test func aFastMachineSpendsBudgetTheTableWouldLeaveUnspent() {
        LearnedUpscaler.calibrationOverride = fastMachine
        defer { LearnedUpscaler.calibrationOverride = nil }
        // 1280x720 is 21.0 ms in the table, so a 60 fps budget rejects it. On
        // hardware this fast it is 8.4 ms and should be taken.
        #expect(LearnedUpscaler.variant(width: 1280, height: 720, budget: 16.7)?.width == 1280)
    }

    @Test func theSameSourceGetsDifferentAnswersOnDifferentMachines() {
        // The whole point, stated as one assertion: identical input, identical
        // code, different measured hardware, different decision.
        LearnedUpscaler.calibrationOverride = slowMachine
        let slow = LearnedUpscaler.variant(width: 640, height: 360, budget: 33.0)
        LearnedUpscaler.calibrationOverride = fastMachine
        let fast = LearnedUpscaler.variant(width: 640, height: 360, budget: 33.0)
        LearnedUpscaler.calibrationOverride = nil
        #expect(slow == nil)
        #expect(fast != nil)
    }

    @Test func budgetFollowsTheFrameRateRatherThanAConstant() {
        #expect(abs(LearnedUpscaler.budget(forFrameRate: 30) - 33.333) < 0.01)
        #expect(abs(LearnedUpscaler.budget(forFrameRate: 60) - 16.667) < 0.01)
        #expect(abs(LearnedUpscaler.budget(forFrameRate: 24) - 41.667) < 0.01)
        #expect(LearnedUpscaler.budget(forFrameRate: 0) == .infinity)
    }

    @Test func calibrationSizesCoverEveryRungTheSelectorCanChoose() {
        // If a rung can be selected but was never timed, the selector is back to
        // reading someone else's machine for that one size.
        let sizes = Set(LearnedUpscaler.calibrationSizes.map { "\($0.width)x\($0.height)" })
        for variant in LearnedUpscaler.variants {
            #expect(sizes.contains("\(variant.width)x\(variant.height)"))
        }
    }

    @Test func withoutACalibrationTheBootstrapTableIsUsedUnchanged() {
        LearnedUpscaler.calibrationOverride = nil
        guard LearnedUpscaler.activeCalibration == nil else { return }
        for variant in LearnedUpscaler.variants {
            #expect(LearnedUpscaler.costMilliseconds(width: variant.width,
                                                    height: variant.height) == variant.milliseconds)
        }
    }

    @Test func nonsenseMeasurementsAreRejectedRatherThanPersisted() {
        // A bad calibration is worse than none: it is written to disk and
        // believed until the next OS upgrade.
        #expect(DeviceCalibration.measure(sizes: [(640, 360)], modelStem: "test") { _, _ in -1.0 } == nil)
        #expect(DeviceCalibration.measure(sizes: [(640, 360)], modelStem: "test") { _, _ in .infinity } == nil)
        #expect(DeviceCalibration.measure(sizes: [], modelStem: "test") { _, _ in 5.0 } == nil)
        #expect(calibration([(640, 360, 99_999.0)]).isPlausible == false)
    }

    @Test func oneUnusableRungDoesNotCostTheOthersTheirMeasurements() {
        struct Unloadable: Error {}
        let partial = DeviceCalibration.measure(
            sizes: [(256, 144), (640, 360)], modelStem: "test") { width, _ in
                if width == 640 { throw Unloadable() }
                return 2.9
            }
        #expect(partial?.milliseconds(width: 256, height: 144) == 2.9)
        #expect(partial?.milliseconds(width: 640, height: 360) == nil)
    }

    @Test func aCalibrationThatGotCheaperWithMorePixelsIsRejected() {
        // These are the real numbers the first end-to-end run produced while the
        // test suite was saturating the machine: 480x270 came back cheaper than
        // 432x240, through the same network, for 25% more pixels. Range checks
        // accepted it. A persisted calibration is believed until the next OS
        // upgrade, so this one has to be thrown away instead.
        let contended = calibration([(256, 144, 4.152), (320, 180, 4.277), (432, 240, 6.401),
                                     (480, 270, 4.892), (640, 360, 6.184), (864, 480, 11.562),
                                     (1280, 720, 22.231)])
        #expect(contended.isPlausible == false)
        #expect(DeviceCalibration.measure(sizes: [(432, 240), (480, 270)], modelStem: "test") {
            width, _ in width == 432 ? 6.401 : 4.892
        } == nil)
        // Ordinary jitter in the right direction must still be accepted, or a
        // quiet machine never gets calibrated at all.
        let jittery = calibration([(432, 240, 5.80), (480, 270, 5.75), (640, 360, 10.7)])
        #expect(jittery.isPlausible)
    }

    @Test func aMonotonicCalibrationOnAQuietMachineIsAccepted() {
        #expect(calibration(LearnedUpscaler.variants.map {
            ($0.width, $0.height, $0.milliseconds) }).isPlausible)
        #expect(slowMachine.isPlausible)
        #expect(fastMachine.isPlausible)
    }

    @Test func signatureChangesWithTheModelFamily() {
        #expect(DeviceCalibration.signature(modelStem: "lucidbig2k_")
                != DeviceCalibration.signature(modelStem: "other_"))
    }

    @Test func calibrationRunsWhenNoRecordExistsAndIsSkippedOtherwise() {
        // `persist` is stubbed: a test must not write a record into the real
        // Application Support directory, which the first version of this did.
        let discard: @Sendable (DeviceCalibration) -> Void = { _ in }
        // An unmatchable stem guarantees no stored record, so this one runs.
        #expect(DeviceCalibration.calibrateIfNeeded(
            modelStem: "unlikely-stem-\(UUID().uuidString)",
            sizes: LearnedUpscaler.calibrationSizes, persist: discard) { _, _ in 5.0 })
        // No sizes means nothing to measure; it must not spin up a queue.
        #expect(DeviceCalibration.calibrateIfNeeded(
            modelStem: "unlikely-stem-\(UUID().uuidString)",
            sizes: [], persist: discard) { _, _ in 5.0 } == false)
    }
}
