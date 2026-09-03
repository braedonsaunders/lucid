//
//  EnhancementPipeline.swift
//  Lucid
//
//  Off-main-actor stage graph: capture stream → tiled super resolution →
//  timestamp-paced presentation. Backpressure comes from the capture stream's
//  newest-frame buffering: if processing lags, older frames are dropped and
//  the pipeline never queues unboundedly.
//

import CoreMedia
import CoreVideo
import Foundation

struct PipelineStats: Sendable {
    var sourceFPS: Double = 0
    var outputFPS: Double = 0
    var processingMilliseconds: Double = 0
    var tileCount: Int = 0
    var outputSize: CGSize = .zero
    var lastError: String?
}

/// Everything that depends on the frame size or the chosen engine.
struct PipelineStages: @unchecked Sendable {
    let upscaler: TiledVideoToolboxUpscaler?
    let detail: DetailEnhancer?
    let inputWidth: Int
    let inputHeight: Int
    let preprocessRadius: Int
    let label: String
}

@available(macOS 26.0, *)
actor EnhancementPipeline {
    typealias StageFactory = @Sendable (_ width: Int, _ height: Int) throws -> PipelineStages

    private var stages: PipelineStages?
    private var factory: StageFactory
    private var generation = 0
    private let presenter: FramePresenter
    /// Set when the page draws the result itself, which is the preferred path.
    private var onEnhancedFrame: (@Sendable (CVPixelBuffer) -> Void)?

    func setEnhancedFrameHandler(_ handler: @escaping @Sendable (CVPixelBuffer) -> Void) {
        onEnhancedFrame = handler
    }
    private let onStats: @Sendable (PipelineStats) -> Void
    private let onFirstFrame: @Sendable () -> Void
    private var task: Task<Void, Never>?

    private var sourceFrames = 0
    private var outputFrames = 0
    private var processingEMA: Double = 0
    private var windowStart = ContinuousClock.now
    private var lastError: String?
    private var producedFrame = false
    private var lastOutputSize: CGSize = .zero

    init(
        factory: @escaping StageFactory,
        presenter: FramePresenter,
        onFirstFrame: @escaping @Sendable () -> Void,
        onStats: @escaping @Sendable (PipelineStats) -> Void
    ) {
        self.factory = factory
        self.presenter = presenter
        self.onFirstFrame = onFirstFrame
        self.onStats = onStats
    }

    /// Decoded frames from the page: no errors, newest wins.
    func run(_ stream: AsyncStream<CapturedFrame>) {
        task?.cancel()
        task = Task { [weak self] in
            for await frame in stream {
                guard !Task.isCancelled, let self else { return }
                await self.process(frame)
            }
        }
    }

    func run(_ stream: AsyncThrowingStream<CapturedFrame, Error>) {
        task?.cancel()
        task = Task { [weak self] in
            do {
                for try await frame in stream {
                    guard !Task.isCancelled, let self else { return }
                    await self.process(frame)
                }
            } catch {
                guard let self else { return }
                await self.record(error: "capture stream: \(error.localizedDescription)")
            }
        }
    }

    /// Applies new detail settings to the stages already running. Tuning does
    /// not change geometry, so rebuilding the scaler sessions for it costs a
    /// visible stall - long enough for the page to decide the video has gone
    /// away - for no benefit.
    func updateDetail(_ transform: @Sendable (DetailSettings) -> DetailSettings) {
        guard let detail = stages?.detail else { return }
        detail.settings = transform(detail.settings)
    }

    /// Swaps the stage factory (engine change, quality change). Stages are
    /// rebuilt on the next frame; capture keeps running throughout.
    func reconfigure(_ factory: @escaping StageFactory) async {
        self.factory = factory
        if let old = stages { await old.upscaler?.end() }
        stages = nil
    }

    func stop() async {
        let running = task
        task = nil
        running?.cancel()
        // Let the in-flight frame finish before the sessions go away.
        await running?.value
        await stages?.upscaler?.end()
        stages = nil
        presenter.flush()
    }

    private func record(error: String) {
        lastError = error
    }

    private func process(_ frame: CapturedFrame) async {
        sourceFrames += 1
        let started = ContinuousClock.now
        do {
            let width = CVPixelBufferGetWidth(frame.pixelBuffer)
            let height = CVPixelBufferGetHeight(frame.pixelBuffer)
            if stages == nil || stages!.inputWidth != width || stages!.inputHeight != height {
                if let old = stages { await old.upscaler?.end() }
                let built = try factory(width, height)
                stages = built
                print("   🔬 \(built.label)")
            }
            guard let stages else { return }
            let cleaned = (try? stages.detail?.preprocess(frame.pixelBuffer, sourceRect: frame.sourceRect, radius: stages.preprocessRadius)) ?? frame.pixelBuffer
            let reconstructed: CVPixelBuffer
            if let upscaler = stages.upscaler {
                reconstructed = try await upscaler.upscale(cleaned, pts: frame.presentationTimestamp)
            } else {
                reconstructed = cleaned
            }
            let output = (try? stages.detail?.process(reconstructed, original: frame.pixelBuffer)) ?? reconstructed
            let elapsed = (ContinuousClock.now - started).milliseconds
            processingEMA = processingEMA == 0 ? elapsed : processingEMA * 0.9 + elapsed * 0.1
            lastOutputSize = CGSize(width: CVPixelBufferGetWidth(output), height: CVPixelBufferGetHeight(output))
            if let onEnhancedFrame {
                onEnhancedFrame(output)
            } else {
                presenter.present(output, capturePTS: frame.presentationTimestamp, sourceRect: frame.sourceRect)
            }
            outputFrames += 1
            lastError = nil
            if !producedFrame {
                producedFrame = true
                onFirstFrame()
            }
        } catch is CancellationError {
            // Shutting down or switching engines; not a failure.
        } catch {
            lastError = "\(error)"
        }
        publishStatsIfDue()
    }

    private func publishStatsIfDue() {
        let elapsed = (ContinuousClock.now - windowStart).milliseconds / 1000
        guard elapsed >= 1 else { return }
        let stats = PipelineStats(
            sourceFPS: Double(sourceFrames) / elapsed,
            outputFPS: Double(outputFrames) / elapsed,
            processingMilliseconds: processingEMA,
            tileCount: stages?.upscaler?.totalTileCount ?? 0,
            outputSize: lastOutputSize,
            lastError: lastError
        )
        sourceFrames = 0
        outputFrames = 0
        windowStart = .now
        onStats(stats)
    }
}

extension Duration {
    var milliseconds: Double {
        let parts = components
        return Double(parts.seconds) * 1000 + Double(parts.attoseconds) / 1e15
    }
}
