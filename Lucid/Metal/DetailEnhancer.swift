//
//  DetailEnhancer.swift
//  Lucid
//
//  Runs the luma detail kernel over a reconstructed frame. Chroma is copied
//  untouched, so this can only change sharpness, never colour.
//

import CoreVideo
import Foundation
import Metal

struct DetailSettings: Equatable, Sendable {
    /// Contrast-adaptive sharpening strength on the finest scale.
    var sharpness: Float = 0.85
    /// Extra straight gain on the finest band.
    var fine: Float = 0.35
    /// Gain on the true one-pixel band. `fine` and `mid` are measured at the
    /// detail scale, which after a 4x upscale is three or four pixels wide, so
    /// neither of them touches the frequency the display actually resolves.
    var micro: Float = 0.0
    /// Spacing for the adaptive lobe, as a fraction of `radius`. 1 keeps the
    /// lobe at the detail scale; smaller values tighten it toward one pixel.
    var lobeScale: Float = 1.0
    /// Signed gain on the coarse band. Negative pulls back the blobby
    /// over-emphasis the scaler leaves behind.
    var mid: Float = -0.30
    var flatThreshold: Float = 0.004
    var edgeThreshold: Float = 0.030
    var deblock: Float = 0.30
    /// Source-resolution deblocking: luma differences below this (0-1) are
    /// treated as compression noise and smoothed before scaling. 0 disables.
    var sourceDeblock: Float = 0.035
    var sourceDeblockRadius: Float = 1.6
    /// Unsharp gain applied at source resolution, before the scaler runs.
    var presharpen: Float = 0.0
    /// How much the deblocker is allowed to set its own strength from the
    /// picture. 0 uses `sourceDeblock` literally; 1 scales it entirely by how
    /// much blocking the frame actually has. This is what removes the need for
    /// a per-site setting.
    var adaptive: Float = 1.0
    /// Gated temporal accumulation at source resolution. 0 disables.
    var temporal: Float = 0.5
    var motionLow: Float = 0.02
    var motionHigh: Float = 0.08
    /// How many pixels one decoded video pixel spans in the frame being
    /// filtered. Sets the scale every kernel works at.
    var radius: Int = 1
    /// The radius the gains below were chosen at. A wider kernel amplifies
    /// lower frequencies, which carry more energy, so the same number produces
    /// a stronger picture at a bigger upscale - which is why a setting tuned on
    /// one site looked wrong on the next. The gains are normalised to this.
    var referenceRadius: Int = 4
    /// Tone grade. Compression lifts blacks and flattens contrast; these put
    /// them back. Points are in normalised video-range luma.
    var blackPoint: Float = 0.02
    var whitePoint: Float = 0.99
    var contrast: Float = 0.18
    var saturation: Float = 1.10

    // Stage toggles. Each is independent so they can be judged one at a time.
    var stageLoopFilter = false
    var stageCdef = false
    var stageDeband = false
    var stageTaa = false
    var stageOklab = false
    /// Blind quantiser for the loop filter: higher filters harder.
    var loopFilterQuant: Float = 32
    var cdefPrimary: Float = 4
    var cdefSecondary: Float = 2
    var debandThreshold: Float = 0.008
    var debandRadius: Float = 16
    var debandIterations: Float = 2
    var grain: Float = 0.010
    var taaGamma: Float = 1.25
    var taaFeedback: Float = 0.90
    var skinProtect: Float = 1.0
    /// Frame counter, so stochastic stages do not stand still.
    var frame: Float = 0

    static let off = DetailSettings(
        sharpness: 0, fine: 0, micro: 0, lobeScale: 1, mid: 0, flatThreshold: 0, edgeThreshold: 1,
        deblock: 0, sourceDeblock: 0, sourceDeblockRadius: 0,
        presharpen: 0, temporal: 0, motionLow: 0, motionHigh: 1, radius: 1,
        blackPoint: 0, whitePoint: 1, contrast: 0, saturation: 1,
        stageLoopFilter: false, stageCdef: false, stageDeband: false, stageTaa: false, stageOklab: false,
        grain: 0
    )
}

/// Metal Shading Language source, compiled at run time so the app does not need
/// the Metal toolchain at build time.
///
/// Detail reconstruction that runs after the 4× SPAN upscale.
///
/// Luma only, so chroma - and therefore colour - cannot shift. Per pixel:
///   * flat areas are smoothed, because in a low-bitrate stream that is where
///     block edges and mosquito noise live and sharpening only amplifies them;
///   * structured areas get their high-frequency component amplified in
///     proportion to how strongly structured they are;
///   * the result is clamped to a band around the local minimum and maximum, so
///     edges cannot ring or halo.
private let shaderSource = """
#include <metal_stdlib>
using namespace metal;


inline float sign_of(float v) { return (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f); }

struct DetailParams {
    int radius;             // one source pixel spans this many pixels here
    float sharpness;        // 0 = off, 1 = maximum lobe
    float fine;             // extra amplification of the finest band
    float micro;            // amplification of the true one-pixel band
    float lobeScale;        // lobe spacing as a fraction of radius
    float mid;              // signed: negative suppresses coarse structure
    float flatThreshold;    // local deviation below which a pixel counts as flat
    float edgeThreshold;    // local deviation at which activation reaches 1
    float deblock;          // how strongly flat areas are smoothed
};

// Contrast-adaptive sharpening on the finest scale only.
//
// An unsharp mask with a wide radius amplifies whatever structure sits at that
// radius, which after a 4x upscale means coarse blobs: small details end up
// looking like part of something larger. This instead works on immediately
// adjacent pixels, and scales the effect by how much local contrast is already
// present, so flat regions and already-strong edges are left alone and only
// genuine fine structure is lifted. The result cannot leave the range of the
// pixels it was computed from, so no halos and no ringing.
kernel void detail_enhance_luma(texture2d<float, access::read>  source      [[texture(0)]],
                                texture2d<float, access::write> destination [[texture(1)]],
                                constant DetailParams&          params      [[buffer(0)]],
                                uint2                           gid         [[thread_position_in_grid]])
{
    const int width = int(source.get_width());
    const int height = int(source.get_height());
    if (int(gid.x) >= width || int(gid.y) >= height) { return; }

    // Sampling at the scale the picture actually carries detail. After a
    // browser upscale one decoded pixel covers several pixels here, so a
    // radius-1 kernel would sharpen a band that holds nothing.
    const int r = max(params.radius, 1);
    const int x = int(gid.x), y = int(gid.y);
    const int xm = max(x - r, 0), xp = min(x + r, width - 1);
    const int ym = max(y - r, 0), yp = min(y + r, height - 1);

    const int xm2 = max(x - 2 * r, 0), xp2 = min(x + 2 * r, width - 1);
    const int ym2 = max(y - 2 * r, 0), yp2 = min(y + 2 * r, height - 1);

    // The lobe may run tighter than the detail scale.
    const int rl = clamp(int(round(float(r) * params.lobeScale)), 1, r);
    const int xml = max(x - rl, 0), xpl = min(x + rl, width - 1);
    const int yml = max(y - rl, 0), ypl = min(y + rl, height - 1);

    const float e = source.read(uint2(x, y)).r;      // centre
    const float b = source.read(uint2(x, ym)).r;     // up
    const float d = source.read(uint2(xm, y)).r;     // left
    const float f = source.read(uint2(xp, y)).r;     // right
    const float h = source.read(uint2(x, yp)).r;     // down
    const float a = source.read(uint2(xm, ym)).r;
    const float c = source.read(uint2(xp, ym)).r;
    const float g = source.read(uint2(xm, yp)).r;
    const float i = source.read(uint2(xp, yp)).r;

    // Same pattern at double spacing: the coarse band.
    // Spacing of exactly one pixel: the band the panel resolves.
    const int x1m = max(x - 1, 0), x1p = min(x + 1, width - 1);
    const int y1m = max(y - 1, 0), y1p = min(y + 1, height - 1);
    const float a1 = source.read(uint2(x1m, y1m)).r;
    const float b1 = source.read(uint2(x,   y1m)).r;
    const float c1 = source.read(uint2(x1p, y1m)).r;
    const float d1 = source.read(uint2(x1m, y  )).r;
    const float f1 = source.read(uint2(x1p, y  )).r;
    const float g1 = source.read(uint2(x1m, y1p)).r;
    const float h1 = source.read(uint2(x,   y1p)).r;
    const float i1 = source.read(uint2(x1p, y1p)).r;

    const float a2 = source.read(uint2(xm2, ym2)).r;
    const float b2 = source.read(uint2(x,   ym2)).r;
    const float c2 = source.read(uint2(xp2, ym2)).r;
    const float d2 = source.read(uint2(xm2, y  )).r;
    const float f2 = source.read(uint2(xp2, y  )).r;
    const float g2 = source.read(uint2(xm2, yp2)).r;
    const float h2 = source.read(uint2(x,   yp2)).r;
    const float i2 = source.read(uint2(xp2, yp2)).r;

    // How structured is this neighbourhood? Flat means compression damage.
    const float mean = (a + b + c + d + e + f + g + h + i) / 9.0f;
    const float meanSq = (a*a + b*b + c*c + d*d + e*e + f*f + g*g + h*h + i*i) / 9.0f;
    const float deviation = sqrt(max(meanSq - mean * mean, 0.0f));
    const float activation = smoothstep(params.flatThreshold, params.edgeThreshold, deviation);

    // Flat areas: fade toward the local average to suppress blocking.
    const float blur = (a + 2.0f * b + c + 2.0f * d + 4.0f * e + 2.0f * f + g + 2.0f * h + i) / 16.0f;
    const float base = mix(mix(e, blur, params.deblock), e, activation);

    // Adaptive lobe over the four direct neighbours: the strength each pixel
    // can take is set by how close it already sits to the local extremes.
    const float bl = source.read(uint2(x, yml)).r;
    const float dl = source.read(uint2(xml, y)).r;
    const float fl = source.read(uint2(xpl, y)).r;
    const float hl = source.read(uint2(x, ypl)).r;
    const float mn = min(min(min(bl, dl), min(fl, hl)), e);
    const float mx = max(max(max(bl, dl), max(fl, hl)), e);
    const float eps = 1.0e-4f;
    const float hitMin = mn / (4.0f * max(mx, eps));
    const float hitMax = (1.0f - mx) / max(4.0f * mn - 4.0f, -4.0f + eps);
    float lobe = max(-hitMin, hitMax);
    lobe = clamp(lobe, -0.1875f, 0.0f) * params.sharpness * activation;
    const float sharpened = (lobe * (bl + dl + fl + hl) + base) / (4.0f * lobe + 1.0f);

    // Band rebalancing. Measured against ground truth, the scaler returns only
    // about half the reference's fine detail while over-producing the coarse
    // band by a quarter, which is what makes small details read as part of
    // something larger. So the fine band is lifted and the coarse band is
    // pulled back toward the reference's balance.
    const float blurWide = (a2 + 2.0f * b2 + c2 + 2.0f * d2 + 4.0f * e + 2.0f * f2 + g2 + 2.0f * h2 + i2) / 16.0f;
    const float blurTight = (a1 + 2.0f * b1 + c1 + 2.0f * d1 + 4.0f * e
                             + 2.0f * f1 + g1 + 2.0f * h1 + i1) / 16.0f;
    const float fineBand = e - blur;
    const float midBand = blur - blurWide;
    const float microBand = e - blurTight;
    float enhanced = sharpened + activation * (params.fine * fineBand
                                               + params.micro * microBand
                                               + params.mid * midBand);

    // Never leave the range of the neighbourhood the bands came from, so
    // neither the lift nor the suppression can ring.
    // (grade is applied after clamping, below)
    const float lo1 = min(min(min(a1, b1), min(c1, d1)), min(min(f1, g1), min(h1, i1)));
    const float hi1 = max(max(max(a1, b1), max(c1, d1)), max(max(f1, g1), max(h1, i1)));
    const float lo = min(lo1,
                     min(min(min(min(min(a, b), min(c, d)), min(min(e, f), min(g, h))), i),
                         min(min(min(a2, b2), min(c2, d2)), min(min(f2, g2), min(h2, i2)))));
    const float hi = max(hi1,
                     max(max(max(max(max(a, b), max(c, d)), max(max(e, f), max(g, h))), i),
                         max(max(max(a2, b2), max(c2, d2)), max(max(f2, g2), max(h2, i2)))));
    enhanced = clamp(enhanced, lo, hi);

    destination.write(float4(enhanced, 0.0f, 0.0f, 1.0f), gid);
}

// Source cleanup, before scaling.
//
// Compression leaves small steps between flat regions (block edges, mosquito
// noise). A scaler cannot tell those from real edges, so it sharpens them into
// visible blocks. This smooths differences below a quantisation-sized
// threshold while leaving larger, real edges alone: a bilateral filter with a
// tight range, run once at the small source size where it is nearly free.
struct DeblockParams {
    float presharpen;       // unsharp gain applied before the scaler sees the frame
    float adaptive;         // 0 = use `range` as given, 1 = scale it by the measurement
    int radius;         // one source pixel spans this many pixels here
    float range;        // luma difference (0-1) treated as compression noise
    float spatial;      // spatial falloff in pixels
    float temporal;     // how much of the previous cleaned frame to keep, 0 = off
    float motionLow;    // frame difference below which a pixel is static
    float motionHigh;   // frame difference above which a pixel is moving
};

// Compression noise differs from frame to frame while the picture underneath
// does not, so where a pixel has not moved, averaging it with the previous
// cleaned frame removes noise that no single-frame filter can. The blend is
// gated per pixel by the frame difference, so motion never smears.
kernel void deblock_luma(texture2d<float, access::read>  source      [[texture(0)]],
                         texture2d<float, access::read>  history     [[texture(1)]],
                         texture2d<float, access::write> destination [[texture(2)]],
                         texture2d<float, access::write> historyOut  [[texture(3)]],
                         constant DeblockParams&         params      [[buffer(0)]],
                         device const uint*              stats       [[buffer(1)]],
                         uint2                           gid         [[thread_position_in_grid]])
{
    // How much more step energy sits on the transform grid than off it. Around
    // zero means a clean stream and the filter should stay out of the way;
    // a large excess means visible blocking and it should work.
    float damage = 1.0f;
    if (params.adaptive > 0.0f && stats[2] > 0u && stats[3] > 0u) {
        const float onGrid  = float(stats[0]) / float(stats[2]);
        const float offGrid = float(stats[1]) / float(stats[3]);
        const float excess = (onGrid - offGrid) / max(offGrid, 1.0f);
        damage = mix(1.0f, clamp(excess * 6.0f, 0.0f, 2.0f), params.adaptive);
    }
    const int width = int(source.get_width());
    const int height = int(source.get_height());
    if (int(gid.x) >= width || int(gid.y) >= height) { return; }

    const float center = source.read(gid).r;
    const int step = max(params.radius, 1);
    float spatialResult = center;
    const float range = params.range * damage;
    if (range > 0.0f) {
        const float rangeK = -0.5f / max(range * range, 1e-6f);
        const float spatialK = -0.5f / max(params.spatial * params.spatial, 1e-6f);
        float sum = 0.0f, weight = 0.0f;
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                uint2 c = uint2(clamp(int(gid.x) + dx * step, 0, width - 1), clamp(int(gid.y) + dy * step, 0, height - 1));
                const float v = source.read(c).r;
                const float d = v - center;
                const float w = exp(d * d * rangeK + float(dx * dx + dy * dy) * spatialK);
                sum += v * w;
                weight += w;
            }
        }
        spatialResult = sum / weight;
    }

    // Sharpening applied here runs at source resolution, where one pixel is
    // one real sample, and the scaler then reconstructs from the sharpened
    // signal rather than having detail pushed onto its output afterwards.
    if (params.presharpen > 0.0f) {
        const int xm = max(int(gid.x) - 1, 0), xp = min(int(gid.x) + 1, width - 1);
        const int ym = max(int(gid.y) - 1, 0), yp = min(int(gid.y) + 1, height - 1);
        const float n0 = source.read(uint2(xm, ym)).r, n1 = source.read(uint2(gid.x, ym)).r;
        const float n2 = source.read(uint2(xp, ym)).r, n3 = source.read(uint2(xm, gid.y)).r;
        const float n5 = source.read(uint2(xp, gid.y)).r, n6 = source.read(uint2(xm, yp)).r;
        const float n7 = source.read(uint2(gid.x, yp)).r, n8 = source.read(uint2(xp, yp)).r;
        const float blur = (n0 + 2.0f * n1 + n2 + 2.0f * n3 + 4.0f * spatialResult
                            + 2.0f * n5 + n6 + 2.0f * n7 + n8) / 16.0f;
        const float lo = min(min(min(n0, n1), min(n2, n3)), min(min(n5, n6), min(n7, n8)));
        const float hi = max(max(max(n0, n1), max(n2, n3)), max(max(n5, n6), max(n7, n8)));
        spatialResult = clamp(spatialResult + params.presharpen * (spatialResult - blur), lo, hi);
    }

    float result = spatialResult;
    if (params.temporal > 0.0f) {
        const float previous = history.read(gid).r;
        const float motion = smoothstep(params.motionLow, params.motionHigh, abs(spatialResult - previous));
        result = mix(spatialResult, previous, params.temporal * (1.0f - motion));
    }
    destination.write(float4(result, 0.0f, 0.0f, 1.0f), gid);
    historyOut.write(float4(result, 0.0f, 0.0f, 1.0f), gid);
}

// Tone grade. A compressed stream arrives with its blacks lifted and its
// contrast flattened, which reads as washed out however sharp it is. Work in
// video range (16-235) so the levels mean what they should.
kernel void grade_luma(texture2d<float, access::read>  source      [[texture(0)]],
                       texture2d<float, access::write> destination [[texture(1)]],
                       constant float4&                params      [[buffer(0)]],
                       device const uint*              stats       [[buffer(1)]],
                       uint2                           gid         [[thread_position_in_grid]])
{
    if (gid.x >= source.get_width() || gid.y >= source.get_height()) { return; }
    const float blackPoint = params.x, whitePoint = params.y, contrast = params.z;
    const float lo709 = 16.0f / 255.0f, range709 = 219.0f / 255.0f;
    float y = (source.read(gid).r - lo709) / range709;
    const float span = max(whitePoint - blackPoint, 1e-3f);
    y = clamp((y - blackPoint) / span, 0.0f, 1.0f);
    const float sCurve = y * y * (3.0f - 2.0f * y);   // deepens shadows, lifts highlights
    y = mix(y, sCurve, clamp(contrast, 0.0f, 1.0f));
    // Grain, at output resolution and fused here so it costs nothing extra.
    // Debanding replaces a hard quantisation step with a smoothed one and the
    // eye still finds the residual; a little noise decorrelates it. Interleaved
    // gradient noise is not true blue noise but has none of white noise's
    // low-frequency energy, and the per-frame offset keeps it from standing
    // still. params.w carries strength, and the frame index rides in its sign.
    // Grain is scaled to how much real detail this frame has, and that is not
    // a refinement - a fixed amount is wrong by an order of magnitude between
    // clips. Measured on two valid pairs: on a textured clip the stage moved
    // fine-band energy by 0.057, on a smooth one by 1.366, from the same
    // setting. On that smooth clip the output carried 1.55x the reference's
    // high-frequency energy and 88% of it was grain - the model reconstructed
    // 0.18 and grain invented the rest.
    //
    // stats[1]/stats[3] is the mean off-grid second difference, already
    // computed for the adaptive deblocker: real detail, measured off the
    // transform grid so blocking does not count as texture. Grain is held to a
    // fraction of it, with a floor so a genuinely flat gradient still gets
    // enough noise to break banding, which is what the stage is for.
    float grain = abs(params.w);
    if (grain > 0.0f && stats[3] > 0u) {
        const float detail = float(stats[1]) / float(stats[3]);
        // 0.02 is roughly the off-grid figure of a normally textured frame.
        const float scale = clamp(detail / 0.02f, 0.35f, 1.0f);
        grain *= scale;
    }
    if (grain > 0.0f) {
        const float2 p = float2(gid) + fract(params.w * 100.0f) * 5.588238f;
        const float ign = fract(52.9829189f * fract(0.06711056f * p.x + 0.00583715f * p.y));
        y = clamp(y + (ign - 0.5f) * grain, 0.0f, 1.0f);
    }
    destination.write(float4(clamp(y * range709 + lo709, lo709, lo709 + range709), 0.0f, 0.0f, 1.0f), gid);
}

// Colour lives in the second plane as two channels centred on neutral, so
// pushing them away from centre is a straight saturation control.
kernel void grade_chroma(texture2d<float, access::read>  source      [[texture(0)]],
                         texture2d<float, access::write> destination [[texture(1)]],
                         constant float&                 saturation  [[buffer(0)]],
                         uint2                           gid         [[thread_position_in_grid]])
{
    if (gid.x >= source.get_width() || gid.y >= source.get_height()) { return; }
    const float2 c = source.read(gid).rg;
    const float2 graded = clamp(0.5f + (c - 0.5f) * saturation, 0.0f, 1.0f);
    destination.write(float4(graded.x, graded.y, 0.0f, 1.0f), gid);
}

// ---------------------------------------------------------------------------
// Source-resolution stages. Everything here runs BEFORE the scaler, because
// every one of these algorithms is calibrated on a one-pixel artifact: after a
// 4x upscale a block edge is a twelve-pixel ramp and none of the tests fire.
// Running here is also sixteen times fewer pixels.
// ---------------------------------------------------------------------------

struct LoopFilterParams {
    float alpha;        // spec Table 8-16 threshold on |p0 - q0|
    float beta;         // spec Table 8-16 threshold on |p1 - p0| and |q1 - q0|
    float tc0;          // spec Table 8-17 clip, boundary strength 2
    int   vertical;     // 1 filters vertical edges, 0 horizontal
};

// The normative H.264 deblocking filter run blind: we have no bitstream, so
// boundary strength is fixed at 2 and the quantiser is a tuning knob. Each
// thread owns one pixel and works out its own role relative to the nearest
// 4-pixel transform boundary, so every thread reads only the input frame and
// no two threads write the same pixel.
kernel void loop_filter_luma(texture2d<float, access::read>  source      [[texture(0)]],
                             texture2d<float, access::write> destination [[texture(1)]],
                             constant LoopFilterParams&      params      [[buffer(0)]],
                             uint2                           gid         [[thread_position_in_grid]])
{
    const int width = int(source.get_width()), height = int(source.get_height());
    const int x = int(gid.x), y = int(gid.y);
    if (x >= width || y >= height) { return; }

    const int along = params.vertical ? x : y;
    const int limit = params.vertical ? width : height;
    const int m = along & 3;
    // The boundary this pixel belongs to, expressed as the position of q0.
    const int b = (m < 2) ? (along - m) : (along + 4 - m);

    float centre = source.read(gid).r;
    if (b < 3 || b + 2 >= limit) { destination.write(float4(centre, 0, 0, 1), gid); return; }

    // Six samples straddling the boundary, scaled to the 0-255 the tables use.
    float v[6];
    for (int k = 0; k < 6; ++k) {
        const int c = b - 3 + k;
        const uint2 at = params.vertical ? uint2(uint(c), gid.y) : uint2(gid.x, uint(c));
        v[k] = source.read(at).r * 255.0f;
    }
    const float p2 = v[0], p1 = v[1], p0 = v[2], q0 = v[3], q1 = v[4], q2 = v[5];

    float out = centre * 255.0f;
    if (abs(p0 - q0) < params.alpha && abs(p1 - p0) < params.beta && abs(q1 - q0) < params.beta) {
        const float midpoint = floor((p0 + q0 + 1.0f) * 0.5f);
        float tc = params.tc0;
        float p1n = p1, q1n = q1;
        if (abs(p2 - p0) < params.beta) {
            p1n = p1 + clamp(floor((p2 + midpoint - 2.0f * p1) * 0.5f), -params.tc0, params.tc0);
            tc += 1.0f;
        }
        if (abs(q2 - q0) < params.beta) {
            q1n = q1 + clamp(floor((q2 + midpoint - 2.0f * q1) * 0.5f), -params.tc0, params.tc0);
            tc += 1.0f;
        }
        const float delta = clamp(floor(((q0 - p0) * 4.0f + (p1 - q1) + 4.0f) / 8.0f), -tc, tc);
        if (along == b)          { out = q0 - delta; }
        else if (along == b - 1) { out = p0 + delta; }
        else if (along == b + 1) { out = q1n; }
        else if (along == b - 2) { out = p1n; }
    }
    destination.write(float4(clamp(out / 255.0f, 0.0f, 1.0f), 0, 0, 1), gid);
}

// CDEF direction search, one thread per 8x8 block. Maximises the sum of
// squared per-line averages over eight directions; the runner-up gap doubles
// as a variance estimate, which is what lets the filter pick its own strength
// with nothing from the bitstream.
kernel void cdef_direction(texture2d<float, access::read>  source    [[texture(0)]],
                           texture2d<float, access::write> directions [[texture(1)]],
                           uint2                           gid       [[thread_position_in_grid]])
{
    const int width = int(source.get_width()), height = int(source.get_height());
    const int bx = int(gid.x) * 8, by = int(gid.y) * 8;
    if (bx >= width || by >= height) { return; }

    float partial[8][15];
    for (int d = 0; d < 8; ++d) for (int k = 0; k < 15; ++k) partial[d][k] = 0.0f;

    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            const uint2 at = uint2(uint(min(bx + x, width - 1)), uint(min(by + y, height - 1)));
            // Centred so a flat block scores zero in every direction.
            const float px = source.read(at).r * 255.0f - 128.0f;
            partial[0][y + x]        += px;   // 45 degrees
            partial[1][y + (x >> 1)] += px;
            partial[2][y]            += px;   // horizontal
            partial[3][3 + y - (x >> 1)] += px;
            partial[4][7 + y - x]    += px;   // 135 degrees
            partial[5][3 - (y >> 1) + x] += px;
            partial[6][x]            += px;   // vertical
            partial[7][(y >> 1) + x] += px;
        }
    }

    // dav1d's divisor table: lines through an 8x8 block have unequal length.
    const float divTable[7] = { 840.0f, 420.0f, 280.0f, 210.0f, 168.0f, 140.0f, 120.0f };
    float cost[8];
    for (int d = 0; d < 8; ++d) {
        float total = 0.0f;
        if (d == 2 || d == 6) {
            for (int k = 0; k < 8; ++k) total += partial[d][k] * partial[d][k];
            total *= 105.0f;
        } else {
            // The two full-length lines, then the tapering ones in pairs.
            for (int k = 0; k < 15; ++k) {
                const int length = 8 - abs(k - 7);
                if (length <= 0) { continue; }
                total += partial[d][k] * partial[d][k] * ((length >= 8) ? 105.0f : divTable[length - 1]);
            }
        }
        cost[d] = total;
    }

    int best = 0;
    for (int d = 1; d < 8; ++d) if (cost[d] > cost[best]) best = d;
    // Distance to the perpendicular direction: high means strongly directional
    // structure, low means noise. This is the whole blind-strength mechanism.
    const float variance = max(cost[best] - cost[best ^ 4], 0.0f) / 1024.0f;
    directions.write(float4(float(best), variance, 0, 1), gid);
}

struct CdefParams {
    float primary;      // 0-15
    float secondary;    // 0, 1, 2 or 4
    float damping;      // 3-6 for luma
};

// The constrained directional filter. Twelve taps inside a 5x5 window: four
// along the detected direction, eight along the two directions 45 degrees off.
// Every tap is clamped by how far it already sits from the centre, and the
// result is clamped to the range of the taps, so it cannot ring.
kernel void cdef_filter(texture2d<float, access::read>  source     [[texture(0)]],
                        texture2d<float, access::read>  directions [[texture(1)]],
                        texture2d<float, access::write> destination[[texture(2)]],
                        constant CdefParams&            params     [[buffer(0)]],
                        uint2                           gid        [[thread_position_in_grid]])
{
    const int width = int(source.get_width()), height = int(source.get_height());
    const int x = int(gid.x), y = int(gid.y);
    if (x >= width || y >= height) { return; }

    const float centre = source.read(gid).r * 255.0f;
    const float4 info = directions.read(uint2(gid.x / 8, gid.y / 8));
    const int direction = int(info.x);
    const float variance = info.y;

    // Strength falls to zero in flat blocks and rises with how directional the
    // block is, which is how CDEF avoids smearing texture without side data.
    float primary = params.primary;
    if (variance > 0.0f) {
        primary = params.primary * (4.0f + min(log2(max(variance / 64.0f, 1.0f)), 12.0f)) / 16.0f;
    } else {
        primary = 0.0f;
    }
    if (primary <= 0.0f && params.secondary <= 0.0f) {
        destination.write(float4(centre / 255.0f, 0, 0, 1), gid); return;
    }

    // Tap offsets per direction, as (dy, dx) at distance 1 and 2.
    const int2 taps[8][2] = {
        { int2(-1,  1), int2(-2,  2) },
        { int2( 0,  1), int2(-1,  2) },
        { int2( 0,  1), int2( 0,  2) },
        { int2( 0,  1), int2( 1,  2) },
        { int2( 1,  1), int2( 2,  2) },
        { int2( 1,  0), int2( 2,  1) },
        { int2( 1,  0), int2( 2,  0) },
        { int2( 1,  0), int2( 2, -1) },
    };

    const float priShift = max(0.0f, params.damping - floor(log2(max(primary, 1.0f))));
    const float secShift = max(0.0f, params.damping - 1.0f - floor(log2(max(params.secondary, 1.0f))));

    float sum = 0.0f, lo = centre, hi = centre;
    const float priWeight[2] = { 4.0f, 2.0f };
    const float secWeight[2] = { 2.0f, 1.0f };

    for (int k = 0; k < 2; ++k) {
        for (int sign = -1; sign <= 1; sign += 2) {
            const int2 o = taps[direction][k] * sign;
            const int sx = clamp(x + o.y, 0, width - 1), sy = clamp(y + o.x, 0, height - 1);
            const float v = source.read(uint2(uint(sx), uint(sy))).r * 255.0f;
            const float d = v - centre;
            const float c = sign_of(d) * max(0.0f, min(abs(d), primary - floor(abs(d) / exp2(priShift))));
            sum += priWeight[k] * c;
            lo = min(lo, v); hi = max(hi, v);
        }
    }
    for (int s = 0; s < 2; ++s) {
        const int dir = (s == 0) ? ((direction + 2) & 7) : ((direction + 6) & 7);
        for (int k = 0; k < 2; ++k) {
            for (int sign = -1; sign <= 1; sign += 2) {
                const int2 o = taps[dir][k] * sign;
                const int sx = clamp(x + o.y, 0, width - 1), sy = clamp(y + o.x, 0, height - 1);
                const float v = source.read(uint2(uint(sx), uint(sy))).r * 255.0f;
                const float d = v - centre;
                const float c = sign_of(d) * max(0.0f, min(abs(d), params.secondary - floor(abs(d) / exp2(secShift))));
                sum += secWeight[k] * c;
                lo = min(lo, v); hi = max(hi, v);
            }
        }
    }

    const float result = clamp(centre + floor((sum - (sum < 0.0f ? 1.0f : 0.0f) + 8.0f) / 16.0f), lo, hi);
    destination.write(float4(clamp(result / 255.0f, 0.0f, 1.0f), 0, 0, 1), gid);
}

struct DebandParams {
    float threshold;    // difference below which a neighbourhood counts as flat
    float radius;       // search distance in pixels
    float iterations;
    float frame;        // decorrelates the sampling pattern over time
};

// Stochastic debanding. Each iteration samples four points on a rotated square
// at a random angle and a progressively larger radius, and takes the average
// only where it is close enough to the original to be a quantisation plateau
// rather than real structure. Later, wider iterations use a tighter threshold.
kernel void deband_plane(texture2d<float, access::read>  source      [[texture(0)]],
                         texture2d<float, access::write> destination [[texture(1)]],
                         constant DebandParams&          params      [[buffer(0)]],
                         uint2                           gid         [[thread_position_in_grid]])
{
    const int width = int(source.get_width()), height = int(source.get_height());
    if (int(gid.x) >= width || int(gid.y) >= height) { return; }
    float4 centre = source.read(gid);
    float2 result = centre.rg;

    uint h = (gid.x * 73856093u) ^ (gid.y * 19349663u) ^ (uint(params.frame) * 83492791u);
    const int iterations = max(1, int(params.iterations));
    for (int i = 1; i <= iterations; ++i) {
        h = h * 1664525u + 1013904223u;
        const float distance = float(h & 0xFFFFu) / 65535.0f;
        h = h * 1664525u + 1013904223u;
        const float angle = float(h & 0xFFFFu) / 65535.0f * 6.2831853f;
        const float2 o = distance * float(i) * params.radius * float2(cos(angle), sin(angle));
        // Four points 90 degrees apart: a rotated square, uniform in angle.
        const float2 offsets[4] = { float2(o.x, o.y), float2(-o.y, o.x), float2(-o.x, -o.y), float2(o.y, -o.x) };
        float2 average = 0.0f;
        for (int k = 0; k < 4; ++k) {
            const int sx = clamp(int(gid.x) + int(round(offsets[k].x)), 0, width - 1);
            const int sy = clamp(int(gid.y) + int(round(offsets[k].y)), 0, height - 1);
            average += source.read(uint2(uint(sx), uint(sy))).rg;
        }
        average *= 0.25f;
        const float bound = params.threshold / float(i);
        result.r = (abs(result.r - average.r) > bound) ? result.r : average.r;
        result.g = (abs(result.g - average.g) > bound) ? result.g : average.g;
    }
    destination.write(float4(result.r, result.g, centre.b, centre.a), gid);
}

struct TaaParams {
    float feedbackMin;
    float feedbackMax;
    float gamma;        // how many standard deviations the history may sit from the mean
    float valid;        // 0 on the first frame after a reset
};

// Temporal accumulation with neighbourhood clipping. Instead of asking whether
// a pixel moved, this asks whether the history is still a plausible value for
// this neighbourhood: if it is, it is kept almost entirely, and if it is not it
// is pulled to the edge of the plausible range. Static areas integrate their
// noise away and moving areas degrade to a slight blur rather than smearing.
kernel void taa_luma(texture2d<float, access::read>  source      [[texture(0)]],
                     texture2d<float, access::read>  history     [[texture(1)]],
                     texture2d<float, access::write> destination [[texture(2)]],
                     texture2d<float, access::write> historyOut  [[texture(3)]],
                     constant TaaParams&             params      [[buffer(0)]],
                     uint2                           gid         [[thread_position_in_grid]])
{
    const int width = int(source.get_width()), height = int(source.get_height());
    const int x = int(gid.x), y = int(gid.y);
    if (x >= width || y >= height) { return; }

    const float centre = source.read(gid).r;
    if (params.valid < 0.5f) {
        destination.write(float4(centre, 0, 0, 1), gid);
        historyOut.write(float4(centre, 0, 0, 1), gid);
        return;
    }

    float m1 = 0.0f, m2 = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const uint2 at = uint2(uint(clamp(x + dx, 0, width - 1)), uint(clamp(y + dy, 0, height - 1)));
            const float v = source.read(at).r;
            m1 += v; m2 += v * v;
        }
    }
    const float mean = m1 / 9.0f;
    const float sigma = sqrt(max(m2 / 9.0f - mean * mean, 0.0f));
    const float lo = mean - params.gamma * sigma;
    const float hi = mean + params.gamma * sigma;

    const float previous = history.read(gid).r;
    const float clipped = clamp(previous, lo, hi);

    // Where the clipped history still disagrees in luma, trust it less.
    const float difference = abs(centre - clipped) / max(max(centre, clipped), 0.2f);
    const float keep = mix(params.feedbackMin, params.feedbackMax, (1.0f - difference) * (1.0f - difference));
    const float result = mix(centre, clipped, keep);

    destination.write(float4(result, 0, 0, 1), gid);
    historyOut.write(float4(result, 0, 0, 1), gid);
}

// ---------------------------------------------------------------------------
// Perceptual saturation. Scaling CbCr at a fixed Y' is not a pure colour move:
// Y'CbCr is non-constant-luminance, so a saturated red gets visibly brighter as
// well as redder. Doing the same move as a chroma scale in Oklab leaves
// lightness and hue alone. Runs on the chroma plane, a quarter of the pixels,
// and emits the luma correction separately so the full-resolution plane only
// has to be multiplied.
// ---------------------------------------------------------------------------

inline float3 srgb_to_linear(float3 c) {
    return select(c / 12.92f, pow(max((c + 0.055f) / 1.055f, 0.0f), 2.4f), c > 0.04045f);
}
inline float3 linear_to_srgb(float3 c) {
    return select(c * 12.92f, 1.055f * pow(max(c, 0.0f), 1.0f / 2.4f) - 0.055f, c > 0.0031308f);
}

inline float3 linear_to_oklab(float3 c) {
    const float l = 0.4122214708f * c.r + 0.5363325363f * c.g + 0.0514459929f * c.b;
    const float m = 0.2119034982f * c.r + 0.6806995451f * c.g + 0.1073969566f * c.b;
    const float s = 0.0883024619f * c.r + 0.2817188376f * c.g + 0.6299787005f * c.b;
    const float third = 1.0f / 3.0f;
    const float l_ = pow(max(l, 0.0f), third), m_ = pow(max(m, 0.0f), third), s_ = pow(max(s, 0.0f), third);
    return float3(0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_,
                  1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_,
                  0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_);
}
inline float3 oklab_to_linear(float3 lab) {
    const float l_ = lab.x + 0.3963377774f * lab.y + 0.2158037573f * lab.z;
    const float m_ = lab.x - 0.1055613458f * lab.y - 0.0638541728f * lab.z;
    const float s_ = lab.x - 0.0894841775f * lab.y - 1.2914855480f * lab.z;
    const float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    return float3(+4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
                  -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
                  -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s);
}

struct OklabParams {
    float saturation;   // chroma multiplier
    float skinProtect;  // 0 disables, 1 fully damps the boost on skin hues
};

kernel void oklab_chroma(texture2d<float, access::read>  chroma      [[texture(0)]],
                         texture2d<float, access::read>  luma        [[texture(1)]],
                         texture2d<float, access::write> destination [[texture(2)]],
                         texture2d<float, access::write> lumaGain    [[texture(3)]],
                         constant OklabParams&           params      [[buffer(0)]],
                         uint2                           gid         [[thread_position_in_grid]])
{
    if (gid.x >= chroma.get_width() || gid.y >= chroma.get_height()) { return; }
    const float2 cbcr = chroma.read(gid).rg;

    // One chroma sample covers a 2x2 luma block.
    const uint lw = luma.get_width(), lh = luma.get_height();
    float y = 0.0f;
    for (uint dy = 0; dy < 2; ++dy) {
        for (uint dx = 0; dx < 2; ++dx) {
            y += luma.read(uint2(min(gid.x * 2 + dx, lw - 1), min(gid.y * 2 + dy, lh - 1))).r;
        }
    }
    y *= 0.25f;

    const float Y = clamp((y * 255.0f - 16.0f) / 219.0f, 0.0f, 1.0f);
    const float Cb = (cbcr.r * 255.0f - 128.0f) / 224.0f;
    const float Cr = (cbcr.g * 255.0f - 128.0f) / 224.0f;

    const float3 gamma = clamp(float3(Y + 1.5748f * Cr,
                                      Y - 0.1873f * Cb - 0.4681f * Cr,
                                      Y + 1.8556f * Cb), 0.0f, 1.0f);
    float3 lab = linear_to_oklab(srgb_to_linear(gamma));

    float k = params.saturation;
    if (params.skinProtect > 0.0f) {
        // Skin sits on a narrow hue band and should not take the full boost, or
        // faces go orange before anything else has visibly gained.
        const float chromaMag = length(lab.yz);
        const float hue = atan2(lab.z, lab.y) * 57.29578f;
        float d = abs(hue - 45.0f); if (d > 180.0f) d = 360.0f - d;
        float protect = smoothstep(40.0f, 15.0f, d);
        protect *= smoothstep(0.14f, 0.06f, chromaMag);   // saturated orange is not skin
        protect *= smoothstep(0.25f, 0.40f, lab.x) * smoothstep(0.98f, 0.85f, lab.x);
        protect *= params.skinProtect;
        k = mix(k, 1.0f + (k - 1.0f) * 0.35f, protect);
    }
    lab.y *= k; lab.z *= k;

    const float3 outGamma = clamp(linear_to_srgb(clamp(oklab_to_linear(lab), 0.0f, 1.0f)), 0.0f, 1.0f);
    const float Yn = 0.2126f * outGamma.r + 0.7152f * outGamma.g + 0.0722f * outGamma.b;
    const float Cbn = (outGamma.b - Yn) / 1.8556f;
    const float Crn = (outGamma.r - Yn) / 1.5748f;

    destination.write(float4(clamp(Cbn * 224.0f / 255.0f + 128.0f / 255.0f, 0.0f, 1.0f),
                             clamp(Crn * 224.0f / 255.0f + 128.0f / 255.0f, 0.0f, 1.0f), 0, 1), gid);
    // The luma the perceptual move implies, relative to what is already there.
    lumaGain.write(float4(Y > 1e-3f ? clamp(Yn / Y, 0.5f, 2.0f) : 1.0f, 0, 0, 1), gid);
}

// Applies the half-resolution luma correction, interpolated, so the perceptual
// saturation keeps its promise of leaving lightness where it was.
kernel void apply_luma_gain(texture2d<float, access::read>  source      [[texture(0)]],
                            texture2d<float, access::read>  gainMap     [[texture(1)]],
                            texture2d<float, access::write> destination [[texture(2)]],
                            uint2                           gid         [[thread_position_in_grid]])
{
    const int width = int(source.get_width()), height = int(source.get_height());
    if (int(gid.x) >= width || int(gid.y) >= height) { return; }
    const int gw = int(gainMap.get_width()), gh = int(gainMap.get_height());
    const float fx = (float(gid.x) + 0.5f) * 0.5f - 0.5f, fy = (float(gid.y) + 0.5f) * 0.5f - 0.5f;
    const int x0 = clamp(int(floor(fx)), 0, gw - 1), y0 = clamp(int(floor(fy)), 0, gh - 1);
    const int x1 = min(x0 + 1, gw - 1), y1 = min(y0 + 1, gh - 1);
    const float tx = clamp(fx - float(x0), 0.0f, 1.0f), ty = clamp(fy - float(y0), 0.0f, 1.0f);
    const float g = mix(mix(gainMap.read(uint2(uint(x0), uint(y0))).r, gainMap.read(uint2(uint(x1), uint(y0))).r, tx),
                        mix(gainMap.read(uint2(uint(x0), uint(y1))).r, gainMap.read(uint2(uint(x1), uint(y1))).r, tx), ty);
    const float lo = 16.0f / 255.0f, range = 219.0f / 255.0f;
    const float y = (source.read(gid).r - lo) / range;
    destination.write(float4(clamp(y * g * range + lo, lo, lo + range), 0, 0, 1), gid);
}

// ---------------------------------------------------------------------------
// Frame analysis. The point of this is that nobody should have to tell Lucid
// how damaged a stream is. A 200 kbps 240p stream and a 2 Mbps 720p stream need
// very different amounts of deblocking, and the frame itself says which it is:
// compression puts steps on the transform grid that are not there off it.
// ---------------------------------------------------------------------------

// stats[0] = second differences across grid-aligned columns and rows
// stats[1] = the same measured off the grid, as the control
// stats[2] = how many samples went into each
kernel void measure_frame(texture2d<float, access::read> source [[texture(0)]],
                          device atomic_uint*            stats  [[buffer(0)]],
                          uint2                          gid    [[thread_position_in_grid]])
{
    const int width = int(source.get_width()), height = int(source.get_height());
    // One sample per 4x4 tile is plenty to characterise a whole frame.
    const int x = int(gid.x) * 4, y = int(gid.y) * 4;
    if (x < 2 || y < 2 || x >= width - 2 || y >= height - 2) { return; }

    // A block edge is a step: the second difference across it is large while the
    // pixels either side of it are flat. Real detail does not respect the grid,
    // so the same measurement taken off the grid is the baseline to compare to.
    const float left  = source.read(uint2(uint(x - 1), uint(y))).r;
    const float here  = source.read(uint2(uint(x),     uint(y))).r;
    const float right = source.read(uint2(uint(x + 1), uint(y))).r;
    const float step = abs(2.0f * here - left - right);

    const bool onGrid = (x % 8) == 0 || (y % 8) == 0;
    atomic_fetch_add_explicit(&stats[onGrid ? 0 : 1],
                              uint(step * 8192.0f), memory_order_relaxed);
    atomic_fetch_add_explicit(&stats[onGrid ? 2 : 3], 1u, memory_order_relaxed);
}

"""

private struct DeblockParams {
    var presharpen: Float
    var adaptive: Float
    var radius: Int32
    var range: Float
    var spatial: Float
    var temporal: Float
    var motionLow: Float
    var motionHigh: Float
}

private struct LoopFilterParams { var alpha: Float; var beta: Float; var tc0: Float; var vertical: Int32 }
private struct CdefParams { var primary: Float; var secondary: Float; var damping: Float }
private struct DebandParams2 { var threshold: Float; var radius: Float; var iterations: Float; var frame: Float }
private struct TaaParams { var feedbackMin: Float; var feedbackMax: Float; var gamma: Float; var valid: Float }
private struct OklabParams { var saturation: Float; var skinProtect: Float }

private struct DetailParams {
    var radius: Int32
    var sharpness: Float
    var fine: Float
    var micro: Float
    var lobeScale: Float
    var mid: Float
    var flatThreshold: Float
    var edgeThreshold: Float
    var deblock: Float
}

final class DetailEnhancer: @unchecked Sendable {
    enum Failure: Error { case library, pipeline, textureCache, texture, pool }

    var settings: DetailSettings

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let deblockPipeline: MTLComputePipelineState
    private let chromaPipeline: MTLComputePipelineState
    private let lumaGradePipeline: MTLComputePipelineState
    private let loopFilterPipeline: MTLComputePipelineState
    private let cdefDirectionPipeline: MTLComputePipelineState
    private let cdefFilterPipeline: MTLComputePipelineState
    private let debandPipeline: MTLComputePipelineState
    private let measurePipeline: MTLComputePipelineState
    /// Four counters the analysis pass fills in and the deblocker reads, so the
    /// measurement never has to make a round trip through the CPU.
    private var frameStats: MTLBuffer?
    private let taaPipeline: MTLComputePipelineState
    private let oklabPipeline: MTLComputePipelineState
    private let lumaGainPipeline: MTLComputePipelineState
    private var directionTexture: MTLTexture?
    private var gainTexture: MTLTexture?
    /// Destination for the luma correction. Applying it in place would have a
    /// kernel read and write the same texture in one dispatch.
    private var gainScratch: MTLTexture?
    private var scratch: [MTLTexture] = []
    /// Advances per processed frame so the stochastic stages keep moving even
    /// though the settings struct itself is rebuilt only on reconfiguration.
    private var frameIndex: Float = 0
    private var sourcePool: CVPixelBufferPool?
    private var sourcePoolSize = (width: 0, height: 0, format: OSType(0))
    private var history: [MTLTexture] = []
    private var historyIndex = 0
    private var historyValid = false
    private var historyRect: CGRect = .zero
    private var textureCache: CVMetalTextureCache
    private var pool: CVPixelBufferPool?
    private var poolSize = (width: 0, height: 0, format: OSType(0))

    init(device: MTLDevice, settings: DetailSettings = DetailSettings()) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw Failure.pipeline }
        self.queue = queue
        self.settings = settings
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: shaderSource, options: nil)
        } catch {
            // The kernels are compiled at run time, so a mistake in them shows
            // up here and nowhere else. Say what it was.
            print("   ⚠️ Metal compile failed: \(error)")
            throw Failure.library
        }
        guard let function = library.makeFunction(name: "detail_enhance_luma") else { throw Failure.library }
        guard let state = try? device.makeComputePipelineState(function: function),
              let deblockFunction = library.makeFunction(name: "deblock_luma"),
              let deblock = try? device.makeComputePipelineState(function: deblockFunction),
              let chromaFunction = library.makeFunction(name: "grade_chroma"),
              let chroma = try? device.makeComputePipelineState(function: chromaFunction),
              let lumaGradeFunction = library.makeFunction(name: "grade_luma"),
              let lumaGrade = try? device.makeComputePipelineState(function: lumaGradeFunction)
        else { throw Failure.pipeline }
        func makeStage(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name), let p = try? device.makeComputePipelineState(function: f)
            else { throw Failure.pipeline }
            return p
        }
        loopFilterPipeline = try makeStage("loop_filter_luma")
        cdefDirectionPipeline = try makeStage("cdef_direction")
        cdefFilterPipeline = try makeStage("cdef_filter")
        debandPipeline = try makeStage("deband_plane")
        measurePipeline = try makeStage("measure_frame")
        taaPipeline = try makeStage("taa_luma")
        oklabPipeline = try makeStage("oklab_chroma")
        lumaGainPipeline = try makeStage("apply_luma_gain")
        lumaGradePipeline = lumaGrade
        deblockPipeline = deblock
        chromaPipeline = chroma
        pipeline = state
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache) == kCVReturnSuccess,
              let cache else { throw Failure.textureCache }
        textureCache = cache
    }

    /// The factor that makes a gain mean the same thing at any upscale.
    /// `2 - log2(r)/2`, normalised so it is 1 at the reference radius.
    static func gainNormalisation(radius: Int, reference: Int) -> Float {
        func curve(_ r: Int) -> Float { 2 - log2(Float(max(r, 1))) / 2 }
        let value = curve(radius) / max(curve(reference), 0.0001)
        return min(max(value, 0.25), 2.0)
    }

    /// Cleans compression damage out of a source frame before it is scaled.
    /// Luma is filtered, chroma is copied. Returns `source` unchanged when
    /// deblocking is off.
    func preprocess(_ source: CVPixelBuffer, sourceRect: CGRect = .zero, radius: Int = 1) throws -> CVPixelBuffer {
        let runsBilateral = settings.sourceDeblock > 0 || settings.presharpen > 0
            || (settings.temporal > 0 && !settings.stageTaa)
        guard runsBilateral || settings.stageLoopFilter || settings.stageCdef
            || settings.stageDeband || settings.stageTaa else { return source }
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        let format = CVPixelBufferGetPixelFormatType(source)
        let lumaFormat = MetalTileCompositor.metalFormat(for: format, plane: 0)

        if sourcePool == nil || sourcePoolSize != (width, height, format) {
            sourcePool = try Self.makePool(width: width, height: height, format: format)
            sourcePoolSize = (width, height, format)
        }
        var outBuffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, sourcePool!, &outBuffer) == kCVReturnSuccess,
              let out = outBuffer, let commandBuffer = queue.makeCommandBuffer() else { return source }

        var retained: [CVMetalTexture] = []
        let (inLuma, inRef) = try texture(source, plane: 0, format: lumaFormat)
        let (outLuma, outRef) = try texture(out, plane: 0, format: lumaFormat)
        retained.append(inRef); retained.append(outRef)

        // History is only meaningful while the capture rect and size hold
        // still; a re-centred or resized capture starts over.
        if history.count != 2 || history[0].width != width || history[0].height != height {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: width, height: height, mipmapped: false)
            descriptor.usage = [.shaderRead, .shaderWrite]
            descriptor.storageMode = .private
            guard let a = device.makeTexture(descriptor: descriptor), let b = device.makeTexture(descriptor: descriptor) else { throw Failure.texture }
            history = [a, b]
            historyValid = false
        }
        if sourceRect != historyRect { historyValid = false; historyRect = sourceRect }

        // Scratch pair for ping-ponging between stages.
        if scratch.count != 2 || scratch[0].width != width || scratch[0].height != height {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: lumaFormat, width: width, height: height, mipmapped: false)
            descriptor.usage = [.shaderRead, .shaderWrite]
            descriptor.storageMode = .private
            guard let a = device.makeTexture(descriptor: descriptor),
                  let b = device.makeTexture(descriptor: descriptor) else { throw Failure.texture }
            scratch = [a, b]
        }

        frameIndex += 1
        if frameStats == nil {
            frameStats = device.makeBuffer(length: MemoryLayout<UInt32>.stride * 4,
                                           options: .storageModePrivate)
        }
        let threads = MTLSize(width: 16, height: 16, depth: 1)
        func grid(_ w: Int, _ h: Int) -> MTLSize {
            MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1)
        }

        // Measure the frame before anything touches it, so the stages that
        // follow can set their own strength from what is actually there.
        if settings.adaptive > 0, let stats = frameStats {
            if let blit = commandBuffer.makeBlitCommandEncoder() {
                blit.fill(buffer: stats, range: 0..<stats.length, value: 0)
                blit.endEncoding()
            }
            if let e = commandBuffer.makeComputeCommandEncoder() {
                e.setComputePipelineState(measurePipeline)
                e.setTexture(inLuma, index: 0)
                e.setBuffer(stats, offset: 0, index: 0)
                e.dispatchThreadgroups(grid((width + 3) / 4, (height + 3) / 4),
                                       threadsPerThreadgroup: threads)
                e.endEncoding()
            }
        }

        // Each stage is one pass over the plane; the list decides the order and
        // which buffer each one reads and writes.
        var passes: [(MTLTexture, MTLTexture) -> Void] = []

        if settings.stageLoopFilter {
            // Spec Tables 8-16 and 8-17 evaluated at four quantisers; between
            // them the thresholds are interpolated rather than guessed.
            let anchors: [(q: Float, a: Float, b: Float, t: Float)] = [
                (25, 13, 4, 1), (32, 32, 9, 2), (40, 80, 13, 5), (50, 255, 18, 15)
            ]
            let q = min(max(settings.loopFilterQuant, anchors[0].q), anchors[anchors.count - 1].q)
            var lower = anchors[0], upper = anchors[anchors.count - 1]
            for i in 0..<(anchors.count - 1) where q >= anchors[i].q && q <= anchors[i + 1].q {
                lower = anchors[i]; upper = anchors[i + 1]
            }
            let t = upper.q > lower.q ? (q - lower.q) / (upper.q - lower.q) : 0
            let alpha = lower.a + (upper.a - lower.a) * t
            let beta = lower.b + (upper.b - lower.b) * t
            let tc0 = lower.t + (upper.t - lower.t) * t
            for vertical in [Int32(1), Int32(0)] {
                passes.append { src, dst in
                    guard let e = commandBuffer.makeComputeCommandEncoder() else { return }
                    e.setComputePipelineState(self.loopFilterPipeline)
                    e.setTexture(src, index: 0); e.setTexture(dst, index: 1)
                    var p = LoopFilterParams(alpha: alpha, beta: beta, tc0: tc0, vertical: vertical)
                    e.setBytes(&p, length: MemoryLayout<LoopFilterParams>.stride, index: 0)
                    e.dispatchThreadgroups(grid(width, height), threadsPerThreadgroup: threads)
                    e.endEncoding()
                }
            }
        }

        if settings.stageCdef {
            let bw = (width + 7) / 8, bh = (height + 7) / 8
            if directionTexture == nil || directionTexture!.width != bw || directionTexture!.height != bh {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Float, width: bw, height: bh, mipmapped: false)
                d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
                directionTexture = device.makeTexture(descriptor: d)
            }
            let primary = settings.cdefPrimary, secondary = settings.cdefSecondary
            passes.append { src, dst in
                guard let directions = self.directionTexture,
                      let e = commandBuffer.makeComputeCommandEncoder() else { return }
                e.setComputePipelineState(self.cdefDirectionPipeline)
                e.setTexture(src, index: 0); e.setTexture(directions, index: 1)
                e.dispatchThreadgroups(grid(bw, bh), threadsPerThreadgroup: threads)
                e.endEncoding()
                guard let f = commandBuffer.makeComputeCommandEncoder() else { return }
                f.setComputePipelineState(self.cdefFilterPipeline)
                f.setTexture(src, index: 0); f.setTexture(directions, index: 1); f.setTexture(dst, index: 2)
                var p = CdefParams(primary: primary, secondary: secondary, damping: 4)
                f.setBytes(&p, length: MemoryLayout<CdefParams>.stride, index: 0)
                f.dispatchThreadgroups(grid(width, height), threadsPerThreadgroup: threads)
                f.endEncoding()
            }
        }

        if settings.stageDeband {
            let threshold = settings.debandThreshold, radius = settings.debandRadius
            let iterations = settings.debandIterations, frame = frameIndex
            passes.append { src, dst in
                guard let e = commandBuffer.makeComputeCommandEncoder() else { return }
                e.setComputePipelineState(self.debandPipeline)
                e.setTexture(src, index: 0); e.setTexture(dst, index: 1)
                var p = DebandParams2(threshold: threshold, radius: radius, iterations: iterations, frame: frame)
                e.setBytes(&p, length: MemoryLayout<DebandParams2>.stride, index: 0)
                e.dispatchThreadgroups(grid(width, height), threadsPerThreadgroup: threads)
                e.endEncoding()
            }
        }

        if runsBilateral {
            let params = DeblockParams(
                presharpen: settings.presharpen,
                adaptive: settings.adaptive,
                radius: Int32(max(1, radius)),
                range: settings.sourceDeblock, spatial: settings.sourceDeblockRadius,
                temporal: (historyValid && !settings.stageTaa) ? settings.temporal : 0,
                motionLow: settings.motionLow, motionHigh: settings.motionHigh
            )
            passes.append { src, dst in
                guard let e = commandBuffer.makeComputeCommandEncoder() else { return }
                e.setComputePipelineState(self.deblockPipeline)
                e.setBuffer(self.frameStats, offset: 0, index: 1)
                e.setTexture(src, index: 0)
                e.setTexture(self.history[self.historyIndex], index: 1)
                e.setTexture(dst, index: 2)
                e.setTexture(self.history[1 - self.historyIndex], index: 3)
                var p = params
                e.setBytes(&p, length: MemoryLayout<DeblockParams>.stride, index: 0)
                e.dispatchThreadgroups(grid(width, height), threadsPerThreadgroup: threads)
                e.endEncoding()
            }
        }

        if settings.stageTaa {
            let gamma = settings.taaGamma, feedback = settings.taaFeedback
            let valid: Float = historyValid ? 1 : 0
            passes.append { src, dst in
                guard let e = commandBuffer.makeComputeCommandEncoder() else { return }
                e.setComputePipelineState(self.taaPipeline)
                e.setTexture(src, index: 0)
                e.setTexture(self.history[self.historyIndex], index: 1)
                e.setTexture(dst, index: 2)
                e.setTexture(self.history[1 - self.historyIndex], index: 3)
                var p = TaaParams(feedbackMin: max(0, feedback - 0.05), feedbackMax: feedback, gamma: gamma, valid: valid)
                e.setBytes(&p, length: MemoryLayout<TaaParams>.stride, index: 0)
                e.dispatchThreadgroups(grid(width, height), threadsPerThreadgroup: threads)
                e.endEncoding()
            }
        }

        guard !passes.isEmpty else { return source }
        var current = inLuma
        for (index, pass) in passes.enumerated() {
            let destination = index == passes.count - 1 ? outLuma : scratch[index % 2]
            pass(current, destination)
            current = destination
        }
        historyIndex = 1 - historyIndex
        historyValid = true

        if CVPixelBufferGetPlaneCount(source) > 1 {
            for plane in 1..<CVPixelBufferGetPlaneCount(source) {
                let planeFormat = MetalTileCompositor.metalFormat(for: format, plane: plane)
                let (sPlane, sRef) = try texture(source, plane: plane, format: planeFormat)
                let (dPlane, dRef) = try texture(out, plane: plane, format: planeFormat)
                retained.append(sRef); retained.append(dRef)
                // Colour contouring in fades and dark scenes is very visible and
                // the chroma planes are a quarter of the pixels each, so both
                // together cost half of luma.
                if settings.stageDeband, let e = commandBuffer.makeComputeCommandEncoder() {
                    e.setComputePipelineState(debandPipeline)
                    e.setTexture(sPlane, index: 0); e.setTexture(dPlane, index: 1)
                    var p = DebandParams2(threshold: settings.debandThreshold * 1.5,
                                          radius: settings.debandRadius * 0.5,
                                          iterations: settings.debandIterations, frame: frameIndex)
                    e.setBytes(&p, length: MemoryLayout<DebandParams2>.stride, index: 0)
                    e.dispatchThreadgroups(grid(sPlane.width, sPlane.height), threadsPerThreadgroup: threads)
                    e.endEncoding()
                } else if let blit = commandBuffer.makeBlitCommandEncoder() {
                    blit.copy(from: sPlane, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                              sourceSize: MTLSize(width: sPlane.width, height: sPlane.height, depth: 1),
                              to: dPlane, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
                    blit.endEncoding()
                }
            }
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        withExtendedLifetime(retained) {}
        if let attachments = CVBufferCopyAttachments(source, .shouldPropagate) {
            CVBufferSetAttachments(out, attachments, .shouldPropagate)
        }
        return out
    }

    private static func makePool(width: Int, height: Int, format: OSType) throws -> CVPixelBufferPool {
        let attributes: [String: Any] = [
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferPixelFormatTypeKey as String: format,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault,
                                      [kCVPixelBufferPoolMinimumBufferCountKey as String: 4] as CFDictionary,
                                      attributes as CFDictionary, &pool) == kCVReturnSuccess, let pool
        else { throw Failure.pool }
        return pool
    }

    /// Returns an enhanced copy of `reconstructed`. Luma is filtered, chroma is
    /// copied.
    func process(_ reconstructed: CVPixelBuffer) throws -> CVPixelBuffer {
        let doesGrade = settings.blackPoint > 0 || settings.whitePoint < 1 || settings.contrast > 0
        let doesDetail = settings.sharpness > 0 || settings.fine != 0 || settings.micro != 0 || settings.mid != 0 || settings.deblock > 0 || doesGrade
        let width = CVPixelBufferGetWidth(reconstructed)
        let height = CVPixelBufferGetHeight(reconstructed)
        let format = CVPixelBufferGetPixelFormatType(reconstructed)
        let lumaFormat = MetalTileCompositor.metalFormat(for: format, plane: 0)

        guard doesDetail else { return reconstructed }

        guard let commandBuffer = queue.makeCommandBuffer() else { return reconstructed }
        var retained: [CVMetalTexture] = []
        var buffers: [CVPixelBuffer] = []
        func nextBuffer() throws -> CVPixelBuffer {
            let b = try buffer(width: width, height: height, format: format)
            buffers.append(b)
            return b
        }

        var currentBuffer = reconstructed
        var currentLuma: MTLTexture
        do {
            let (t, r) = try texture(reconstructed, plane: 0, format: lumaFormat)
            retained.append(r)
            currentLuma = t
        }

        let threads = MTLSize(width: 16, height: 16, depth: 1)
        func groups(_ w: Int, _ h: Int) -> MTLSize {
            MTLSize(width: (w + threads.width - 1) / threads.width,
                    height: (h + threads.height - 1) / threads.height, depth: 1)
        }

        if doesDetail {
            let out = try nextBuffer()
            let (outLuma, outRef) = try texture(out, plane: 0, format: lumaFormat)
            retained.append(outRef)
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return reconstructed }
            encoder.setComputePipelineState(pipeline)
            encoder.setTexture(currentLuma, index: 0)
            encoder.setTexture(outLuma, index: 1)
            // Measured, not guessed: holding the settings fixed and sweeping
            // the gain at each radius, the multipliers that reproduce the
            // reference radius's band energy are 1.5, 1.0 and 0.5 at radius
            // 2, 4 and 8 - exactly linear in log2(radius).
            let scale = Self.gainNormalisation(radius: settings.radius,
                                               reference: settings.referenceRadius)
            // The lobe is what produces crispness, and crispness is a property
            // of the display grid, not of the source. Hold it at a fixed pixel
            // spacing instead of letting it widen with the upscale.
            let lobeSpacing = settings.lobeScale * Float(settings.referenceRadius)
            let lobe = min(max(lobeSpacing / Float(max(settings.radius, 1)), 0.12), 1.0)
            var params = DetailParams(
                radius: Int32(max(1, settings.radius)),
                // Only the radius-scaled terms are normalised. `micro` works at
                // a fixed one-pixel spacing, so it is already scale-free and
                // normalising it would make the fine band drift the other way -
                // it is the term that holds fine detail steady across scales.
                sharpness: settings.sharpness * scale, fine: settings.fine * scale,
                micro: settings.micro, lobeScale: lobe,
                mid: settings.mid * scale,
                flatThreshold: settings.flatThreshold, edgeThreshold: settings.edgeThreshold,
                deblock: settings.deblock
            )
            encoder.setBytes(&params, length: MemoryLayout<DetailParams>.stride, index: 0)
            encoder.dispatchThreadgroups(groups(width, height), threadsPerThreadgroup: threads)
            encoder.endEncoding()
            currentBuffer = out
            currentLuma = outLuma
        }

        // Tone grade last, so sharpening works on untouched levels.
        if doesGrade {
            let out = try nextBuffer()
            let (outLuma, outRef) = try texture(out, plane: 0, format: lumaFormat)
            retained.append(outRef)
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(lumaGradePipeline)
                encoder.setTexture(currentLuma, index: 0)
                encoder.setTexture(outLuma, index: 1)
                // w carries grain strength; its fractional part phases the
                // noise so it does not stand still between frames.
                // Grain exists to cover the residual of debanding, so it
                // follows that switch rather than running on its own.
                let grainTerm = settings.stageDeband && settings.grain > 0
                    ? settings.grain + (frameIndex.truncatingRemainder(dividingBy: 64) / 100)
                    : 0
                var params = SIMD4<Float>(settings.blackPoint, settings.whitePoint, settings.contrast, grainTerm)
                encoder.setBytes(&params, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
                // The same per-frame statistics the adaptive deblocker uses, so
                // grain can be held to a fraction of the detail actually present
                // rather than added at a fixed amount to every kind of picture.
                encoder.setBuffer(frameStats, offset: 0, index: 1)
                encoder.dispatchThreadgroups(groups(width, height), threadsPerThreadgroup: threads)
                encoder.endEncoding()
                currentBuffer = out
                currentLuma = outLuma
            }
        }

        guard currentBuffer !== reconstructed else { return reconstructed }

        // Chroma: graded for saturation, or copied straight through.
        if CVPixelBufferGetPlaneCount(reconstructed) > 1 {
            for plane in 1..<CVPixelBufferGetPlaneCount(reconstructed) {
                let planeFormat = MetalTileCompositor.metalFormat(for: format, plane: plane)
                let (sourcePlane, sRef) = try texture(reconstructed, plane: plane, format: planeFormat)
                let (destinationPlane, dRef) = try texture(currentBuffer, plane: plane, format: planeFormat)
                retained.append(sRef); retained.append(dRef)
                if settings.stageOklab, abs(settings.saturation - 1) > 0.001,
                   let encoder = commandBuffer.makeComputeCommandEncoder() {
                    // Perceptual saturation needs the luma that goes with each
                    // chroma sample, and returns the luma correction that keeps
                    // lightness where it was.
                    if gainTexture == nil || gainTexture!.width != sourcePlane.width
                        || gainTexture!.height != sourcePlane.height {
                        let d = MTLTextureDescriptor.texture2DDescriptor(
                            pixelFormat: .r16Float, width: sourcePlane.width,
                            height: sourcePlane.height, mipmapped: false)
                        d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
                        gainTexture = device.makeTexture(descriptor: d)
                    }
                    encoder.setComputePipelineState(oklabPipeline)
                    encoder.setTexture(sourcePlane, index: 0)
                    encoder.setTexture(currentLuma, index: 1)
                    encoder.setTexture(destinationPlane, index: 2)
                    encoder.setTexture(gainTexture, index: 3)
                    var params = OklabParams(saturation: settings.saturation, skinProtect: settings.skinProtect)
                    encoder.setBytes(&params, length: MemoryLayout<OklabParams>.stride, index: 0)
                    encoder.dispatchThreadgroups(groups(sourcePlane.width, sourcePlane.height), threadsPerThreadgroup: threads)
                    encoder.endEncoding()
                    if plane == 1, let gain = gainTexture {
                        if gainScratch == nil || gainScratch!.width != width || gainScratch!.height != height {
                            let d = MTLTextureDescriptor.texture2DDescriptor(
                                pixelFormat: lumaFormat, width: width, height: height, mipmapped: false)
                            d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
                            gainScratch = device.makeTexture(descriptor: d)
                        }
                        if let scratchLuma = gainScratch,
                           let apply = commandBuffer.makeComputeCommandEncoder() {
                            apply.setComputePipelineState(lumaGainPipeline)
                            apply.setTexture(currentLuma, index: 0)
                            apply.setTexture(gain, index: 1)
                            apply.setTexture(scratchLuma, index: 2)
                            apply.dispatchThreadgroups(groups(width, height), threadsPerThreadgroup: threads)
                            apply.endEncoding()
                            if let blit = commandBuffer.makeBlitCommandEncoder() {
                                blit.copy(from: scratchLuma, sourceSlice: 0, sourceLevel: 0,
                                          sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                                          sourceSize: MTLSize(width: width, height: height, depth: 1),
                                          to: currentLuma, destinationSlice: 0, destinationLevel: 0,
                                          destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
                                blit.endEncoding()
                            }
                        }
                    }
                } else if abs(settings.saturation - 1) > 0.001, let encoder = commandBuffer.makeComputeCommandEncoder() {
                    encoder.setComputePipelineState(chromaPipeline)
                    encoder.setTexture(sourcePlane, index: 0)
                    encoder.setTexture(destinationPlane, index: 1)
                    var saturation = settings.saturation
                    encoder.setBytes(&saturation, length: MemoryLayout<Float>.stride, index: 0)
                    encoder.dispatchThreadgroups(groups(sourcePlane.width, sourcePlane.height), threadsPerThreadgroup: threads)
                    encoder.endEncoding()
                } else if let blit = commandBuffer.makeBlitCommandEncoder() {
                    blit.copy(
                        from: sourcePlane, sourceSlice: 0, sourceLevel: 0,
                        sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                        sourceSize: MTLSize(width: sourcePlane.width, height: sourcePlane.height, depth: 1),
                        to: destinationPlane, destinationSlice: 0, destinationLevel: 0,
                        destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
                    )
                    blit.endEncoding()
                }
            }
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        withExtendedLifetime(retained) {}

        if let attachments = CVBufferCopyAttachments(reconstructed, .shouldPropagate) {
            CVBufferSetAttachments(currentBuffer, attachments, .shouldPropagate)
        }
        return currentBuffer
    }

    private func texture(_ buffer: CVPixelBuffer, plane: Int, format: MTLPixelFormat) throws -> (MTLTexture, CVMetalTexture) {
        let planar = CVPixelBufferIsPlanar(buffer)
        let width = planar ? CVPixelBufferGetWidthOfPlane(buffer, plane) : CVPixelBufferGetWidth(buffer)
        let height = planar ? CVPixelBufferGetHeightOfPlane(buffer, plane) : CVPixelBufferGetHeight(buffer)
        var ref: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, buffer, nil, format, width, height, plane, &ref
        )
        guard status == kCVReturnSuccess, let ref, let texture = CVMetalTextureGetTexture(ref) else { throw Failure.texture }
        return (texture, ref)
    }

    private func buffer(width: Int, height: Int, format: OSType) throws -> CVPixelBuffer {
        if pool == nil || poolSize != (width, height, format) {
            let attributes: [String: Any] = [
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferPixelFormatTypeKey as String: format,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
            var created: CVPixelBufferPool?
            guard CVPixelBufferPoolCreate(
                kCFAllocatorDefault, [kCVPixelBufferPoolMinimumBufferCountKey as String: 4] as CFDictionary,
                attributes as CFDictionary, &created) == kCVReturnSuccess, let created
            else { throw Failure.pool }
            pool = created
            poolSize = (width, height, format)
        }
        var out: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool!, &out) == kCVReturnSuccess, let out
        else { throw Failure.pool }
        return out
    }
}
