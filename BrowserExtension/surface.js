// Lucid browser companion — drawing surface.
//
// Holds its own socket to the native app and paints the enhanced frames it is
// sent. Lives at the extension's origin so the page's CSP cannot stop it
// connecting; see surface.html for why that is the whole point of this file.
(() => {
  const BRIDGE_URL = 'ws://127.0.0.1:47811';
  const ENHANCED_MAGIC = 0x4c554345; // 'LUCE'
  const session = location.hash.slice(1);
  if (!session) return;

  const canvas = document.getElementById('surface');
  const context = canvas.getContext('2d', { alpha: true, desynchronized: true });
  let lastNV12 = null;
  // WebGL2 final-stretch (r93, banking r87/r88/r89): the browser's own canvas
  // magnification on this element is bilinear, measured to cost 5.86-9.85%
  // LPIPS / 3.37-4.11% DISTS vs a Lanczos-quality stretch at 1.5x-3x display
  // scale (see .build/quality-breakthrough-r89-realstretch/README.md). This
  // module owns that final stretch with a WebGL2 separable Lanczos-3 pass
  // instead of leaving it to the browser, and is a pure additive presentation
  // layer: `canvas`/`context` above are drawn into exactly as before by
  // paint(), unchanged, and remain what's on screen whenever WebGL2 is
  // unavailable, disabled, or the target box isn't a magnification.
  const gls = createGLStretch();
  let socket = null;
  let connecting = false;
  let backoff = 400;
  let lastFrameAt = 0;
  let frozen = false;
  let enabled = false, comparing = false, scheduled = false, newest = null, lastSequence = -1;
  function clear() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    gls.clear();
    newest = null; lastNV12 = null; lastFrameAt = 0;
  }
  // Height, in device pixels, of the strip at the bottom left clear so the
  // browser's own video controls show through. The content script measures it.
  let gap = 0;
  let capturePort = null;
  function captureReady() {
    capturePort?.postMessage({type: 'captureReady', session,
      ready: !frozen && enabled && socket?.readyState === 1});
  }
  function receiveCapture(event) {
    const message = event.data;
    if (message?.t !== 'capture' || message.session !== session || !(message.bytes instanceof Uint8Array)) return;
    const bytes = message.bytes;
    // This channel forwards only decoded-frame packets for this iframe's own
    // active session. It never forwards hello, tokens, attach or control JSON.
    if (bytes.byteLength < 8 || bytes.byteLength > 32 * 1024 * 1024) return;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (view.getUint32(0, false) !== 0x4c554346) return;
    const length = view.getUint32(4, false);
    if (length > 8192 || length + 8 >= bytes.byteLength) return;
    let header;
    try { header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8+length))); } catch { return; }
    if (!header || header.session !== session || !Number.isSafeInteger(header.seq) || header.seq < 0) return;
    if (frozen || !enabled || socket?.readyState !== 1 || socket.bufferedAmount > 2 * 1024 * 1024) {
      capturePort?.postMessage({type: 'captureReleased', session, seq: header.seq});
      return;
    }
    try { socket.send(bytes); } catch {
      capturePort?.postMessage({type: 'captureReleased', session, seq: header.seq});
    }
  }

  async function connect() {
    if (frozen || connecting) return;
    if (socket && (socket.readyState === 0 || socket.readyState === 1)) return;
    connecting = true;
    let token;
    try {
      token = await lucidFetchToken();
    } catch (e) {
      connecting = false;
      retry();
      return;
    }
    try { socket = new WebSocket(BRIDGE_URL); } catch (e) { connecting = false; socket = null; retry(); return; }
    socket.binaryType = 'arraybuffer';
    socket.onopen = () => {
      connecting = false;
      backoff = 400;
      // Token first: the bridge drops anything else until it has seen hello.
      try { socket.send(lucidHello(token)); } catch (e) {}
      // Tell the app which video's frames to send here. Without this the bridge
      // has no way to know this connection belongs to that session.
      socket.send(JSON.stringify({ type: 'attach', session }));
    };
    socket.onclose = () => { connecting = false; socket = null; captureReady(); retry(); };
    socket.onerror = () => {};
    socket.onmessage = (event) => {
      if (!(event.data instanceof ArrayBuffer)) {
        try {
          const message = JSON.parse(event.data);
          if (message.type === 'accepted' && message.session === session) capturePort?.postMessage(message);
          if (message.type === 'status') {
            comparing = message.comparing === true;
            canvas.style.visibility = comparing ? 'hidden' : '';
            enabled = message.enabled && message.activeSession === session;
            if (!enabled) clear();
            captureReady();
          }
        } catch {}
        return;
      }
      const view = new DataView(event.data);
      if (view.byteLength < 8 || view.getUint32(0, false) !== ENHANCED_MAGIC) return;
      const headerLength = view.getUint32(4, false);
      let meta;
      try {
        meta = JSON.parse(new TextDecoder().decode(new Uint8Array(event.data, 8, headerLength)));
      } catch (e) { return; }
      if (meta.session !== session || !enabled || comparing || meta.seq <= lastSequence) return;
      const pixels = new Uint8Array(event.data, 8 + headerLength);
      if (!Number.isInteger(meta.w) || !Number.isInteger(meta.h) || meta.w < 2 || meta.h < 2 || meta.w * meta.h > 16777216) return;
      if (pixels.byteLength < meta.w * meta.h * 1.5) return;
      newest = { meta, pixels };
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(() => {
        scheduled = false;
        const next = newest; newest = null;
        if (!next || !enabled || comparing) return;
        const latency = next.meta.captureTime ? performance.timeOrigin + performance.now() - next.meta.captureTime : 0;
        if (latency > 150 || latency < 0) return;
        if (!draw(next.meta.w, next.meta.h, next.pixels, next.meta.format, next.meta)) return;
        lastSequence = next.meta.seq;
        if (next.meta.captureTime && socket?.readyState === 1) socket.send(JSON.stringify({
          type: 'presented', session, seq: next.meta.seq,
          latencyMilliseconds: performance.timeOrigin + performance.now() - next.meta.captureTime
        }));
      });
    };
  }

  function retry() {
    if (frozen) return;
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 6000);
  }

  function draw(width, height, pixels, format, meta) {
    if (width < 2 || height < 2) return false;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    lastNV12 = { width, height, pixels, format, meta };
    if (!paint()) return false;
    lastFrameAt = performance.now();
    return true;
  }

  function paint() {
    if (!lastNV12 || comparing) return false;
    const { width, height, pixels, format, meta } = lastNV12;
    if (format !== 'NV12' || typeof VideoFrame !== 'function') return false;
    try {
      const frame = new VideoFrame(pixels, {
        format: 'NV12',
        codedWidth: width,
        codedHeight: height,
        timestamp: Math.max(0, meta?.ts || 0),
        colorSpace: meta?.colorSpace || { primaries: "bt709", transfer: "bt709", matrix: "bt709", fullRange: false },
        layout: [
          { offset: 0, stride: width },
          { offset: width * height, stride: width },
        ],
      });
      context.drawImage(frame, 0, 0, width, height);
      frame.close();
    } catch (e) {
      return false;
    }
    if (gap > 0) context.clearRect(0, height - Math.min(gap, height), width, Math.min(gap, height));
    // Present through the WebGL2 final-stretch when it applies (pure
    // magnification in both axes); otherwise this is a no-op and the 2D
    // canvas above, with its existing CSS 100% stretch, is what's on screen.
    gls.present(canvas, width, height);
    return true;
  }

  // The content script measures the control strip and reports whether the
  // browser is currently showing it.
  addEventListener('message', (event) => {
    if (event.source !== parent) return;
    const message = event.data;
    if (message?.lucid === 'capture-port' && message.session === session && event.ports.length === 1) {
      // A page owns its input video. Bind its port only to this session and let
      // authenticated native status remain the authority for enablement.
      capturePort?.close(); capturePort = event.ports[0];
      capturePort.onmessage = receiveCapture;
      captureReady();
      return;
    }
    if (message?.lucid === 'clear') { clear(); return; }
    if (message?.lucid === 'compare') {
      comparing = message.active === true;
      canvas.style.visibility = comparing ? 'hidden' : 'visible';
      gls.setVisibility(comparing ? 'hidden' : 'visible');
      return;
    }
    // Testing switch: postMessage({lucid:'glStretch', enabled}) to the surface
    // iframe's contentWindow toggles the WebGL2 path on/off live. Not wired
    // to content.js by default (see also the ?gl=0 startup query param).
    if (message?.lucid === 'glStretch') { gls.setEnabled(message.enabled !== false); paint(); return; }
    if (!message || message.lucid !== 'gap') return;
    const next = Math.max(0, Math.round(message.band || 0));
    if (next === gap) return;
    gap = next;
    // Re-composite immediately so the gap opens and closes with the controls
    // rather than waiting for whatever frame happens to arrive next.
    paint();
  });

  // If frames stop - the app quit, the video changed, the pipeline stalled -
  // clear the canvas so the real video shows through rather than a frozen
  // enhanced still. Doing it here rather than asking the page to hide us keeps
  // the whole decision on this side of the frame boundary.
  setInterval(() => {
    if (lastFrameAt && performance.now() - lastFrameAt > 400) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      gls.clear();
      lastNV12 = null;
      lastFrameAt = 0;
    }
  }, 200);

  // An open WebSocket makes a page ineligible for the back/forward cache, and
  // this frame is inside someone else's page - so let go of the socket while
  // the page is frozen rather than making their navigation slower.
  addEventListener('pagehide', (event) => {
    if (!event.persisted) return;
    frozen = true;
    captureReady();
    if (socket) { try { socket.close(); } catch (e) {} socket = null; }
  });
  addEventListener('pageshow', (event) => {
    if (!event.persisted) return;
    frozen = false;
    connect();
  });

  // ---- WebGL2 final-stretch ------------------------------------------------
  //
  // Two-pass separable Lanczos-3 resample, ported from the r89 prototype
  // (.build/quality-breakthrough-r89-realstretch/webgl_lanczos.html), run
  // against a live source canvas every frame instead of a static image.
  //
  // The target size is this document's own layout viewport (innerWidth x
  // innerHeight) times devicePixelRatio: content.js positions the surface
  // iframe with `surface.style.width/height = content.w/h + 'px'`
  // (content.js:329-330) and this document has no margin/border/scrollbar
  // (surface.html sets margin:0, overflow:hidden on html/body), so the
  // iframe's layout viewport is exactly that CSS box. That means the target
  // device-pixel size can be read entirely inside this iframe, with no new
  // message from content.js and no bridge/protocol change.
  function createGLStretch() {
    const glCanvas = document.getElementById('surfaceGL');
    const disabledByQuery = new URLSearchParams(location.search).get('gl') === '0';
    let userEnabled = !disabledByQuery;
    let gl = null, prog = null, uSrc, uSrcSize, uDstSize, uAxis;
    let srcTex = null, midTex = null, midFBO = null;
    let srcW = 0, srcH = 0; // last-allocated source-texture size
    let midTexW = 0, midTexH = 0; // last-allocated intermediate (pass-1 output) size
    let active = false; // whether the GL canvas is the one currently visible
    let compareHidden = false;
    // WebGL context loss (GPU process reset, driver crash, or Chromium's
    // per-tab/global context-count limit — plausible for an extension that
    // opens one WebGL2 context per active tab) makes every GL call a silent
    // no-op rather than throwing, so the try/catch in present() below can
    // never see it. `lost` plus the isContextLost() checks are what actually
    // detect it and drop to the 2D/CSS fallback instead of a blank canvas.
    let lost = false;

    if (!userEnabled) return { present() {}, clear() {}, setEnabled, setVisibility: () => {} };

    try { gl = init(); } catch (e) { gl = null; }
    if (!gl) return { present() {}, clear() {}, setEnabled, setVisibility: () => {} };

    // preventDefault() is required for the context to ever be eligible for
    // restoration at all; without it the browser treats the loss as
    // permanent. Fall back immediately rather than waiting for the next
    // present() call to notice via isContextLost(), so a stalled/frozen
    // video (no new frames driving present()) can't stay stuck on a blank
    // GL canvas after a loss.
    glCanvas.addEventListener('webglcontextlost', (event) => {
      event.preventDefault();
      lost = true;
      show(false);
    }, false);
    // Restoration hands back the *same* context object with all resources
    // (program, textures, framebuffer) gone, so re-run the same init() used
    // at startup to recreate them. Only clear `lost` — re-enabling the GL
    // path — if that fully succeeds; otherwise stay on the 2D fallback for
    // the rest of the session, same fail-safe policy present() already uses
    // for a run() exception.
    glCanvas.addEventListener('webglcontextrestored', () => {
      try {
        gl = init();
        if (!gl) throw new Error('reinit after context restore returned null');
        lost = false;
      } catch (e) {
        gl = null;
      }
    }, false);

    function init() {
      const g = glCanvas.getContext('webgl2', {
        alpha: true, antialias: false, depth: false, stencil: false,
        preserveDrawingBuffer: false, premultipliedAlpha: true,
      });
      if (!g) return null;
      const VS = `#version 300 es
        in vec2 aPos;
        out vec2 vUv;
        void main(){ vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }
      `;
      const FS = `#version 300 es
        precision highp float;
        in vec2 vUv;
        out vec4 outColor;
        uniform sampler2D uSrc;
        uniform vec2 uSrcSize;
        uniform vec2 uDstSize;
        uniform int uAxis;
        const float PI = 3.14159265358979;
        float sinc(float x) {
          if (abs(x) < 1e-6) return 1.0;
          float px = PI * x;
          return sin(px) / px;
        }
        float lanczos3(float x) {
          if (abs(x) >= 3.0) return 0.0;
          return sinc(x) * sinc(x / 3.0);
        }
        void main() {
          vec2 dstPixel = vUv * uDstSize;
          float scale = (uAxis == 0) ? (uSrcSize.x / uDstSize.x) : (uSrcSize.y / uDstSize.y);
          float dstCoord = (uAxis == 0) ? dstPixel.x : dstPixel.y;
          float srcCoord = (dstCoord + 0.5) * scale - 0.5;
          float base = floor(srcCoord);
          vec4 sum = vec4(0.0);
          float wsum = 0.0;
          for (int i = -2; i <= 3; i++) {
            float sampleCoord = base + float(i);
            float w = lanczos3(srcCoord - sampleCoord);
            if (w == 0.0) continue;
            vec2 uv;
            if (uAxis == 0) {
              float sx = clamp((sampleCoord + 0.5) / uSrcSize.x, 0.0, 1.0);
              uv = vec2(sx, vUv.y);
            } else {
              float sy = clamp((sampleCoord + 0.5) / uSrcSize.y, 0.0, 1.0);
              uv = vec2(vUv.x, sy);
            }
            sum += texture(uSrc, uv) * w;
            wsum += w;
          }
          outColor = sum / max(wsum, 1e-6);
        }
      `;
      function compile(type, src) {
        const s = g.createShader(type);
        g.shaderSource(s, src);
        g.compileShader(s);
        if (!g.getShaderParameter(s, g.COMPILE_STATUS)) { const info = g.getShaderInfoLog(s); g.deleteShader(s); throw new Error(info); }
        return s;
      }
      const vs = compile(g.VERTEX_SHADER, VS);
      const fs = compile(g.FRAGMENT_SHADER, FS);
      const p = g.createProgram();
      g.attachShader(p, vs); g.attachShader(p, fs); g.linkProgram(p);
      if (!g.getProgramParameter(p, g.LINK_STATUS)) { const info = g.getProgramInfoLog(p); g.deleteProgram(p); throw new Error(info); }
      g.deleteShader(vs); g.deleteShader(fs);
      prog = p;
      g.useProgram(prog);
      const quad = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
      const vbo = g.createBuffer();
      g.bindBuffer(g.ARRAY_BUFFER, vbo);
      g.bufferData(g.ARRAY_BUFFER, quad, g.STATIC_DRAW);
      const aPos = g.getAttribLocation(prog, 'aPos');
      g.enableVertexAttribArray(aPos);
      g.vertexAttribPointer(aPos, 2, g.FLOAT, false, 0, 0);
      uSrc = g.getUniformLocation(prog, 'uSrc');
      uSrcSize = g.getUniformLocation(prog, 'uSrcSize');
      uDstSize = g.getUniformLocation(prog, 'uDstSize');
      uAxis = g.getUniformLocation(prog, 'uAxis');
      srcTex = makeTexture(g);
      midTex = makeTexture(g);
      midFBO = g.createFramebuffer();
      // srcTex/midTex above are freshly created with no backing storage.
      // run() only calls texImage2D to allocate it when the requested size
      // differs from srcW/srcH/midTexW/midTexH, so those trackers must be
      // reset here too (not just at first-load, when they're already 0) or
      // a post-restore run() at an unchanged frame size would skip
      // allocating storage on the new, empty texture objects entirely.
      srcW = 0; srcH = 0; midTexW = 0; midTexH = 0;
      return g;
    }

    function makeTexture(g) {
      const tex = g.createTexture();
      g.bindTexture(g.TEXTURE_2D, tex);
      g.texParameteri(g.TEXTURE_2D, g.TEXTURE_MIN_FILTER, g.NEAREST);
      g.texParameteri(g.TEXTURE_2D, g.TEXTURE_MAG_FILTER, g.NEAREST);
      g.texParameteri(g.TEXTURE_2D, g.TEXTURE_WRAP_S, g.CLAMP_TO_EDGE);
      g.texParameteri(g.TEXTURE_2D, g.TEXTURE_WRAP_T, g.CLAMP_TO_EDGE);
      return tex;
    }

    function targetDevicePixels() {
      const dpr = self.devicePixelRatio || 1;
      return {
        w: Math.max(1, Math.round(innerWidth * dpr)),
        h: Math.max(1, Math.round(innerHeight * dpr)),
      };
    }

    function setEnabled(v) { userEnabled = v; if (!userEnabled) show(false); }
    function setVisibility(v) { compareHidden = v === 'hidden'; glCanvas.style.visibility = v; }

    function show(on) {
      active = on;
      glCanvas.style.display = on ? 'block' : 'none';
      canvas.style.display = on ? 'none' : 'block';
    }

    function clear() {
      if (!gl) return;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, glCanvas.width || 1, glCanvas.height || 1);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }

    // Only a pure magnification (target device-px size >= source size on
    // BOTH axes) is in scope: r89 measured and shipped a magnification
    // filter, and a non-widened Lanczos-3 kernel can alias on a downscaled
    // axis, which was never tested. Any other case (no stretch, or a
    // downscale on either axis) falls back to the untouched 2D/CSS path.
    function shouldStretch(sw, sh, tw, th) {
      return tw >= sw && th >= sh && (tw > sw || th > sh);
    }

    function present(srcCanvas, width, height) {
      // gl.isContextLost() is the belt to the `lost` flag's suspenders:
      // webglcontextlost fires asynchronously, so there's a real window
      // right after an actual loss where `lost` hasn't been set yet but the
      // context already reports lost and every GL call on it is a no-op.
      if (!gl || !userEnabled || lost || gl.isContextLost()) { if (active) show(false); return; }
      const target = targetDevicePixels();
      if (!shouldStretch(width, height, target.w, target.h)) { if (active) show(false); return; }
      try { run(srcCanvas, width, height, target.w, target.h); }
      catch (e) { show(false); gl = null; return; } // fail safe to the 2D path for the rest of the session
      show(true);
      glCanvas.style.visibility = compareHidden ? 'hidden' : 'visible';
    }

    function run(srcCanvas, sw, sh, tw, th) {
      if (sw !== srcW || sh !== srcH) {
        gl.bindTexture(gl.TEXTURE_2D, srcTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, sw, sh, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        srcW = sw; srcH = sh;
      }
      if (tw !== midTexW || sh !== midTexH) {
        gl.bindTexture(gl.TEXTURE_2D, midTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, tw, sh, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, midFBO);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, midTex, 0);
        midTexW = tw; midTexH = sh;
      }
      if (glCanvas.width !== tw || glCanvas.height !== th) {
        glCanvas.width = tw; glCanvas.height = th;
      }

      gl.useProgram(prog);
      gl.bindTexture(gl.TEXTURE_2D, srcTex);
      // Flip Y on upload: canvas-space row 0 is the top, WebGL texture row 0
      // is conventionally sampled as the bottom, and the GL default
      // framebuffer's row 0 likewise displays at the bottom of the canvas —
      // flipping once on the way in keeps both passes' pure resample math
      // (no per-pass flip) and lands right-side-up on screen.
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, sw, sh, gl.RGBA, gl.UNSIGNED_BYTE, srcCanvas);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

      // Pass 1: horizontal sw x sh -> tw x sh, into midTex.
      gl.bindFramebuffer(gl.FRAMEBUFFER, midFBO);
      gl.viewport(0, 0, tw, sh);
      gl.bindTexture(gl.TEXTURE_2D, srcTex);
      gl.uniform1i(uSrc, 0);
      gl.uniform2f(uSrcSize, sw, sh);
      gl.uniform2f(uDstSize, tw, sh);
      gl.uniform1i(uAxis, 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      // Pass 2: vertical tw x sh -> tw x th, into the default framebuffer.
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, tw, th);
      gl.bindTexture(gl.TEXTURE_2D, midTex);
      gl.uniform2f(uSrcSize, tw, sh);
      gl.uniform2f(uDstSize, tw, th);
      gl.uniform1i(uAxis, 1);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    return { present, clear, setEnabled, setVisibility };
  }

  connect();
})();
