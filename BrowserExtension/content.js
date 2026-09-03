// Lucid browser companion — content script.
//
// Reports the geometry and state of the most prominent <video> on the page to
// the native enhancer. It never draws anything and never touches the video.
// Runs as an extension content script (Chrome, Edge, Safari) or, when loaded
// directly by a page, talks to the native app over its loopback WebSocket.
(() => {
  if (window.top !== window) return;            // top frame only (v1)
  // One reporter per document: the extension's content script and a page that
  // embeds this script directly live in different JS worlds but share the DOM.
  const marker = 'data-video-enhancer-reporter';
  if (document.documentElement.hasAttribute(marker)) return;
  document.documentElement.setAttribute(marker, '1');

  const runtime = (globalThis.browser && browser.runtime && browser.runtime.id) ? browser.runtime
                : (globalThis.chrome && chrome.runtime && chrome.runtime.id) ? chrome.runtime : null;
  const BRIDGE_URL = 'ws://127.0.0.1:47811';
  const FRAME_MAGIC = 0x4c554346; // 'LUCF' decoded frame, page -> app
  const ENHANCED_MAGIC = 0x4c554345; // 'LUCE' enhanced frame, app -> page
  const HEARTBEAT_MS = 500;
  const CUTOUT_HOVER_MS = 150;
  const CUTOUT_IDLE_MS = 500;
  const HOVER_TIMEOUT_MS = 3000;

  const session = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2));
  const ua = navigator.userAgent;
  const browserName = /Edg\//.test(ua) ? 'edge' : /Chrome\//.test(ua) ? 'chrome' : /Safari\//.test(ua) ? 'safari' : 'unknown';

  // ---- transport -----------------------------------------------------------
  //
  // Most large sites set a Content-Security-Policy that forbids connecting to
  // ws://127.0.0.1, and that policy applies to this content script as well as
  // to the page. The extension's service worker is not bound by it, so when we
  // are running as an extension every byte - reports and video frames alike -
  // goes through the worker. Only a page that loads this file directly (the
  // test lab) opens its own socket.
  let send;                       // JSON messages, app-bound
  let sendBinary = () => false;   // video frames, app-bound
  let onBinary = null;            // set further down, once the decoder exists
  let transportReady = () => false;
  if (runtime) {
    let port = null;
    const connect = () => {
      try {
        port = runtime.connect({ name: 'lucid' });
        port.onMessage.addListener((message) => {
          if (!message) return;
          // Enhanced frames are megabytes and would have to be base64'd back
          // through the same JSON channel, which is far too expensive at frame
          // rate. On these sites the app presents through its own overlay
          // instead, so nothing binary is expected in this direction.
          if (message.type === 'nudge' && current) sendFrame(current);
        });
        port.onDisconnect.addListener(() => { port = null; setTimeout(connect, 1000); });
      } catch (e) { port = null; setTimeout(connect, 2000); }
    };
    connect();
    send = (message) => { if (port) { try { port.postMessage(message); } catch (e) { port = null; connect(); } } };
    // Runtime ports serialise as JSON, so an ArrayBuffer does not survive the
    // trip - it arrives as {}. Decoded frames are small (a 640x360 NV12 frame
    // is ~340 kB) so base64 is an acceptable price for being able to reach the
    // app at all on a site whose CSP blocks a direct socket.
    sendBinary = (buffer) => {
      if (!port) return false;
      try {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.length; i += 0x8000) {
          binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
        }
        port.postMessage({ t: 'b64', b: btoa(binary) });
        return true;
      } catch (e) { port = null; connect(); return false; }
    };
    transportReady = () => !!port;
  } else {
    let socket = null, backoff = 1000;
    const connect = () => {
      try {
        socket = new WebSocket(BRIDGE_URL);
        socket.onopen = () => { backoff = 1000; };
        socket.onclose = () => { socket = null; setTimeout(connect, backoff); backoff = Math.min(backoff * 2, 10000); };
        socket.onerror = () => {};
      } catch (e) { socket = null; setTimeout(connect, backoff); }
    };
    connect();
    send = (message) => { if (socket && socket.readyState === 1) socket.send(JSON.stringify(message)); };
  }

  // ---- drawing the enhanced frame back into the page -----------------------
  //
  // The result is drawn by the page, into a canvas that sits in the document
  // right where the video is. That way it inherits the page's scrolling,
  // clipping and stacking for free, which is the only way to make it behave
  // like part of the page rather than something pasted over the top. It never
  // takes pointer events, so the video's own controls keep working.
  let surface = null, surfaceCtx = null, surfaceFor = null, imageData = null;
  let drawing = false, lastDraw = 0, controlsUntil = 0;

  /// Paints the last decoded frame and, only while the browser's own controls
  /// are actually on screen, clears the strip they occupy so they show through.
  /// This is separate from receiving a frame: the gap has to close when the
  /// controls fade even if no new frame has arrived, or it stays open forever.
  let compositedWithGap = false;
  /// True when the enhanced frames are painted by the extension's own iframe
  /// rather than by a canvas in the page.
  const usesFrameSurface = () => !!(runtime && runtime.getURL);
  /// Tells the surface how tall the browser's control strip is right now, so it
  /// can clear that band. Sent only on change.
  let sentGap = -1;
  function updateSurfaceGap(video) {
    if (!surface || surface.tagName !== 'IFRAME' || !surface.contentWindow) return;
    let band = 0;
    if (video.controls && controlsShowing(video)) {
      const box = contentBox(video, viewportRect(video));
      band = Math.round(Math.min(Math.max(44, box.h * 0.12), 72) * (window.devicePixelRatio || 1));
    }
    if (band === sentGap) return;
    sentGap = band;
    try { surface.contentWindow.postMessage({ lucid: 'gap', band }, '*'); } catch (e) {}
  }
  function composite() {
    if (!surface || !surfaceCtx || !imageData || !current) return;
    surfaceCtx.putImageData(imageData, 0, 0);
    const gap = current.controls && controlsShowing(current);
    if (gap) {
      // `surface` is the overlay canvas. Sizing this from anything else clears
      // the wrong strip and the control bar stays covered.
      const box = contentBox(current, viewportRect(current));
      const scale = box.h > 0 ? surface.height / box.h : 1;
      // Chrome's control bar is a fixed height, not a fraction of the video.
      const band = Math.min(surface.height, Math.round(Math.min(Math.max(44, box.h * 0.12), 72) * scale));
      surfaceCtx.clearRect(0, surface.height - band, surface.width, band);
    }
    compositedWithGap = gap;
  }

  /// The video's own controls are drawn inside the video element, so a canvas
  /// over it hides them. Clear the strip they occupy while they are showing and
  /// the real video - controls and all - shows through there instead.
  function controlsShowing(video) {
    return video.paused || video.ended || performance.now() < controlsUntil;
  }

  function ensureSurface(video) {
    if (surface && surfaceFor === video && surface.isConnected) return surface;
    removeSurface();
    // As an extension we draw from an iframe at the extension's own origin.
    // A canvas in the page cannot receive the enhanced frames on any site whose
    // CSP forbids ws://127.0.0.1 - which is most large sites - because that
    // policy binds content scripts too. The iframe is still a normal element in
    // the page, so it scrolls, clips and stacks exactly as the canvas did.
    if (runtime && runtime.getURL) {
      const frame = document.createElement('iframe');
      frame.dataset.lucid = 'surface';
      frame.setAttribute('aria-hidden', 'true');
      frame.setAttribute('tabindex', '-1');
      frame.allowTransparency = 'true';
      frame.style.cssText = 'position:absolute; pointer-events:none; margin:0; padding:0; border:0; background:transparent; colorScheme:normal;';
      frame.src = runtime.getURL('surface.html') + '#' + session;
      const host = video.parentElement || document.body;
      if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
      video.insertAdjacentElement('afterend', frame);
      surface = frame;
      surfaceCtx = null;
      surfaceFor = video;
      return frame;
    }
    const canvas = document.createElement('canvas');
    canvas.dataset.lucid = 'surface';
    canvas.style.cssText = 'position:absolute; pointer-events:none; margin:0; padding:0; border:0;';
    const host = video.parentElement || document.body;
    if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
    // Insert directly after the video, so anything the player draws over the
    // video - its control bar, captions, overlays - is later in the document
    // and paints on top of us without being told about. Real players build
    // their controls from ordinary elements, so this is the whole story for
    // them; only the browser's own built-in controls live inside the video
    // element where nothing can be placed above them.
    video.insertAdjacentElement('afterend', canvas);
    surface = canvas;
    surfaceCtx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    surfaceFor = video;
    return canvas;
  }

  function removeSurface() {
    if (surface && surface.parentElement) surface.parentElement.removeChild(surface);
    surface = null; surfaceCtx = null; surfaceFor = null; imageData = null; drawing = false;
  }

  // Keep the canvas exactly over the video's rendered content box, in the
  // video's own coordinate space, so scrolling moves both together.
  function positionSurface(video) {
    if (!surface) return;
    const host = surface.parentElement;
    const hostRect = host.getBoundingClientRect();
    const box = viewportRect(video);
    const content = contentBox(video, box);
    const style = getComputedStyle(video);
    surface.style.left = (content.x - hostRect.left + host.scrollLeft) + 'px';
    surface.style.top = (content.y - hostRect.top + host.scrollTop) + 'px';
    surface.style.width = content.w + 'px';
    surface.style.height = content.h + 'px';
    surface.style.zIndex = style.zIndex === 'auto' ? 'auto' : style.zIndex;
    surface.style.borderRadius = style.borderRadius;
    const isFrame = surface.tagName === 'IFRAME';
    surface.style.display = ((drawing || isFrame) && content.w > 1) ? 'block' : 'none';
    // The iframe decides for itself whether it has anything to show: it clears
    // its own canvas when frames stop, so it is transparent rather than stale.
    if (isFrame) surface.style.opacity = '1';
  }

  function drawEnhanced(width, height, pixels) {
    if (!current) return;
    // Under the extension the surface iframe receives frames directly and
    // paints them itself; nothing arrives here.
    if (runtime && runtime.getURL) return;
    const canvas = ensureSurface(current);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width; canvas.height = height; imageData = null;
    }
    if (!imageData || imageData.width !== width || imageData.height !== height) {
      imageData = new ImageData(new Uint8ClampedArray(width * height * 4), width, height);
    }
    imageData.data.set(pixels);
    composite();
    drawing = true; lastDraw = performance.now();
    positionSurface(current);
  }

  // ---- video discovery -----------------------------------------------------
  function collectVideos(root, out, depth) {
    if (depth > 4) return;
    let nodes;
    try { nodes = root.querySelectorAll('video, iframe, *'); } catch (e) { return; }
    for (const node of nodes) {
      if (node.tagName === 'VIDEO') out.push(node);
      else if (node.tagName === 'IFRAME') {
        try { if (node.contentDocument) collectVideos(node.contentDocument, out, depth + 1); } catch (e) {}
      } else if (node.shadowRoot) collectVideos(node.shadowRoot, out, depth + 1);
    }
  }

  function viewportRect(el) {
    // Bounding rect in the top viewport, walking up through same-origin iframes.
    let rect = el.getBoundingClientRect();
    let x = rect.left, y = rect.top, w = rect.width, h = rect.height;
    let win = el.ownerDocument.defaultView;
    while (win && win !== window && win.frameElement) {
      const fr = win.frameElement.getBoundingClientRect();
      x += fr.left + win.frameElement.clientLeft; y += fr.top + win.frameElement.clientTop;
      win = win.parent;
    }
    return { x, y, w, h };
  }

  function visibleArea(r) {
    const w = Math.max(0, Math.min(r.x + r.w, innerWidth) - Math.max(r.x, 0));
    const h = Math.max(0, Math.min(r.y + r.h, innerHeight) - Math.max(r.y, 0));
    return w * h;
  }

  function contentBox(video, box) {
    // Rendered content box after object-fit (video default is 'contain').
    const iw = video.videoWidth, ih = video.videoHeight;
    if (!iw || !ih || !box.w || !box.h) return box;
    const fit = getComputedStyle(video).objectFit || 'contain';
    let w = box.w, h = box.h;
    if (fit === 'contain' || fit === 'scale-down') { const s = Math.min(box.w / iw, box.h / ih); w = iw * s; h = ih * s; }
    else if (fit === 'cover') { const s = Math.max(box.w / iw, box.h / ih); w = iw * s; h = ih * s; }
    else if (fit === 'none') { w = iw; h = ih; }
    const x = box.x + (box.w - w) / 2, y = box.y + (box.h - h) / 2;
    // Clip 'cover'/'none' overflow to the element box.
    const cx = Math.max(x, box.x), cy = Math.max(y, box.y);
    return { x: cx, y: cy, w: Math.min(x + w, box.x + box.w) - cx, h: Math.min(y + h, box.y + box.h) - cy };
  }

  function pickVideo() {
    const videos = [];
    collectVideos(document, videos, 0);
    let best = null, bestScore = 0;
    for (const v of videos) {
      if (!v.videoWidth || !v.videoHeight) continue;
      const cs = getComputedStyle(v);
      if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity) < 0.5) continue;
      const r = viewportRect(v);
      let score = visibleArea(r);
      if (v.paused) score *= 0.25;
      if (score > bestScore) { best = v; bestScore = score; }
    }
    return bestScore > 0 ? best : null;
  }

  // ---- cutouts (page elements drawn over the video) ------------------------
  function isPainted(el) {
    const cs = getComputedStyle(el);
    if (cs.visibility !== 'visible' || parseFloat(cs.opacity) < 0.05 || cs.display === 'none') return false;
    if (cs.pointerEvents === 'none' && !el.textContent.trim()) return false;
    const bg = cs.backgroundColor;
    const hasBg = bg && !/rgba\(\s*\d+,\s*\d+,\s*\d+,\s*0\)/.test(bg) && bg !== 'transparent';
    const tag = el.tagName;
    return hasBg || cs.backgroundImage !== 'none' || /^(IMG|SVG|CANVAS|BUTTON|INPUT|SELECT|TEXTAREA|PROGRESS)$/.test(tag)
      || (el.childElementCount === 0 && el.textContent.trim().length > 0);
  }

  function computeCutouts(video, box) {
    const cols = 12, rows = 7, found = new Map();
    const round = (n) => Math.round(n * 10) / 10;
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      const px = box.x + (i + 0.5) * box.w / cols, py = box.y + (j + 0.5) * box.h / rows;
      if (px < 0 || py < 0 || px >= innerWidth || py >= innerHeight) continue;
      let stack;
      try { stack = document.elementsFromPoint(px, py); } catch (e) { continue; }
      for (const el of stack) {
        if (el === video) break;                       // everything after is below the video
        if (el.contains(video)) continue;              // containers / wrappers
        if (found.has(el)) continue;
        if (el.tagName === 'HTML' || el.tagName === 'BODY') continue;
        if (!isPainted(el)) continue;
        const r = el.getBoundingClientRect();
        if (r.width * r.height >= box.w * box.h * 0.9) continue;   // full-size overlays are not controls
        if (r.width < 4 || r.height < 4) continue;
        found.set(el, { x: round(r.left), y: round(r.top), w: round(r.width), h: round(r.height) });
      }
    }
    // Merge overlapping rects, cap the count.
    const rects = [...found.values()];
    let merged = true;
    while (merged && rects.length > 1) {
      merged = false;
      outer: for (let a = 0; a < rects.length; a++) for (let b = a + 1; b < rects.length; b++) {
        const A = rects[a], B = rects[b];
        if (A.x < B.x + B.w && B.x < A.x + A.w && A.y < B.y + B.h && B.y < A.y + A.h) {
          const x = Math.min(A.x, B.x), y = Math.min(A.y, B.y);
          rects[a] = { x, y, w: Math.max(A.x + A.w, B.x + B.w) - x, h: Math.max(A.y + A.h, B.y + B.h) - y };
          rects.splice(b, 1); merged = true; break outer;
        }
      }
    }
    return rects.slice(0, 16);
  }

  // ---- decoded frame streaming ----------------------------------------------
  //
  // The whole point: hand the app the frame as the decoder produced it, at its
  // own resolution, before the page stretches it to fit the player. Reading it
  // back off the screen would mean enhancing something the browser has already
  // upscaled and resampled, which throws away most of what there is to work
  // with.
  let frameSocket = null, frameBackoff = 500, framesInFlight = 0, frameSeq = 0;
  const stats = { sent: 0, errors: 0, last: '', socket: 'none', streaming: false, fps: 0, cb: 0, copyMs: 0, buffered: 0 };
  let fpsWindow = [], cbWindow = [];
  const publishStats = () => {
    document.documentElement.dataset.lucidFrames =
      `${stats.socket} sent=${stats.sent} fps=${stats.fps} rx=${stats.rx||0}/${stats.rxType||'-'} drawn=${stats.drawn||0} cb=${stats.cb} inflight=${framesInFlight} copy=${stats.copyMs}ms buf=${Math.round(stats.buffered/1024)}kB err=${stats.errors} ${stats.last}`;
  };
  let canvas = null, canvasCtx = null;
  let streaming = false, rvfcHandle = 0;

  /// Handles an enhanced frame arriving from the app, whichever transport
  /// carried it.
  function receiveBinary(buffer) {
    stats.rx = (stats.rx || 0) + 1;
    stats.rxType = 'binary' + buffer.byteLength;
    const view = new DataView(buffer);
    if (view.byteLength < 8 || view.getUint32(0, false) !== ENHANCED_MAGIC) return;
    const headerLength = view.getUint32(4, false);
    try {
      const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength)));
      if (meta.session !== session) return;
      drawEnhanced(meta.w, meta.h, new Uint8Array(buffer, 8 + headerLength));
      stats.drawn = (stats.drawn || 0) + 1;
      if (stats.drawn % 30 === 1) publishStats();
    } catch (e) { stats.errors++; stats.last = 'draw ' + (e && e.message || e); publishStats(); }
  }
  onBinary = receiveBinary;

  function openFrameSocket() {
    // Under the extension the worker already carries frames; nothing to open.
    if (runtime) { stats.socket = 'port'; publishStats(); return; }
    if (frameSocket && (frameSocket.readyState === 0 || frameSocket.readyState === 1)) return;
    try { frameSocket = new WebSocket(BRIDGE_URL); } catch (e) { frameSocket = null; return; }
    frameSocket.binaryType = 'arraybuffer';
    frameSocket.onopen = () => { frameBackoff = 500; stats.socket = 'open'; publishStats(); };
    frameSocket.onclose = () => {
      frameSocket = null; framesInFlight = 0; stats.socket = 'closed'; publishStats();
      setTimeout(openFrameSocket, frameBackoff);
      frameBackoff = Math.min(frameBackoff * 2, 8000);
    };
    frameSocket.onerror = () => {};
    // The app asks for a frame when it needs one re-rendered, which is how a
    // paused video keeps up with a settings change.
    frameSocket.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) { receiveBinary(event.data); return; }
      stats.rx = (stats.rx || 0) + 1;
      stats.rxType = 'text';
      // The app asks for a frame when it needs one re-rendered, which is how a
      // paused video keeps up with a settings change.
      try {
        const message = JSON.parse(event.data);
        if (message && message.type === 'nudge' && current) sendFrame(current);
      } catch (e) {}
    };
  }

  function deliverFrame(packet) {
    if (runtime) { sendBinary(packet); return; }
    if (frameSocket && frameSocket.readyState === 1) frameSocket.send(packet);
  }

  function header(width, height, format, planes, timestamp) {
    const meta = JSON.stringify({ session, w: width, h: height, format, planes, seq: ++frameSeq, ts: timestamp });
    const metaBytes = new TextEncoder().encode(meta);
    const head = new ArrayBuffer(8 + metaBytes.length);
    const view = new DataView(head);
    view.setUint32(0, FRAME_MAGIC, false);
    view.setUint32(4, metaBytes.length, false);
    new Uint8Array(head, 8).set(metaBytes);
    return head;
  }

  async function sendFrame(video) {
    if (runtime ? !transportReady() : (!frameSocket || frameSocket.readyState !== 1)) return;
    // Allow a couple of copies in flight: the copy resolves on a microtask, so
    // a limit of one refuses the very next frame callback and halves the rate.
    // The socket buffer is what actually bounds latency.
    stats.buffered = frameSocket ? frameSocket.bufferedAmount : 0;
    if (framesInFlight > 2 || (frameSocket && frameSocket.bufferedAmount > 6 << 20)) return;
    const width = video.videoWidth, height = video.videoHeight;
    if (!width || !height) return;
    framesInFlight++;
    try {
      if (typeof VideoFrame === 'function') {
        let frame = null;
        try { frame = new VideoFrame(video); } catch (e) { frame = null; }
        if (frame) {
          try {
            const size = frame.allocationSize();
            const buffer = new ArrayBuffer(size);
            const t0 = performance.now();
            const layout = await frame.copyTo(buffer);
            stats.copyMs = (stats.copyMs * 0.8 + (performance.now() - t0) * 0.2).toFixed(1);
            const planes = layout.map(p => ({ offset: p.offset, stride: p.stride }));
            const head = header(frame.codedWidth, frame.codedHeight, frame.format, planes, frame.timestamp);
            const packet = new Uint8Array(head.byteLength + size);
            packet.set(new Uint8Array(head), 0);
            packet.set(new Uint8Array(buffer), head.byteLength);
            deliverFrame(packet);
            stats.sent++; stats.last = `${frame.format} ${frame.codedWidth}x${frame.codedHeight}`;
            const now = performance.now();
            fpsWindow.push(now); while (fpsWindow.length && now - fpsWindow[0] > 1000) fpsWindow.shift();
            stats.fps = fpsWindow.length;
            if (stats.sent % 15 === 1) publishStats();
            return;
          } finally { frame.close(); }
        }
      }
      // Fallback: read the decoded pixels through a canvas at native size.
      if (!canvas || canvas.width !== width || canvas.height !== height) {
        canvas = document.createElement('canvas');
        canvas.width = width; canvas.height = height;
        canvasCtx = canvas.getContext('2d', { willReadFrequently: true, alpha: false });
      }
      canvasCtx.drawImage(video, 0, 0, width, height);
      const data = canvasCtx.getImageData(0, 0, width, height).data;
      const head = header(width, height, 'RGBA', [{ offset: 0, stride: width * 4 }], performance.now() * 1000);
      const packet = new Uint8Array(head.byteLength + data.byteLength);
      packet.set(new Uint8Array(head), 0);
      packet.set(data, head.byteLength);
      deliverFrame(packet);
      stats.sent++; stats.last = `RGBA ${width}x${height}`;
      if (stats.sent % 30 === 1) publishStats();
    } catch (e) {
      stats.errors++; stats.last = 'ERR ' + (e && e.message || e); publishStats();
      // Cross-origin video without CORS cannot be read; let the app fall back
      // to screen capture.
      if (stats.errors > 5) streaming = false;
    } finally {
      framesInFlight--;
    }
  }

  function pump(video) {
    if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype)) return;
    const step = () => {
      if (!streaming || video !== current) return;
      const now = performance.now();
      cbWindow.push(now); while (cbWindow.length && now - cbWindow[0] > 1000) cbWindow.shift();
      stats.cb = cbWindow.length;
      sendFrame(video);
      rvfcHandle = video.requestVideoFrameCallback(step);
    };
    rvfcHandle = video.requestVideoFrameCallback(step);
  }

  // A paused video presents no new frames, so the frame callback stops firing.
  // Keep feeding the current one slowly: a paused video should stay enhanced,
  // and it lets the enhancement be switched while the picture is held still.
  let idleTimer = 0;
  function pumpIdle(video) {
    clearInterval(idleTimer);
    idleTimer = setInterval(() => {
      if (!current || document.visibilityState !== 'visible') return;
      if (!current.paused && !current.ended) return;   // the frame callback has it
      if (current.readyState < 2) return;
      if (!streaming) { streaming = true; stats.streaming = true; openFrameSocket(); }
      sendFrame(current);
    }, 250);
  }

  function startStreaming(video) {
    if (streaming) return;
    streaming = true; stats.streaming = true; publishStats();
    openFrameSocket();
    pump(video);
  }

  function stopStreaming() { streaming = false; stats.streaming = false; publishStats(); }

  // ---- reporting loop -------------------------------------------------------
  let current = null, lastKey = '', lastSent = 0, lastCutoutAt = 0, cutouts = [];
  let hoverUntil = 0, movingFrames = 0, lastRectKey = '', absentSince = 0;

  function snapshot(video) {
    const box = viewportRect(video);
    const rect = contentBox(video, box);
    const fs = document.fullscreenElement || document.webkitFullscreenElement;
    const radius = parseFloat(getComputedStyle(video).borderTopLeftRadius) || 0;
    return {
      rect: { x: +rect.x.toFixed(2), y: +rect.y.toFixed(2), w: +rect.w.toFixed(2), h: +rect.h.toFixed(2) },
      iw: video.videoWidth, ih: video.videoHeight,
      paused: video.paused, ended: video.ended,
      fullscreen: !!fs, pip: document.pictureInPictureElement === video || (video.webkitPresentationMode === 'picture-in-picture'),
      radius: +radius.toFixed(2)
    };
  }

  function tick() {
    const now = performance.now();
    // A page can look momentarily hidden or video-less while the layout
    // settles. Withdrawing instantly tears the app's session down and makes it
    // rebuild the whole pipeline, so require the condition to actually hold.
    const absent = document.visibilityState !== 'visible' || !pickVideo();
    if (absent) {
      if (!absentSince) absentSince = now;
      if (now - absentSince > 500) sendGone();
      requestAnimationFrame(tick); return;
    }
    absentSince = 0;
    const video = pickVideo();
    if (video !== current) { current = video; cutouts = []; lastCutoutAt = 0; stopStreaming(); removeSurface(); }

    const v = snapshot(video);
    const rectKey = `${v.rect.x},${v.rect.y},${v.rect.w},${v.rect.h}`;
    if (rectKey !== lastRectKey) { movingFrames = 3; lastRectKey = rectKey; } else if (movingFrames > 0) movingFrames--;
    const moving = movingFrames > 0;
    const hover = now < hoverUntil;

    const cutoutInterval = hover ? CUTOUT_HOVER_MS : CUTOUT_IDLE_MS;
    if (!moving && now - lastCutoutAt > cutoutInterval) { cutouts = computeCutouts(video, v.rect); lastCutoutAt = now; }

    if (!v.paused && !v.ended) startStreaming(video);
    // The canvas has to follow the video every frame, not on a timer, or it
    // slides behind during a scroll.
    // The surface has to exist before the app has anywhere to send frames, so
    // put it in place as soon as there is a video worth enhancing. Under the
    // extension it is an iframe that fetches its own frames and paints itself;
    // all we do from here is keep it positioned and tell it about the controls.
    // Not gated on playback: a paused video still gets re-rendered when a
    // setting changes, and the app needs somewhere to put that frame.
    if (usesFrameSurface()) {
      ensureSurface(video);
      updateSurfaceGap(video);
      // Report that the page is drawing as soon as the surface exists, not once
      // frames arrive. The app will not send a frame until the page claims to
      // draw, so waiting for one first would deadlock. Whether anything is
      // actually on screen is a separate question, handled by the opacity.
      drawing = !!surface;
      positionSurface(video);
    } else if (drawing) {
      // Canvas path only. If frames stop - the app quit, or the bridge fell
      // behind - take the canvas away rather than leave a stale image, and
      // especially rather than leave a punched control gap, over playing video.
      if (performance.now() - lastDraw > 400) {
        drawing = false;
        removeSurface();
      } else {
        // Close the control gap as soon as the controls fade, without waiting
        // for another frame that may never come.
        if ((video.controls && controlsShowing(video)) !== compositedWithGap) composite();
        positionSurface(video);
      }
    }

    const message = {
      type: 'video', browser: browserName, session, title: document.title, url: location.href.slice(0, 512),
      frames: streaming,
      draws: drawing,
      visible: true,
      screenX: window.screenX, screenY: window.screenY, outerWidth: window.outerWidth, outerHeight: window.outerHeight,
      innerWidth: window.innerWidth, innerHeight: window.innerHeight, dpr: window.devicePixelRatio,
      video: v, moving, hover, cutouts, ts: now
    };
    const key = JSON.stringify([message.title, message.screenX, message.screenY, message.outerWidth, message.outerHeight,
      message.innerWidth, message.innerHeight, message.dpr, v, moving, hover, cutouts]);
    if (key !== lastKey || now - lastSent > HEARTBEAT_MS) { send(message); lastKey = key; lastSent = now; }
    requestAnimationFrame(tick);
  }

  function sendGone() {
    stopStreaming();
    removeSurface();
    if (lastKey === 'gone') return;
    send({ type: 'gone', browser: browserName, session, title: document.title, visible: false,
      screenX: 0, screenY: 0, outerWidth: 0, outerHeight: 0, innerWidth: 0, innerHeight: 0, dpr: 1,
      video: null, moving: false, hover: false, cutouts: [], ts: performance.now() });
    lastKey = 'gone'; current = null;
  }

  // A looping video fires ended/seeked/play as it wraps; each one must put the
  // frame pump back to work, or the overlay is left holding the last frame of
  // the previous pass while the video plays on underneath.
  for (const type of ['play', 'playing', 'seeked', 'ended', 'loadeddata', 'timeupdate']) {
    document.addEventListener(type, (event) => {
      const video = event.target;
      if (!(video instanceof HTMLVideoElement)) return;
      if (video !== current) return;
      if (!video.paused && !video.ended) startStreaming(video);
      else if (streaming && video.readyState >= 2) sendFrame(video);
    }, true);
  }

  document.addEventListener('mousemove', (e) => {
    if (!current) return;
    const r = viewportRect(current);
    if (e.clientX >= r.x && e.clientX <= r.x + r.w && e.clientY >= r.y && e.clientY <= r.y + r.h) {
      hoverUntil = performance.now() + HOVER_TIMEOUT_MS;
      // Chrome keeps its controls up for about three seconds after the pointer
      // moves; match that so the hole appears and disappears with them.
      controlsUntil = performance.now() + 3200;
      if (streaming && current.readyState >= 2) sendFrame(current);
    }
  }, { passive: true, capture: true });
  document.addEventListener('mouseleave', () => { hoverUntil = 0; }, { capture: true });
  addEventListener('scroll', () => { movingFrames = 3; }, { passive: true, capture: true });
  addEventListener('resize', () => { movingFrames = 3; });
  addEventListener('pagehide', sendGone);
  document.addEventListener('visibilitychange', () => { if (document.visibilityState !== 'visible') sendGone(); });

  pumpIdle();
  requestAnimationFrame(tick);
})();
