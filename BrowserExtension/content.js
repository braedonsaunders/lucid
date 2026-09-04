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
  const TOKEN_URL = 'http://127.0.0.1:47812/token';
  async function fetchBridgeToken() {
    const response = await fetch(TOKEN_URL, { cache: 'no-store' });
    if (!response.ok) throw new Error('token ' + response.status);
    const token = (await response.text()).trim();
    if (!token) throw new Error('token empty');
    return token;
  }
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
  // True while the document is in the back/forward cache. A frozen page must
  // not hold an extension port open or try to reopen one.
  let frozen = false;
  let send;                       // JSON messages, app-bound
  let sendBinary = () => false;   // video frames, app-bound
  let releasePort = () => {};     // let go of the port before the page freezes
  let reconnectPort = () => {};   // and take it again when the page comes back
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
        port.onDisconnect.addListener(() => {
          // Reading lastError acknowledges it. Chrome severs extension ports
          // when a page enters the back/forward cache, and without this the
          // browser logs "Unchecked runtime.lastError" every time that happens.
          void (runtime.lastError && runtime.lastError.message);
          port = null;
          // Do not reconnect from a frozen page; pageshow does that instead.
          if (!frozen) setTimeout(connect, 1000);
        });
      } catch (e) { port = null; setTimeout(connect, 2000); }
    };
    connect();
    send = (message) => { if (port) { try { port.postMessage(message); } catch (e) { port = null; connect(); } } };
    // Runtime ports serialise as JSON, so an ArrayBuffer does not survive the
    // trip - it arrives as {}. Decoded frames are small (a 640x360 NV12 frame
    // is ~340 kB) so base64 is an acceptable price for being able to reach the
    // app at all on a site whose CSP blocks a direct socket.
    releasePort = () => { try { if (port) port.disconnect(); } catch (e) {} port = null; };
    reconnectPort = () => { if (!port) connect(); };
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
    let socket = null, backoff = 1000, connecting = false, reportReady = false;
    const connect = async () => {
      if (connecting || (socket && (socket.readyState === 0 || socket.readyState === 1))) return;
      connecting = true;
      reportReady = false;
      let token;
      try {
        token = await fetchBridgeToken();
      } catch (e) {
        connecting = false;
        socket = null;
        setTimeout(connect, backoff);
        backoff = Math.min(backoff * 2, 10000);
        return;
      }
      try {
        socket = new WebSocket(BRIDGE_URL);
        socket.onopen = () => {
          connecting = false;
          backoff = 1000;
          try { socket.send(JSON.stringify({ type: 'hello', token })); } catch (e) {}
          reportReady = true;
        };
        socket.onclose = () => {
          connecting = false;
          reportReady = false;
          socket = null;
          setTimeout(connect, backoff);
          backoff = Math.min(backoff * 2, 10000);
        };
        socket.onerror = () => {};
      } catch (e) {
        connecting = false;
        socket = null;
        setTimeout(connect, backoff);
      }
    };
    connect();
    send = (message) => { if (socket && socket.readyState === 1 && reportReady) socket.send(JSON.stringify(message)); };
    // This socket carries reports. Frames arrive on frameSocket, opened further
    // down, so that is the one that decides whether the page can draw.
    transportReady = () => !!(frameSocket && frameSocket.readyState === 1);
  }

  // ---- drawing the enhanced frame back into the page -----------------------
  //
  // The result is drawn by the page, into a canvas that sits in the document
  // right where the video is. That way it inherits the page's scrolling,
  // clipping and stacking for free, which is the only way to make it behave
  // like part of the page rather than something pasted over the top. It never
  // takes pointer events, so the video's own controls keep working.
  let surface = null, surfaceCtx = null, surfaceFor = null, lastNV12 = null;
  let drawing = false, lastDraw = 0, controlsUntil = 0;
  // Whether a frame has ever been painted, so the stale-frame teardown below
  // cannot fire before the first one arrives.
  let everDrew = false;

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
  function paintNV12(ctx, width, height, pixels) {
    const frame = new VideoFrame(pixels, {
      format: 'NV12',
      codedWidth: width,
      codedHeight: height,
      timestamp: 0,
      layout: [
        { offset: 0, stride: width },
        { offset: width * height, stride: width },
      ],
    });
    ctx.drawImage(frame, 0, 0, width, height);
    frame.close();
  }

  function composite() {
    if (!surface || !surfaceCtx || !lastNV12 || !current) return;
    paintNV12(surfaceCtx, lastNV12.width, lastNV12.height, lastNV12.pixels);
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
    surface = null; surfaceCtx = null; surfaceFor = null; lastNV12 = null; drawing = false; everDrew = false;
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

  function drawEnhanced(width, height, pixels, format) {
    if (!current) return;
    // Under the extension the surface iframe receives frames directly and
    // paints them itself; nothing arrives here.
    if (runtime && runtime.getURL) return;
    if (format && format !== 'NV12') return;
    const canvas = ensureSurface(current);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width; canvas.height = height;
    }
    lastNV12 = { width, height, pixels };
    composite();
    drawing = true; everDrew = true; lastDraw = performance.now();
    positionSurface(current);
  }

  // ---- video discovery -----------------------------------------------------
  //
  // Do not walk the whole DOM every animation frame. The old collector ran
  // querySelectorAll('video, iframe, *') — every element — twice per frame,
  // ~120 full walks a second in the page's renderer. Keep a live set and
  // only score that.
  const knownVideos = new Set();
  const observedRoots = new Set();

  function watchRoot(root) {
    if (!root || observedRoots.has(root)) return;
    observedRoots.add(root);
    const observer = new MutationObserver((records) => {
      for (const record of records) {
        for (const node of record.addedNodes) ingestNode(node);
        for (const node of record.removedNodes) forgetNode(node);
      }
    });
    try {
      observer.observe(root, { childList: true, subtree: true });
    } catch (e) {
      observedRoots.delete(root);
      return;
    }
    ingestNode(root === document ? document.documentElement : root);
  }

  function bindIframe(frame) {
    const bind = () => {
      try { if (frame.contentDocument) watchRoot(frame.contentDocument); } catch (e) {}
    };
    bind();
    frame.addEventListener('load', bind);
  }

  function ingestNode(node) {
    if (!node) return;
    if (node.nodeType === 11) { watchRoot(node); return; }
    if (node.nodeType !== 1) return;
    if (node.tagName === 'VIDEO') knownVideos.add(node);
    else if (node.tagName === 'IFRAME') bindIframe(node);
    if (node.shadowRoot) watchRoot(node.shadowRoot);
    if (!node.querySelectorAll) return;
    let found;
    try { found = node.querySelectorAll('video, iframe'); } catch (e) { return; }
    for (const el of found) {
      if (el.tagName === 'VIDEO') knownVideos.add(el);
      else if (el.tagName === 'IFRAME') bindIframe(el);
    }
    // Open shadow roots have no selector. Walk this subtree once, when it
    // appears, not every frame.
    let all;
    try { all = node.querySelectorAll('*'); } catch (e) { return; }
    for (const el of all) {
      if (el.shadowRoot) watchRoot(el.shadowRoot);
    }
  }

  function forgetNode(node) {
    if (!node || node.nodeType !== 1) return;
    if (node.tagName === 'VIDEO') knownVideos.delete(node);
    if (!node.querySelectorAll) return;
    try {
      for (const video of node.querySelectorAll('video')) knownVideos.delete(video);
    } catch (e) {}
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
    let best = null, bestScore = 0;
    for (const v of knownVideos) {
      if (!v.isConnected) { knownVideos.delete(v); continue; }
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
      drawEnhanced(meta.w, meta.h, new Uint8Array(buffer, 8 + headerLength), meta.format);
      stats.drawn = (stats.drawn || 0) + 1;
      if (stats.drawn % 30 === 1) publishStats();
    } catch (e) { stats.errors++; stats.last = 'draw ' + (e && e.message || e); publishStats(); }
  }
  onBinary = receiveBinary;

  let frameConnecting = false;
  function openFrameSocket() {
    // Under the extension the worker already carries frames; nothing to open.
    if (runtime) { stats.socket = 'port'; publishStats(); return; }
    if (frameConnecting || (frameSocket && (frameSocket.readyState === 0 || frameSocket.readyState === 1))) return;
    frameConnecting = true;
    fetchBridgeToken().then((token) => {
      try { frameSocket = new WebSocket(BRIDGE_URL); } catch (e) { frameConnecting = false; frameSocket = null; return; }
      frameSocket.binaryType = 'arraybuffer';
      frameSocket.onopen = () => {
        frameConnecting = false;
        frameBackoff = 500; stats.socket = 'open';
        try { frameSocket.send(JSON.stringify({ type: 'hello', token })); } catch (e) {}
        // Say which video's frames belong to this socket. Reports travel on a
        // different connection, and the app learns the session from those, so
        // without this it sends every frame to the report socket - which has no
        // onmessage handler and never reads a byte of it. The surface iframe
        // sends the same message for the same reason.
        try { frameSocket.send(JSON.stringify({ type: 'attach', session })); } catch (e) {}
        publishStats();
      };
      frameSocket.onclose = () => {
        frameConnecting = false;
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
        try {
          const message = JSON.parse(event.data);
          if (message && message.type === 'nudge' && current) sendFrame(current);
        } catch (e) {}
      };
    }).catch(() => {
      frameConnecting = false;
      setTimeout(openFrameSocket, frameBackoff);
      frameBackoff = Math.min(frameBackoff * 2, 8000);
    });
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
    const video = document.visibilityState === 'visible' ? pickVideo() : null;
    if (!video) {
      if (!absentSince) absentSince = now;
      if (now - absentSince > 500) sendGone();
      requestAnimationFrame(tick); return;
    }
    absentSince = 0;
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
    } else if (!drawing) {
      // Canvas path: a page that loads this script directly, like the lab.
      // The canvas is only created when the first frame arrives, so the page
      // cannot wait for a frame before claiming to draw - the app will not
      // send one until it does. That is exactly the deadlock the iframe path
      // above avoids, and it left the lab showing an untouched video while
      // every number in the app read healthy. Claim as soon as frames have
      // somewhere to arrive.
      drawing = transportReady();
      if (drawing) lastDraw = performance.now();
    } else {
      // If frames stop - the app quit, or the bridge fell behind - take the
      // canvas away rather than leave a stale image, and especially rather
      // than leave a punched control gap, over playing video. Only once
      // something has actually been drawn: tearing down before the first
      // frame would just re-enter the branch above and oscillate.
      if (everDrew && performance.now() - lastDraw > 400) {
        drawing = false; everDrew = false;
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
      knownVideos.add(video);
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
  // A page put into the back/forward cache keeps running its listeners but has
  // its extension ports cut. Let go of ours deliberately, and pick it back up
  // if the page is restored, rather than being severed and logging an error.
  addEventListener('pagehide', (event) => {
    sendGone();
    if (event.persisted) { frozen = true; releasePort(); }
  });
  addEventListener('pageshow', (event) => {
    if (!event.persisted) return;
    frozen = false;
    reconnectPort();
  });
  document.addEventListener('visibilitychange', () => { if (document.visibilityState !== 'visible') sendGone(); });

  watchRoot(document);
  pumpIdle();
  requestAnimationFrame(tick);
})();
