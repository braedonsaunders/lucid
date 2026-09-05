// Lucid browser companion — drawing surface.
//
// Holds its own socket to the native app and paints the enhanced frames it is
// sent. Lives at the extension's origin so the page's CSP cannot stop it
// connecting; see surface.html for why that is the whole point of this file.
(() => {
  const BRIDGE_URL = 'ws://127.0.0.1:48111';
  const ENHANCED_MAGIC = 0x4c554345; // 'LUCE'
  const session = location.hash.slice(1);
  if (!session) return;

  const canvas = document.getElementById('surface');
  const context = canvas.getContext('2d', { alpha: true, desynchronized: true });
  let lastNV12 = null;
  let socket = null;
  let connecting = false;
  let backoff = 400;
  let lastFrameAt = 0;
  let frozen = false;
  let enabled = false, comparing = false, scheduled = false, newest = null, lastSequence = -1;
  function clear() {
    context.clearRect(0, 0, canvas.width, canvas.height);
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
      return;
    }
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

  connect();
})();
