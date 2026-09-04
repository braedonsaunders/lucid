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
  let socket = null;
  let connecting = false;
  let backoff = 400;
  let lastFrameAt = 0;
  let frozen = false;
  // Height, in device pixels, of the strip at the bottom left clear so the
  // browser's own video controls show through. The content script measures it.
  let gap = 0;

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
    socket.onclose = () => { connecting = false; socket = null; retry(); };
    socket.onerror = () => {};
    socket.onmessage = (event) => {
      if (!(event.data instanceof ArrayBuffer)) return;
      const view = new DataView(event.data);
      if (view.byteLength < 8 || view.getUint32(0, false) !== ENHANCED_MAGIC) return;
      const headerLength = view.getUint32(4, false);
      let meta;
      try {
        meta = JSON.parse(new TextDecoder().decode(new Uint8Array(event.data, 8, headerLength)));
      } catch (e) { return; }
      if (meta.session !== session) return;
      draw(meta.w, meta.h, new Uint8Array(event.data, 8 + headerLength), meta.format);
    };
  }

  function retry() {
    if (frozen) return;
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 6000);
  }

  function draw(width, height, pixels, format) {
    if (width < 2 || height < 2) return;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    lastNV12 = { width, height, pixels, format };
    paint();
    lastFrameAt = performance.now();
  }

  function paint() {
    if (!lastNV12) return;
    const { width, height, pixels, format } = lastNV12;
    if (format !== 'NV12' || typeof VideoFrame !== 'function') return;
    try {
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
      context.drawImage(frame, 0, 0, width, height);
      frame.close();
    } catch (e) {
      return;
    }
    if (gap > 0) context.clearRect(0, height - Math.min(gap, height), width, Math.min(gap, height));
  }

  // The content script measures the control strip and reports whether the
  // browser is currently showing it.
  addEventListener('message', (event) => {
    const message = event.data;
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
    if (socket) { try { socket.close(); } catch (e) {} socket = null; }
  });
  addEventListener('pageshow', (event) => {
    if (!event.persisted) return;
    frozen = false;
    connect();
  });

  connect();
})();
