// Lucid browser companion — background.
//
// Relays everything between content scripts and the native app over a loopback
// WebSocket: geometry reports, control messages, and the video frames in both
// directions. It has to be the worker rather than the content script, because
// most large sites set a Content-Security-Policy that forbids connecting to
// ws://127.0.0.1 and that policy binds the content script too. The worker is
// governed by the extension's own policy, so it can always reach the app.
if (typeof importScripts === 'function') importScripts('bridge-auth.js');

const runtime = (globalThis.browser && browser.runtime) ? browser.runtime : chrome.runtime;
const BRIDGE_URL = 'ws://127.0.0.1:47811';

let socket = null;
let connecting = false;
let bridgeReady = false;
let lastStatus = null;
let backoff = 1000;
const pending = [];
const portSessions = new Map();
const ENHANCED_MAGIC = 0x4c554345; // 'LUCE': an enhanced frame, app -> page

/// Which port owns a session, so an enhanced frame goes to the one tab that
/// asked for it rather than to every tab that happens to be open.
function portForSession(session) {
  for (const [port, sessions] of portSessions) if (sessions.has(session)) return port;
  return null;
}

async function connect() {
  if (connecting) return;
  if (socket && (socket.readyState === 0 || socket.readyState === 1)) return;
  connecting = true;
  bridgeReady = false;
  let token;
  try {
    token = await lucidFetchToken();
  } catch (e) {
    connecting = false;
    socket = null;
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 10000);
    return;
  }
  try {
    socket = new WebSocket(BRIDGE_URL);
  } catch (e) {
    connecting = false;
    socket = null;
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 10000);
    return;
  }
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => {
    connecting = false;
    backoff = 1000;
    try { socket.send(lucidHello(token)); } catch (e) {}
    bridgeReady = true;
    while (pending.length) socket.send(pending.shift());
  };
  socket.onmessage = (event) => {
    if (!(event.data instanceof ArrayBuffer)) {
      let message; try { message = JSON.parse(event.data); } catch { return; }
      if (message.type === 'status') lastStatus = message;
      // Small status and credit acknowledgments travel through runtime ports.
      for (const port of portSessions.keys()) {
        try { port.postMessage(message); } catch (e) {}
      }
      return;
    }
    // Enhanced frames go directly to the authenticated surface iframe.
    // This worker carries input frames and small control messages only.
  };
  socket.onclose = () => {
    connecting = false;
    bridgeReady = false;
    socket = null;
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 10000);
  };
  socket.onerror = () => {};
}

function deliver(message) {
  const text = JSON.stringify(message);
  if (socket && socket.readyState === 1 && bridgeReady) {
    socket.send(text);
  } else {
    // Keep only the newest message per session while disconnected.
    const index = pending.findIndex(p => p.includes(`"session":"${message.session}"`));
    if (index >= 0) pending.splice(index, 1);
    pending.push(text);
    if (pending.length > 32) pending.shift();
    connect();
  }
}

runtime.onConnect.addListener((port) => {
  if (port.name !== 'lucid') return;
  portSessions.set(port, new Set());
  const version = /(?:Chrome|Edg)\/(\d+)/.exec(navigator.userAgent);
  const binary = !version || Number(version[1]) >= 148;
  port.postMessage({ type: 'transport', binary });
  if (lastStatus) port.postMessage(lastStatus);
  port.onMessage.addListener((message) => {
    if (!message) return;
    if (message.t === 'frame' || (message.t === 'b64' && typeof message.b === 'string')) {
      // Typed-array input on modern browsers; base64 on older JSON ports.
      // Dropping it when the socket is down is right: a stale frame is worth
      // nothing and queueing them costs memory.
      if (socket && socket.readyState === 1 && bridgeReady) {
        if (socket.bufferedAmount > (2 << 20)) return;
        if (!lastStatus?.enabled || lastStatus.activeSession !== message.session) return;
        if (message.t === 'frame') {
          if (message.bytes instanceof Uint8Array) socket.send(message.bytes);
        } else {
          const binary = atob(message.b);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
          socket.send(bytes);
        }
      } else { connect(); }
      return;
    }
    if (message.session) portSessions.get(port)?.add(message.session);
    deliver(message);
  });
  port.onDisconnect.addListener(() => {
    // Acknowledge the disconnect reason so Chrome does not log it as unchecked;
    // a page entering the back/forward cache is a normal way for this to happen.
    void (runtime.lastError && runtime.lastError.message);
    const sessions = portSessions.get(port) || new Set();
    portSessions.delete(port);
    for (const session of sessions) {
      deliver({ type: 'gone', browser: 'unknown', session, title: '', visible: false,
        screenX: 0, screenY: 0, outerWidth: 0, outerHeight: 0, innerWidth: 0, innerHeight: 0, dpr: 1,
        video: null, moving: false, hover: false, cutouts: [], ts: 0 });
    }
  });
});

// Reconnect periodically; also keeps the worker alive in Chrome.
if (globalThis.chrome && chrome.alarms) {
  chrome.alarms.create('lucid-keepalive', { periodInMinutes: 0.5 });
  chrome.alarms.onAlarm.addListener(() => connect());
}
connect();
