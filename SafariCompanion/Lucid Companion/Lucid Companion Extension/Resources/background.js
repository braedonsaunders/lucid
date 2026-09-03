// Lucid browser companion — background.
//
// Relays geometry reports from content scripts to the native app over a
// loopback WebSocket, and tells the app when a tab goes away.
const runtime = (globalThis.browser && browser.runtime) ? browser.runtime : chrome.runtime;
const BRIDGE_URL = 'ws://127.0.0.1:47811';

let socket = null;
let backoff = 1000;
const pending = [];
const portSessions = new Map();

function connect() {
  if (socket && (socket.readyState === 0 || socket.readyState === 1)) return;
  try {
    socket = new WebSocket(BRIDGE_URL);
  } catch (e) {
    socket = null;
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 10000);
    return;
  }
  socket.onopen = () => {
    backoff = 1000;
    while (pending.length) socket.send(pending.shift());
  };
  socket.onclose = () => {
    socket = null;
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 10000);
  };
  socket.onerror = () => {};
}

function deliver(message) {
  const text = JSON.stringify(message);
  if (socket && socket.readyState === 1) {
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
  port.onMessage.addListener((message) => {
    if (message && message.session) portSessions.get(port)?.add(message.session);
    deliver(message);
  });
  port.onDisconnect.addListener(() => {
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
