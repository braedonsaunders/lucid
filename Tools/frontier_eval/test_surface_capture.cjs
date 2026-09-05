const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const script = process.argv[2] || path.resolve(__dirname, '../../BrowserExtension/surface.js');
const source = fs.readFileSync(script, 'utf8');

async function surface() {
  const listeners = {};
  const parent = {};
  const sent = [];
  let socket;
  const canvas = {width: 1, height: 1, style: {}, getContext: () => ({clearRect() {}})};
  class Socket {
    constructor() { socket = this; this.readyState = 1; this.bufferedAmount = 0; }
    send(value) { sent.push(value); }
    close() { this.readyState = 3; }
  }
  vm.runInNewContext(source, {
    location: {hash: '#own'}, document: {getElementById: () => canvas}, parent,
    WebSocket: Socket, Uint8Array, ArrayBuffer, DataView, TextDecoder,
    lucidFetchToken: async () => 'test-token', lucidHello: () => 'test-hello',
    setTimeout: () => 0, setInterval: () => 0, requestAnimationFrame: () => 0,
    addEventListener: (name, callback) => { listeners[name] = callback; },
    performance: {now: () => 0, timeOrigin: 0},
  });
  await new Promise(setImmediate);
  socket.onopen();
  const port = {messages: [], postMessage(message) { this.messages.push(message); }, close() {}};
  const bind = (origin = parent, session = 'own', candidate = port) => listeners.message({
    source: origin, data: {lucid: 'capture-port', session}, ports: [candidate],
  });
  const status = enabled => socket.onmessage({data: JSON.stringify({type: 'status', enabled, activeSession: 'own'})});
  const frames = () => sent.filter(value => value instanceof Uint8Array);
  return {socket, port, bind, status, frames, listeners};
}

function packet(session = 'own', seq = 7) {
  const header = new TextEncoder().encode(JSON.stringify({session, seq}));
  const bytes = new Uint8Array(8 + header.length + 4);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0x4c554346); view.setUint32(4, header.length);
  bytes.set(header, 8);
  return bytes;
}

test('only own active decoded frames reach the authenticated socket', async () => {
  const s = await surface(); s.bind();
  s.port.onmessage({data: {t: 'capture', session: 'own', bytes: packet()}});
  assert.equal(s.frames().length, 0);
  assert.equal(s.port.messages.at(-1).type, 'captureReleased');
  s.status(true);
  const bytes = packet();
  s.port.onmessage({data: {t: 'capture', session: 'own', bytes}});
  assert.equal(s.frames().length, 1);
  assert.equal(s.frames()[0], bytes);
  s.socket.onmessage({data: JSON.stringify({type: 'accepted', session: 'own', seq: 7})});
  assert.equal(s.port.messages.at(-1).type, 'accepted');
  s.status(false);
  s.port.onmessage({data: {t: 'capture', session: 'own', bytes: packet()}});
  assert.equal(s.frames().length, 1);
});

test('foreign parent, session and non-frame commands cannot claim the relay', async () => {
  const s = await surface();
  s.bind({}); assert.equal(s.port.onmessage, undefined);
  s.bind(undefined, 'foreign'); assert.equal(s.port.onmessage, undefined);
  s.bind(); s.status(true);
  for (const data of [
    {t: 'capture', session: 'foreign', bytes: packet()},
    {t: 'capture', session: 'own', bytes: packet('foreign')},
    {t: 'control', session: 'own', enabled: true},
    {t: 'capture', session: 'own', bytes: 'hello'},
    {t: 'capture', session: 'own', bytes: packet('own', -1)},
    {t: 'capture', session: 'own', bytes: packet('own', 1.5)},
  ]) s.port.onmessage({data});
  assert.equal(s.frames().length, 0);
});

test('malformed packets, backpressure and frozen pages do not enqueue frames', async () => {
  const s = await surface(); s.bind(); s.status(true);
  const wrongMagic = packet(); new DataView(wrongMagic.buffer).setUint32(0, 0);
  const wrongLength = packet(); new DataView(wrongLength.buffer).setUint32(4, 9000);
  const nullHeader = new Uint8Array(16);
  new DataView(nullHeader.buffer).setUint32(0, 0x4c554346);
  new DataView(nullHeader.buffer).setUint32(4, 4);
  nullHeader.set(new TextEncoder().encode('null'), 8);
  for (const bytes of [new Uint8Array(7), wrongMagic, wrongLength, nullHeader])
    s.port.onmessage({data: {t: 'capture', session: 'own', bytes}});
  assert.equal(s.frames().length, 0);
  s.socket.bufferedAmount = 2 * 1024 * 1024 + 1;
  s.port.onmessage({data: {t: 'capture', session: 'own', bytes: packet()}});
  assert.equal(s.port.messages.at(-1).type, 'captureReleased');
  s.socket.bufferedAmount = 0;
  s.listeners.pagehide({persisted: true});
  s.port.onmessage({data: {t: 'capture', session: 'own', bytes: packet()}});
  assert.equal(s.frames().length, 0);
});

function contentTransport() {
  const text = fs.readFileSync(path.resolve(__dirname, '../../BrowserExtension/content.js'), 'utf8');
  const lifecycle = text.slice(text.indexOf('  let captureLink = null;'), text.indexOf('  let binaryMessaging'));
  const delivery = text.slice(text.indexOf('  function deliverFrame('), text.indexOf('  function header('));
  assert.ok(lifecycle && delivery);
  const channels = [], credits = new Set(), released = [];
  let fallback = 0;
  const stats = {};
  class Channel {
    constructor() {
      const port = () => ({closed: false, messages: [], close() { this.closed = true; },
        postMessage(...args) { this.messages.push(args); }});
      this.port1 = port(); this.port2 = port(); channels.push(this);
    }
  }
  const api = vm.runInNewContext(`${lifecycle}\n${delivery}\n({connectCaptureLink,closeCaptureLink,deliverFrame})`, {
    MessageChannel: Channel, session: 'own', stats, frameSocket: null,
    gate: {acknowledge(seq) { credits.delete(seq); released.push(seq); }},
    bridgeMessage(message) { credits.delete(message.seq); },
    runtime: {getURL: () => 'chrome-extension://unit/'},
    sendBinary: () => { fallback++; return true; },
  });
  const frame = {contentWindow: {postMessage() {}}};
  return {api, frame, channels, credits, released, stats, fallback: () => fallback};
}

test('iframe replacement releases only direct-route credits and ignores stale callbacks', () => {
  const t = contentTransport(); t.api.connectCaptureLink(t.frame);
  const first = t.channels[0].port1;
  first.onmessage({data: {type: 'captureReady', session: 'own', ready: true}});
  t.credits.add(7); t.credits.add(99); // 99 belongs to the runtime fallback.
  assert.equal(t.api.deliverFrame(packet(), 7), true);
  assert.equal(t.stats.socket, 'surface-transfer');
  t.api.connectCaptureLink(t.frame);
  assert.equal(first.closed, true);
  assert.equal(t.credits.has(7), false);
  assert.equal(t.credits.has(99), true);
  first.onmessage({data: {type: 'captureReady', session: 'own', ready: true}});
  t.api.deliverFrame(packet(), 8);
  assert.equal(t.fallback(), 1);
  assert.equal(t.stats.socket, 'port');
});

test('a disconnected surface returns credits and a failed transfer cannot become unreserved fallback work', () => {
  const t = contentTransport(); t.api.connectCaptureLink(t.frame);
  const port = t.channels[0].port1;
  port.onmessage({data: {type: 'captureReady', session: 'own', ready: true}});
  t.credits.add(7); t.api.deliverFrame(packet(), 7);
  port.onmessage({data: {type: 'captureReady', session: 'own', ready: false}});
  assert.equal(t.credits.size, 0);
  port.onmessage({data: {type: 'captureReady', session: 'own', ready: true}});
  port.postMessage = () => { throw new Error('closed'); };
  t.credits.add(8);
  assert.equal(t.api.deliverFrame(packet(), 8), false);
  assert.equal(t.credits.size, 0);
  assert.equal(t.fallback(), 0);
  t.api.deliverFrame(packet(), 9);
  assert.equal(t.fallback(), 1);
  assert.equal(t.stats.socket, 'port');
});

test('failed iframe handshake closes both ports and preserves fallback', () => {
  const t = contentTransport();
  t.frame.contentWindow.postMessage = () => { throw new Error('unavailable frame'); };
  t.api.connectCaptureLink(t.frame);
  assert.equal(t.channels[0].port1.closed, true);
  assert.equal(t.channels[0].port2.closed, true);
  t.api.deliverFrame(packet(), 7);
  assert.equal(t.fallback(), 1);
  assert.equal(t.stats.socket, 'port');
});
