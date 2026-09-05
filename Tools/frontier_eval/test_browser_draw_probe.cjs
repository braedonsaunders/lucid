const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function setup() {
  class Socket {
    constructor() { this.events = {}; this.sent = []; }
    addEventListener(name, callback) { this.events[name] = callback; }
    send(data) { this.sent.push(data); }
  }
  const context = vm.createContext({WebSocket: Socket, ArrayBuffer, Uint8Array, DataView, TextDecoder,
    performance: {timeOrigin: 1000, now: () => 20}});
  vm.runInContext(fs.readFileSync(path.join(__dirname, 'browser_draw_probe.js'), 'utf8'), context);
  const socket = new context.WebSocket();
  function frame(seq, ts, session = 'own') {
    const header = new TextEncoder().encode(JSON.stringify({seq, ts, session}));
    const buffer = new ArrayBuffer(9 + header.length);
    const view = new DataView(buffer); view.setUint32(0, 0x4c554345); view.setUint32(4, header.length);
    new Uint8Array(buffer, 8, header.length).set(header);
    socket.events.message({data: buffer});
  }
  function ack(seq, session = 'own') {
    const data = JSON.stringify({type: 'presented', seq, session, latencyMilliseconds: 20});
    socket.send(data); assert.equal(socket.sent.at(-1), data);
    return context.__lucidDrawProbe.at(-1);
  }
  return {socket, frame, ack};
}

test('source PTS follows the acknowledged packet, including repeated source frames and loop zero', () => {
  const t = setup();
  t.frame(1, 16000); t.frame(2, 16000); t.frame(3, 0);
  assert.equal(t.ack(1).sourceTimestamp, 16000);
  assert.equal(t.ack(2).sourceTimestamp, 16000);
  assert.equal(t.ack(3).sourceTimestamp, 0);
  assert.equal(t.socket.__lucidProbeMetadata.size, 0);
});

test('missing, malformed and foreign metadata cannot supply source cadence evidence', () => {
  const t = setup();
  t.socket.events.message({data: new ArrayBuffer(3)});
  assert.equal(t.ack(1).sourceTimestamp, null);
  t.frame(2, 16000, 'foreign');
  assert.equal(t.ack(2).sourceTimestamp, null);
  t.frame(3, '16000');
  assert.equal(t.ack(3).sourceTimestamp, null);
});

test('probe bounds retained metadata without retaining full image buffers', () => {
  const t = setup();
  for (let i = 0; i < 300; i++) t.frame(i, i * 16000);
  assert.equal(t.socket.__lucidProbeMetadata.size, 256);
  assert.equal(t.ack(0).sourceTimestamp, null);
  assert.equal(t.ack(299).sourceTimestamp, 299 * 16000);
  assert.deepEqual(Object.keys(t.socket.__lucidProbeMetadata.get(298)), ['session', 'sourceTimestamp']);
});
