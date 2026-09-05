const { test } = require('node:test');
const assert = require('node:assert/strict');
const { CaptureGate } = require('../BrowserExtension/stream-policy');
function gate() { let time = 0; const value = new CaptureGate('ours', () => time); return { value, tick: n => time += n }; }
test('off, foreign sessions and expired connection leases never capture', () => {
  const { value: g, tick } = gate(); assert.equal(g.ready, false);
  g.status({ enabled: true, activeSession: 'other' }); assert.equal(g.reserve(1), false);
  g.status({ enabled: true, activeSession: 'ours' }); assert.equal(g.reserve(1), true);
  tick(2501); assert.equal(g.ready, false);
  g.status({ enabled: false, activeSession: 'ours' }); assert.equal(g.allowed, false); assert.equal(g.pending.size, 0);
});
test('credits bound in-flight copies and recover after a lost acknowledgment', () => {
  const { value: g, tick } = gate(); g.status({ enabled: true, activeSession: 'ours' });
  assert.equal(g.reserve(1), true); assert.equal(g.reserve(2), true); assert.equal(g.reserve(3), false);
  g.acknowledge(1); assert.equal(g.reserve(3), true); assert.equal(g.pending.size, 2);
  tick(1001); assert.equal(g.reserve(4), true); assert.equal(g.pending.size, 1);
  g.reset(); assert.equal(g.allowed, false); assert.equal(g.pending.size, 0);
});
test('measured processing time paces capture without making a backlog', () => {
  const { value: g, tick } = gate(); g.status({ enabled: true, activeSession: 'ours', captureIntervalMilliseconds: 40 });
  assert.equal(g.reserve(1), true); g.acknowledge(1); tick(39); assert.equal(g.ready, false);
  tick(1); assert.equal(g.ready, true);
});

test('idle credit admits measured 8.3 ms callback bursts at 50 fps', () => {
  const { value: g, tick } = gate();
  g.status({ enabled: true, activeSession: 'ours', captureIntervalMilliseconds: 10 });
  assert.equal(g.reserve(0), true); g.acknowledge(0);
  for (let seq = 1; seq <= 100; seq++) {
    tick(seq % 2 ? 31.7 : 8.3);
    assert.equal(g.reserve(seq), true, `callback ${seq}`); g.acknowledge(seq);
  }
});

test('long idle grants only a two-frame burst and retains the sustained processing budget', () => {
  const { value: g, tick } = gate();
  g.status({ enabled: true, activeSession: 'ours', captureIntervalMilliseconds: 10 });
  g.reserve(0); g.acknowledge(0); tick(1000);
  let accepted = 0;
  for (let ms = 0; ms < 500; ms++) {
    for (let attempt = 0; attempt < 4; attempt++) {
      const seq = ms * 4 + attempt + 1;
      if (g.reserve(seq)) { accepted++; g.acknowledge(seq); }
    }
    assert.equal(accepted, 2 + Math.floor(ms / 10));
    tick(1);
  }
});

test('idle credit cannot bypass pending copies, Off, lease expiry or reset', () => {
  const { value: g, tick } = gate();
  const status = { enabled: true, activeSession: 'ours', captureIntervalMilliseconds: 10 };
  g.status(status); g.reserve(0); g.acknowledge(0); tick(100);
  assert.equal(g.reserve(1), true); assert.equal(g.reserve(2), true);
  tick(100); assert.equal(g.reserve(3), false);
  g.status({ ...status, enabled: false }); assert.equal(g.reserve(3), false);
  g.status(status); assert.equal(g.reserve(3), true); g.acknowledge(3);
  assert.equal(g.reserve(4), false);
  tick(2501); assert.equal(g.reserve(4), false);
  g.status(status); assert.equal(g.reserve(4), true); g.acknowledge(4);
  assert.equal(g.reserve(5), false);
  g.reset(); assert.equal(g.reserve(5), false);
  g.status(status); assert.equal(g.reserve(5), true);
});

test('updated processing cost preserves fractional debt and supports zero pacing', () => {
  const { value: g, tick } = gate();
  const status = { enabled: true, activeSession: 'ours', captureIntervalMilliseconds: 40 };
  g.status(status); g.reserve(0); g.acknowledge(0); tick(20);
  g.status({ ...status, captureIntervalMilliseconds: 20 });
  tick(9); assert.equal(g.ready, false); tick(1); assert.equal(g.reserve(1), true); g.acknowledge(1);
  g.status({ ...status, captureIntervalMilliseconds: 0 }); assert.equal(g.reserve(2), true); g.acknowledge(2);
  g.status(status); assert.equal(g.reserve(3), true); g.acknowledge(3);
  assert.equal(g.reserve(4), false);
});

test('HDR names and explicit unknown color metadata never enter SDR capture', () => {
  const { colorBlockReason } = require('../BrowserExtension/stream-policy.js');
  for (const transfer of ['pq', 'hlg', 'smpte2084', 'arib-std-b67']) {
    assert.equal(colorBlockReason({ transfer }), 'HDR video stays with the browser');
  }
  for (const field of ['primaries', 'transfer', 'matrix']) {
    assert.ok(colorBlockReason({ [field]: 'future-color-space' }));
  }
  assert.equal(colorBlockReason({ primaries: 'smpte432', transfer: 'iec61966-2-1', matrix: 'rgb' }), null);
  assert.equal(colorBlockReason({ primaries: null, transfer: null, matrix: null }), null);
});
