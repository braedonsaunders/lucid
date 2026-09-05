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
