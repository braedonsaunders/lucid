// Shared by browser companions and the transport regression tests.
(() => {
  class CaptureGate {
    constructor(session, now = () => performance.now()) {
      this.session = session; this.now = now; this.expires = 0;
      this.enabled = false; this.pending = new Map(); this.interval = 0; this.last = -Infinity;
    }
    status(message) {
      this.enabled = message.enabled === true && message.activeSession === this.session;
      this.interval = Math.max(0, Number(message.captureIntervalMilliseconds) || 0);
      this.expires = this.now() + 2500;
      if (!this.enabled) this.pending.clear();
    }
    get allowed() { return this.enabled && this.now() < this.expires; }
    get ready() {
      const now = this.now();
      for (const [seq, at] of this.pending) if (now - at > 1000) this.pending.delete(seq);
      return this.allowed && this.pending.size < 2 && now - this.last >= this.interval;
    }
    reserve(seq) {
      if (!this.ready) return false;
      this.pending.set(seq, this.now()); this.last = this.now(); return true;
    }
    acknowledge(seq) { this.pending.delete(seq); }
    reset() { this.enabled = false; this.expires = 0; this.pending.clear(); }
  }
  // WebCodecs August 2026 names plus the names exposed by older browsers.
  function colorBlockReason(color = {}) {
    if (['pq', 'hlg', 'smpte2084', 'arib-std-b67'].includes(color.transfer))
      return 'HDR video stays with the browser';
    const fields = {
      primaries: ['bt709', 'bt470bg', 'smpte170m', 'bt2020', 'smpte432'],
      transfer: ['bt709', 'smpte170m', 'iec61966-2-1', 'linear'],
      matrix: ['rgb', 'bt709', 'bt470bg', 'smpte170m', 'bt2020-ncl'],
    };
    for (const [field, supported] of Object.entries(fields))
      if (color[field] != null && !supported.includes(color[field]))
        return 'This video color format stays with the browser';
    return null;
  }
  globalThis.LucidColorBlockReason = colorBlockReason;
  globalThis.LucidCaptureGate = CaptureGate;
  if (typeof module !== 'undefined') module.exports = { CaptureGate, colorBlockReason };
})();
