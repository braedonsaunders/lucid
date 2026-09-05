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
  globalThis.LucidCaptureGate = CaptureGate;
  if (typeof module !== 'undefined') module.exports = { CaptureGate };
})();
