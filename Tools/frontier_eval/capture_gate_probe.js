// Append only to an isolated benchmark extension's stream-policy.js.
// Diagnostic counters preserve gate decisions; never ship this instrumentation.
(() => {
  const prototype = globalThis.LucidCaptureGate.prototype;
  const ready = Object.getOwnPropertyDescriptor(prototype, 'ready').get;
  const acknowledge = prototype.acknowledge;
  const reserve = prototype.reserve;
  const totals = {checks: 0, allowed: 0, pending: 0, pacing: 0, disabled: 0,
    expiredPending: 0, reserved: 0, acknowledged: 0, unknownAck: 0};
  let instance = null;
  let rtts = [];
  Object.defineProperty(prototype, 'ready', {get() {
    instance = this;
    const now = this.now();
    for (const at of this.pending.values()) if (now-at > 1000) totals.expiredPending++;
    const result = ready.call(this);
    totals.checks++;
    if (result) totals.allowed++;
    else if (!this.allowed) totals.disabled++;
    else if (this.pending.size >= 2) totals.pending++;
    else totals.pacing++;
    return result;
  }});
  prototype.reserve = function(seq) {
    const result = reserve.call(this, seq);
    if (result) totals.reserved++;
    return result;
  };
  prototype.acknowledge = function(seq) {
    const at = this.pending.get(seq);
    if (at !== undefined) { totals.acknowledged++; rtts.push(this.now()-at); }
    else totals.unknownAck++;
    return acknowledge.call(this, seq);
  };
  setInterval(() => {
    if (!instance) return;
    const sorted = rtts.sort((a,b)=>a-b);
    document.documentElement.dataset.lucidGate = JSON.stringify({
      at: performance.timeOrigin+performance.now(), ...totals,
      pendingNow: instance.pending.size, interval: instance.interval,
      leaseRemaining: instance.expires-instance.now(),
      ackCount: sorted.length,
      ackMeanMs: sorted.length ? sorted.reduce((a,b)=>a+b,0)/sorted.length : null,
      ackP95Ms: sorted.length ? sorted[Math.ceil(sorted.length*.95)-1] : null,
      ackMaxMs: sorted.length ? sorted[sorted.length-1] : null,
    });
    rtts = [];
  }, 1000);
})();
