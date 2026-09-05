// Test-only instrumentation: link each canvas acknowledgment to native output PTS.
// Retains metadata only, never pixel buffers. Install before the surface script.
(() => {
  const samples = globalThis.__lucidDrawProbe = [];
  const NativeWebSocket = WebSocket;
  globalThis.WebSocket = class extends NativeWebSocket {
    constructor(...args) {
      super(...args);
      const metadata = this.__lucidProbeMetadata = new Map();
      this.addEventListener('message', event => {
        if (!(event.data instanceof ArrayBuffer)) return;
        try {
          const view = new DataView(event.data);
          if (view.byteLength < 8 || view.getUint32(0) !== 0x4c554345) return;
          const length = view.getUint32(4);
          if (length > 8192 || length + 8 >= view.byteLength) return;
          const value = JSON.parse(new TextDecoder().decode(new Uint8Array(event.data, 8, length)));
          if (typeof value.session !== 'string' || !Number.isSafeInteger(value.seq)
              || value.seq < 0 || !Number.isFinite(value.ts) || value.ts < 0) return;
          metadata.set(value.seq, {session: value.session, sourceTimestamp: value.ts});
          if (metadata.size > 256) metadata.delete(metadata.keys().next().value);
        } catch {}
      });
    }
    send(data) {
      if (typeof data === 'string') {
        try {
          const value = JSON.parse(data);
          if (value.type === 'presented' && samples.length < 250000) {
            const entry = this.__lucidProbeMetadata.get(value.seq);
            samples.push({at: performance.timeOrigin + performance.now(), session: value.session,
              seq: value.seq, latencyMilliseconds: value.latencyMilliseconds,
              sourceTimestamp: entry?.session === value.session ? entry.sourceTimestamp : null});
            this.__lucidProbeMetadata.delete(value.seq);
          }
        } catch {}
      }
      return super.send(data);
    }
  };
})();
