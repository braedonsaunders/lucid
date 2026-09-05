#!/usr/bin/env python3
"""Build an isolated, session-restricted transferable input-transport experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError('source changed; review experimental patch context: '+before[:80])
    return text.replace(before, after)


def build(source, out):
    if out.exists():
        raise ValueError('fresh experimental extension directory required')
    shutil.copytree(source, out)
    original = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.rglob('*') if p.is_file()}
    content = (out/'content.js').read_text()
    content = replace(content, '  const gate = new LucidCaptureGate(session);', '''  const gate = new LucidCaptureGate(session);
  let captureLink = null;
  function closeCaptureLink() {
    if (captureLink) captureLink.port.close();
    captureLink = null;
  }
  function connectCaptureLink(frame) {
    closeCaptureLink();
    const channel = new MessageChannel();
    const link = captureLink = {port: channel.port1, ready: false};
    link.port.onmessage = event => {
      const message = event.data;
      if (captureLink !== link || message?.session !== session) return;
      if (message.type === 'captureReady') link.ready = message.ready === true;
      else if (message.type === 'captureReleased') gate.acknowledge(message.seq);
      else if (message.type === 'accepted') bridgeMessage(message);
    };
    frame.contentWindow.postMessage({lucid: 'capture-port', session},
      runtime.getURL('').replace(/\\/$/, ''), [channel.port2]);
  }''')
    content = replace(content, "      frame.src = runtime.getURL('surface.html') + '#' + session;", """      frame.addEventListener('load', () => { if (surface === frame) connectCaptureLink(frame); });
      frame.src = runtime.getURL('surface.html') + '#' + session;""")
    content = replace(content, '  function removeSurface() {', '  function removeSurface() {\n    closeCaptureLink();')
    content = replace(content, '''  function deliverFrame(packet) {
    if (runtime) { sendBinary(packet); return; }
    if (frameSocket && frameSocket.readyState === 1) frameSocket.send(packet);
  }''', '''  function deliverFrame(packet) {
    if (captureLink?.ready) {
      try {
        captureLink.port.postMessage({t: 'capture', session, bytes: packet}, [packet.buffer]);
        stats.socket = 'surface-transfer';
        return true;
      } catch { closeCaptureLink(); }
      if (!packet.byteLength) return false;
    }
    if (runtime) return sendBinary(packet);
    if (frameSocket?.readyState === 1) { frameSocket.send(packet); return true; }
    return false;
  }''')
    content = replace(content, 'deliverFrame(packet);', 'if (!deliverFrame(packet)) { gate.acknowledge(sequence); return; }', count=2)
    (out/'content.js').write_text(content)
    surface = (out/'surface.js').read_text()
    surface = replace(surface, '  let gap = 0;', '''  let gap = 0;
  let capturePort = null;
  function captureReady() {
    capturePort?.postMessage({type: 'captureReady', session,
      ready: !frozen && enabled && socket?.readyState === 1});
  }
  function receiveCapture(event) {
    const message = event.data;
    if (message?.t !== 'capture' || message.session !== session || !(message.bytes instanceof Uint8Array)) return;
    const bytes = message.bytes;
    // This channel forwards only decoded-frame packets for this iframe's own
    // active session. It never forwards hello, tokens, attach or control JSON.
    if (bytes.byteLength < 8 || bytes.byteLength > 32 * 1024 * 1024) return;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (view.getUint32(0, false) !== 0x4c554346) return;
    const length = view.getUint32(4, false);
    if (length > 8192 || length + 8 >= bytes.byteLength) return;
    let header;
    try { header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8+length))); } catch { return; }
    if (header.session !== session || !Number.isSafeInteger(header.seq) || header.seq < 0) return;
    if (frozen || !enabled || socket?.readyState !== 1 || socket.bufferedAmount > 2 * 1024 * 1024) {
      capturePort?.postMessage({type: 'captureReleased', session, seq: header.seq});
      return;
    }
    try { socket.send(bytes); } catch {
      capturePort?.postMessage({type: 'captureReleased', session, seq: header.seq});
    }
  }''')
    surface = replace(surface, "socket.onclose = () => { connecting = false; socket = null; retry(); };",
                      "socket.onclose = () => { connecting = false; socket = null; captureReady(); retry(); };")
    surface = replace(surface, "          if (message.type === 'status') {", """          if (message.type === 'accepted' && message.session === session) capturePort?.postMessage(message);
          if (message.type === 'status') {""")
    surface = replace(surface, '            if (!enabled) clear();', '            if (!enabled) clear();\n            captureReady();')
    surface = replace(surface, '    const message = event.data;\n    if (message?.lucid', '''    const message = event.data;
    if (message?.lucid === 'capture-port' && message.session === session && event.ports.length === 1) {
      // A page owns its input video. Bind its port only to this session and let
      // authenticated native status remain the authority for enablement.
      capturePort?.close(); capturePort = event.ports[0];
      capturePort.onmessage = receiveCapture;
      captureReady();
      return;
    }
    if (message?.lucid''')
    surface = replace(surface, '    frozen = true;', '    frozen = true;\n    captureReady();')
    (out/'surface.js').write_text(surface)
    for path in out.rglob('*'):
        if path.suffix in ('.js', '.html', '.json'):
            path.write_text(path.read_text().replace('47811', '48111').replace('47812', '48112'))
    receipt = {'purpose': 'isolated transferable input experiment; production extension unchanged',
               'source_files': original, 'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'experiment.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.out)
