// Navigates the test browser to a URL and reports what the page ended up with.
//   node Tools/lab_nav.js <port> <url> [waitMs]
const port = process.argv[2], url = process.argv[3], wait = Number(process.argv[4] || 9000);
const targets = await (await fetch(`http://localhost:${port}/json`)).json();
const page = targets.find(t => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise(r => ws.onopen = r);
const send = (method, params) => new Promise(resolve => {
  const id = Math.floor(Math.random() * 1e6);
  const onMessage = (e) => { const m = JSON.parse(e.data); if (m.id === id) { ws.removeEventListener('message', onMessage); resolve(m.result); } };
  ws.addEventListener('message', onMessage);
  ws.send(JSON.stringify({ id, method, params }));
});
await send('Page.enable', {});
await send('Page.navigate', { url });
await new Promise(r => setTimeout(r, wait));
const probe = `(() => {
  const vs = [...document.querySelectorAll('video')];
  const v = vs.find(x => !x.paused && x.videoWidth) || vs.find(x => x.videoWidth) || vs[0];
  if (!v) return JSON.stringify({ host: location.host, videos: vs.length, video: null });
  const r = v.getBoundingClientRect();
  return JSON.stringify({ host: location.host, videos: vs.length,
    decoded: v.videoWidth + 'x' + v.videoHeight,
    box: Math.round(r.width) + 'x' + Math.round(r.height),
    paused: v.paused, dpr: devicePixelRatio,
    overlay: !!document.querySelector('canvas[data-lucid]') });
})()`;
const out = (await send('Runtime.evaluate', { expression: probe, returnByValue: true })).result.value;
console.log(out);
ws.close();
