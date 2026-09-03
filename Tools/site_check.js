// Navigates, clicks to start playback like a person would, then reports what
// the page has and whether Lucid drew into it.
//   node Tools/site_check.js <port> <url>
const port = process.argv[2], url = process.argv[3];
const t = await (await fetch(`http://localhost:${port}/json`)).json();
const page = t.find(x => x.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise(r => ws.onopen = r);
const send = (m, p) => new Promise(res => {
  const id = Math.floor(Math.random() * 1e6);
  const on = (e) => { const j = JSON.parse(e.data); if (j.id === id) { ws.removeEventListener('message', on); res(j.result); } };
  ws.addEventListener('message', on); ws.send(JSON.stringify({ id, method: m, params: p }));
});
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
await send('Page.enable', {});
await send('Page.navigate', { url });
await sleep(9000);
const probe = `(() => {
  const vs = [...document.querySelectorAll('video')];
  const v = vs.find(x => !x.paused && x.videoWidth) || vs.find(x => x.videoWidth) || vs[0];
  if (!v) return JSON.stringify({ host: location.host, videos: vs.length, video: null });
  const r = v.getBoundingClientRect();
  return JSON.stringify({ host: location.host, videos: vs.length,
    decoded: v.videoWidth + 'x' + v.videoHeight, box: Math.round(r.width) + 'x' + Math.round(r.height),
    cx: Math.round(r.x + r.width / 2), cy: Math.round(r.y + r.height / 2),
    paused: v.paused, overlay: !!document.querySelector('canvas[data-lucid]') });
})()`;
let info = JSON.parse((await send('Runtime.evaluate', { expression: probe, returnByValue: true })).result.value);
// Click the middle of the player, twice if needed: many sites use the first
// click for consent or to dismiss an overlay.
for (let attempt = 0; attempt < 2 && info.paused; attempt++) {
  const x = info.cx || 700, y = info.cy || 400;
  await send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1, buttons: 1 });
  await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', clickCount: 1, buttons: 0 });
  await sleep(5000);
  info = JSON.parse((await send('Runtime.evaluate', { expression: probe, returnByValue: true })).result.value);
}
await sleep(6000);
info = JSON.parse((await send('Runtime.evaluate', { expression: probe, returnByValue: true })).result.value);
console.log(JSON.stringify(info));
ws.close();
