// Same as site_check but throttles the network first, so an adaptive player
// serves a low resolution - the case Lucid exists for.
const port = process.argv[2], url = process.argv[3], kbps = Number(process.argv[4] || 900);
const t = await (await fetch(`http://localhost:${port}/json`)).json();
const page = t.find(x => x.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise(r => ws.onopen = r);
const send = (m, p) => new Promise(res => { const id = Math.floor(Math.random()*1e6);
  const on = (e) => { const j = JSON.parse(e.data); if (j.id === id) { ws.removeEventListener('message', on); res(j.result); } };
  ws.addEventListener('message', on); ws.send(JSON.stringify({ id, method: m, params: p })); });
const sleep = ms => new Promise(r => setTimeout(r, ms));
await send('Network.enable', {});
await send('Network.emulateNetworkConditions', {
  offline: false, latency: 40,
  downloadThroughput: kbps * 1024 / 8, uploadThroughput: kbps * 1024 / 8,
});
await send('Page.enable', {});
await send('Page.navigate', { url });
await sleep(12000);
const probe = `(() => {
  const vs = [...document.querySelectorAll('video')];
  const v = vs.find(x => !x.paused && x.videoWidth) || vs.find(x => x.videoWidth) || vs[0];
  if (!v) return JSON.stringify({ host: location.host, video: null });
  const r = v.getBoundingClientRect();
  return JSON.stringify({ host: location.host, decoded: v.videoWidth + 'x' + v.videoHeight,
    box: Math.round(r.width) + 'x' + Math.round(r.height),
    physical: Math.round(r.width * devicePixelRatio) + 'x' + Math.round(r.height * devicePixelRatio),
    ratio: +(r.width * devicePixelRatio / Math.max(v.videoWidth,1)).toFixed(2),
    cx: Math.round(r.x + r.width/2), cy: Math.round(r.y + r.height/2),
    paused: v.paused, overlay: !!document.querySelector('canvas[data-lucid]') });
})()`;
let info = JSON.parse((await send('Runtime.evaluate', { expression: probe, returnByValue: true })).result.value);
for (let i = 0; i < 2 && info.paused; i++) {
  await send('Input.dispatchMouseEvent', { type:'mousePressed', x: info.cx||600, y: info.cy||400, button:'left', clickCount:1, buttons:1 });
  await send('Input.dispatchMouseEvent', { type:'mouseReleased', x: info.cx||600, y: info.cy||400, button:'left', clickCount:1, buttons:0 });
  await sleep(6000);
  info = JSON.parse((await send('Runtime.evaluate', { expression: probe, returnByValue: true })).result.value);
}
await sleep(10000);
console.log((await send('Runtime.evaluate', { expression: probe, returnByValue: true })).result.value);
ws.close();
