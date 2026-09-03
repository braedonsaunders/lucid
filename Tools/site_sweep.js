// Visits a list of real sites, starts playback the way a person would, and
// reports what Lucid saw. Run with the app and the test browser already up.
//   node Tools/site_sweep.js <port> <url> [label]
const port = process.argv[2], url = process.argv[3];
const targets = await (await fetch(`http://localhost:${port}/json`)).json();
const page = targets.find(t => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise(r => ws.onopen = r);
const send = (m, p) => new Promise(res => { const id = Math.floor(Math.random()*1e6);
  const on = e => { const j = JSON.parse(e.data); if (j.id === id) { ws.removeEventListener('message', on); res(j.result); } };
  ws.addEventListener('message', on); ws.send(JSON.stringify({ id, method: m, params: p })); });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const ev = x => send('Runtime.evaluate', { expression: x, returnByValue: true }).then(r => r.result && r.result.value);
await send('Page.enable', {});
await send('Page.navigate', { url });
await sleep(11000);
const probe = `(() => {
  const vs = [...document.querySelectorAll('video')];
  const v = vs.find(x => !x.paused && x.videoWidth) || vs.find(x => x.videoWidth) || vs[0];
  if (!v) return JSON.stringify({ host: location.host, videos: vs.length, video: null,
    reporter: document.documentElement.hasAttribute('data-video-enhancer-reporter') });
  const r = v.getBoundingClientRect();
  return JSON.stringify({ host: location.host, videos: vs.length,
    decoded: v.videoWidth + 'x' + v.videoHeight,
    ratio: +(r.width * devicePixelRatio / Math.max(v.videoWidth,1)).toFixed(2),
    paused: v.paused, cx: Math.round(r.x + r.width/2), cy: Math.round(r.y + r.height/2),
    reporter: document.documentElement.hasAttribute('data-video-enhancer-reporter'),
    frames: document.documentElement.dataset.lucidFrames || 'none' });
})()`;
let info = JSON.parse(await ev(probe));
for (let i = 0; i < 2 && info.paused; i++) {
  await send('Input.dispatchMouseEvent', { type:'mousePressed', x: info.cx||700, y: info.cy||400, button:'left', clickCount:1, buttons:1 });
  await send('Input.dispatchMouseEvent', { type:'mouseReleased', x: info.cx||700, y: info.cy||400, button:'left', clickCount:1, buttons:0 });
  await sleep(6000);
  info = JSON.parse(await ev(probe));
}
await sleep(9000);
console.log(await ev(probe));
ws.close();
