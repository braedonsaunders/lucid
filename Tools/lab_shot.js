// Screenshots the video box as it is actually rendered, so enhancement can be
// compared in display space rather than in whatever buffer size the bridge used.
//   node Tools/lab_shot.js <port> <outfile.png>
import { writeFileSync } from 'node:fs';
const port = process.argv[2] || '9222';
const out = process.argv[3];
const targets = await (await fetch(`http://localhost:${port}/json`)).json();
const page = targets.find(t => t.type === 'page' && t.url.includes('lab.html'));
if (!page) { console.error('lab.html not open'); process.exit(1); }
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise(r => ws.onopen = r);
const send = (method, params) => new Promise(resolve => {
  const id = Math.floor(Math.random() * 1e6);
  const onMessage = (e) => { const m = JSON.parse(e.data); if (m.id === id) { ws.removeEventListener('message', onMessage); resolve(m.result); } };
  ws.addEventListener('message', onMessage);
  ws.send(JSON.stringify({ id, method, params }));
});
const box = (await send('Runtime.evaluate', {
  expression: "(()=>{const r=document.getElementById('video').getBoundingClientRect();return JSON.stringify({x:r.x,y:r.y,width:r.width,height:r.height});})()",
  returnByValue: true
})).result.value;
const clip = { ...JSON.parse(box), scale: 1 };
const shot = await send('Page.captureScreenshot', { format: 'png', clip, captureBeyondViewport: false });
writeFileSync(out, Buffer.from(shot.data, 'base64'));
console.log(JSON.stringify({ written: out, clip }));
ws.close();
