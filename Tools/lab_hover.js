// Moves the pointer over the video with a trusted input event, so the browser's
// own controls appear exactly as they do for a person.
//   node Tools/lab_hover.js <port> [fractionFromBottom]
const port = process.argv[2] || '9222';
const fromBottom = Number(process.argv[3] || 30);
const targets = await (await fetch(`http://localhost:${port}/json`)).json();
const page = targets.find(t => t.type === 'page' && t.url.includes('lab.html'));
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise(r => ws.onopen = r);
const send = (method, params) => new Promise(resolve => {
  const id = Math.floor(Math.random() * 1e6);
  const onMessage = (e) => { const m = JSON.parse(e.data); if (m.id === id) { ws.removeEventListener('message', onMessage); resolve(m.result); } };
  ws.addEventListener('message', onMessage);
  ws.send(JSON.stringify({ id, method, params }));
});
const box = JSON.parse((await send('Runtime.evaluate', {
  expression: "(()=>{const r=document.getElementById('video').getBoundingClientRect();return JSON.stringify({x:r.x,y:r.y,w:r.width,h:r.height});})()",
  returnByValue: true
})).result.value);
const x = box.x + box.w / 2;
for (const y of [box.y + box.h / 2, box.y + box.h - fromBottom - 6, box.y + box.h - fromBottom]) {
  await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y, buttons: 0 });
  await new Promise(r => setTimeout(r, 120));
}
console.log(JSON.stringify({ hoveredAt: { x, y: box.y + box.h - fromBottom } }));
ws.close();
