// Drives the lab page over the Chrome DevTools Protocol so tests are repeatable.
//   node Tools/lab_drive.js <port> '<js expression>'
// Prints the JSON result of the expression evaluated in the page.
const port = process.argv[2] || '9222';
const expression = process.argv[3];
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
const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
console.log(JSON.stringify(result.result?.value ?? result.exceptionDetails?.exception?.description ?? null));
ws.close();
