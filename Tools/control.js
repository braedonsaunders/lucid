// Sends one control message to the running app and exits: node control.js '{"enabled":true}'
const TOKEN_URL = 'http://127.0.0.1:47812/token';

async function main() {
  const response = await fetch(TOKEN_URL);
  if (!response.ok) throw new Error('token ' + response.status);
  const token = (await response.text()).trim();
  const ws = new WebSocket('ws://127.0.0.1:47811');
  ws.onopen = () => {
    ws.send(JSON.stringify({ type: 'hello', token }));
    ws.send(JSON.stringify({ type: 'control', ...JSON.parse(process.argv[2] || '{}') }));
    setTimeout(() => process.exit(0), 250);
  };
  ws.onerror = () => process.exit(1);
  setTimeout(() => process.exit(0), 3000);
}

main().catch(() => process.exit(1));
