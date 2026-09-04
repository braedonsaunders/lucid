// Shared token fetch for the loopback bridge. The app issues a per-launch
// token on http://127.0.0.1:47812/token; a webpage cannot read it (wrong
// Origin) and cannot send frames without it.
const LUCID_TOKEN_URL = 'http://127.0.0.1:47812/token';

async function lucidFetchToken() {
  const response = await fetch(LUCID_TOKEN_URL, { cache: 'no-store' });
  if (!response.ok) throw new Error('lucid token ' + response.status);
  const token = (await response.text()).trim();
  if (!token) throw new Error('lucid token empty');
  return token;
}

function lucidHello(token) {
  return JSON.stringify({ type: 'hello', token });
}
