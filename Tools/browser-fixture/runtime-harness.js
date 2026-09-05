// Test only: exercise the actual companion with real structured-clone ports.
const companionWorker = new Worker('worker-harness.js');
function runtimePort(messagePort) {
  const listeners = [];
  messagePort.onmessage = e => listeners.forEach(fn => fn(e.data));
  return { name:'lucid', postMessage:m=>messagePort.postMessage(m),
    onMessage:{addListener:fn=>listeners.push(fn)}, onDisconnect:{addListener:()=>{}} };
}
window.chrome = { runtime: { id:'lucid-local-integration',
  getURL: path => new URL(path,location.href).href,
  connect: () => {const channel=new MessageChannel();const port=runtimePort(channel.port1);companionWorker.postMessage({port:channel.port2},[channel.port2]);return port;}
}};
