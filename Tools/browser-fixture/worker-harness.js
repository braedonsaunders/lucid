let connectListener;
globalThis.chrome={runtime:{onConnect:{addListener:fn=>connectListener=fn}},alarms:{create:()=>{},onAlarm:{addListener:()=>{}}}};
onmessage=e=>{const p=e.data.port;const listeners=[];p.onmessage=e=>listeners.forEach(fn=>fn(e.data));connectListener({name:'lucid',postMessage:m=>p.postMessage(m),onMessage:{addListener:fn=>listeners.push(fn)},onDisconnect:{addListener:()=>{}}});};
importScripts('background.js');
