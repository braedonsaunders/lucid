window.gateTrace=[];
Object.defineProperty(globalThis,'LucidCaptureGate',{configurable:true,set(Gate){
 const ready=Object.getOwnPropertyDescriptor(Gate.prototype,'ready').get;
 const reserve=Gate.prototype.reserve;
 function record(gate,accepted,last) { window.gateTrace.push({at:performance.now(),accepted,last,interval:gate.interval,pending:gate.pending.size,allowed:gate.allowed}); if(window.gateTrace.length>10000)window.gateTrace.shift(); }
 Object.defineProperty(Gate.prototype,'ready',{get(){const result=ready.call(this);if(!result)record(this,false,this.last);return result;}});
 Gate.prototype.reserve=function(seq){const last=this.last;const accepted=reserve.call(this,seq);if(accepted)record(this,true,last);return accepted;};
 Object.defineProperty(globalThis,'LucidCaptureGate',{value:Gate,writable:true,configurable:true});
}});
