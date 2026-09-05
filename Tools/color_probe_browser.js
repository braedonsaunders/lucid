async (page) => {
 const report = await page.evaluate((data) => {
  const decode = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  function render(bytes, init, count) {
   const frame = new VideoFrame(bytes, init);
   const actualColor = frame.colorSpace.toJSON();
   const canvas = document.createElement('canvas'); canvas.width = init.codedWidth; canvas.height = init.codedHeight;
   const ctx = canvas.getContext('2d', {colorSpace:'srgb', willReadFrequently:true});
   ctx.drawImage(frame, 0, 0); frame.close();
   const pixels = ctx.getImageData(0,0,canvas.width,canvas.height).data;
   const patches = Array.from({length:count}, (_,i) => {
    const p=(16*canvas.width+16+i*32)*4; return Array.from(pixels.slice(p,p+3));
   });
   return {actualColor, patches};
  }
  const cases=data.cases.map(c => {
   const original=render(decode(c.sourceBase64),{format:c.format,codedWidth:c.width,codedHeight:c.height,timestamp:0,colorSpace:c.colorSpace},data.colors.length);
   const packet=decode(c.packetBase64); const view=new DataView(packet.buffer);
   if(view.getUint32(0)!==0x4c554345) throw new Error('bad output packet');
   const length=view.getUint32(4); const header=JSON.parse(new TextDecoder().decode(packet.subarray(8,8+length)));
   const normalized=render(packet.subarray(8+length),{format:header.format,codedWidth:header.w,codedHeight:header.h,timestamp:0,colorSpace:header.colorSpace},data.colors.length);
   const errors=original.patches.map((p,i)=>p.map((v,j)=>normalized.patches[i][j]-v));
   return {id:c.id,original,normalized,errors,maxError:Math.max(...errors.flat().map(Math.abs))};
  });
  return {purpose:data.purpose,userAgent:navigator.userAgent,canvasColorSpace:'srgb',cases};
 }, __PROBE_DATA__);
 return report;
}
