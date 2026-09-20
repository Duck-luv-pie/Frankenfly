export const DATASET='male-cns:v1.0';
export const CATEGORIES=['Optic neurons','Central brain neurons','VNC neurons','Ascending / descending'];
export const PALETTE=['#56b7dc','#77d4b4','#dda469','#b69bea'];
export function parseSkeleton(buffer){
  if(!(buffer instanceof ArrayBuffer)||buffer.byteLength<8)throw new Error('Invalid skeleton header');
  const view=new DataView(buffer),nv=view.getUint32(0,true),ne=view.getUint32(4,true);
  const length=8+nv*12+ne*8;if(nv>5e6||ne>1e7||length>buffer.byteLength)throw new Error('Truncated or oversized skeleton');
  const positions=new Float32Array(buffer,8,nv*3),edges=new Uint32Array(buffer,8+nv*12,ne*2);
  validateGeometry(positions,edges);return {positions,edges};
}
export function parseBundle(buffer){
  if(buffer.byteLength<4)throw new Error('Invalid bundle');
  const view=new DataView(buffer),count=view.getUint32(0,true);let offset=4;const result=[];
  if(count>1024)throw new Error('Invalid bundle count');
  for(let i=0;i<count;i++){
    if(offset+12>buffer.byteLength)throw new Error('Truncated bundle header');
    const id=String(view.getUint32(offset,true)),nv=view.getUint32(offset+4,true),ne=view.getUint32(offset+8,true);offset+=12;
    if(offset+nv*12+ne*8>buffer.byteLength)throw new Error('Truncated bundle geometry');
    const positions=new Float32Array(buffer,offset,nv*3);offset+=nv*12;
    const edges=new Uint32Array(buffer,offset,ne*2);offset+=ne*8;
    validateGeometry(positions,edges);result.push({id,positions,edges});
  }
  if(offset!==buffer.byteLength)throw new Error('Unexpected bundle bytes');return result;
}
function validateGeometry(positions,edges){
  for(const value of positions)if(!Number.isFinite(value))throw new Error('Non-finite skeleton coordinate');
  for(const index of edges)if(index>=positions.length/3)throw new Error('Skeleton edge outside vertex array');
}
export function validateEvents(payload,catalog){
  if(payload?.dataset!==DATASET)throw new Error(`Events must declare dataset "${DATASET}".`);
  if(!Array.isArray(payload.events)||!payload.events.length||payload.events.length>100000)throw new Error('Provide 1–100,000 events.');
  return payload.events.map((event,i)=>{
    const id=String(event.bodyId);if(!catalog.has(id))throw new Error(`Unknown body ID ${id} at event ${i+1}.`);
    const amplitude=event.amplitude??1;
    if(!Number.isFinite(amplitude)||amplitude<0||amplitude>1)throw new Error(`Amplitude must be between 0 and 1 (event ${i+1}).`);
    if(!Number.isFinite(event.timeMs)||event.timeMs<0||event.timeMs>86400000)throw new Error(`Invalid timeMs at event ${i+1}.`);
    return {bodyId:id,timeMs:event.timeMs,amplitude};
  }).sort((a,b)=>a.timeMs-b.timeMs);
}
export class ActivityState {
  constructor(size=512){this.size=size;this.data=new Float32Array(size*size*4);}
  fire(slot,time,amplitude=1){
    if(!Number.isInteger(slot)||slot<0||slot>=this.size*this.size)throw new Error('Invalid neuron slot');
    if(!Number.isFinite(time)||!Number.isFinite(amplitude)||amplitude<0||amplitude>1)throw new Error('Invalid spike');
    this.data[slot*4]=time;this.data[slot*4+1]=amplitude;
  }
  value(slot,time,decay=.65){const offset=slot*4;return this.data[offset+1]*Math.exp(-Math.max(0,time-this.data[offset])/decay);}
  clear(){this.data.fill(0);}
}
