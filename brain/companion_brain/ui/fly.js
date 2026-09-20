// Fly / Body Lab procedural Drosophila rig, from the user's ~/Documents/ChatGPT/fly project (src/fly.js).
// Local coordinates: +Y dorsal, +Z anterior, +X animal's left. Units: mm. 36 joint axes, see README there.
import * as THREE from 'three';

// Local coordinates: +Y dorsal, +Z anterior, +X animal's left. Units: mm.
export function createFly() {
  const root = new THREE.Group(); root.name = 'Drosophila_male';
  root.userData = { species:'Drosophila melanogaster', sex:'male', units:'millimeter', anatomy:'approximate procedural visual prototype', controller:'joint angles in radians; no neural simulation' };
  const joints = {}, limits = {}, surfaces = [], membranes = [], markers = [], regions = {};
  const mat = (color, roughness=.55, extra={}) => new THREE.MeshStandardMaterial({color, roughness, ...extra});
  const cuticle=mat('#8c713e'), gold=mat('#b39758'), dark=mat('#3a3025'), legMat=mat('#a18547'), eyeMat=mat('#9d3824',.4), veinMat=mat('#9a9b72',.6,{transparent:true,opacity:.65});
  const sphere = new THREE.SphereGeometry(1,32,24);
  function ellipsoid(parent,name,pos,scale,material,region='thorax') {
    const m=new THREE.Mesh(sphere,material); m.name=name;m.position.set(...pos);m.scale.set(...scale);m.castShadow=true;m.receiveShadow=true;parent.add(m);surfaces.push(m);m.userData.region=region;return m;
  }
  function rod(parent,a,b,r1,r2,material,name='segment',region='legs') {
    const av=new THREE.Vector3(...a),bv=new THREE.Vector3(...b),dir=bv.clone().sub(av);
    const mesh=new THREE.Mesh(new THREE.CylinderGeometry(r2,r1,dir.length(),8),material);mesh.name=name;mesh.position.copy(av.add(bv).multiplyScalar(.5));mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir.normalize());mesh.castShadow=true;mesh.userData.region=region;parent.add(mesh);surfaces.push(mesh);return mesh;
  }
  function pivot(parent,name,pos,axis,range,region) {
    const g=new THREE.Group();g.name=name;g.position.set(...pos);parent.add(g);joints[name]={node:g,axis};limits[name]=range;g.userData={joint:true,axis,limits:range,region};
    const marker=new THREE.Mesh(new THREE.SphereGeometry(.034,10,8),mat('#c4ed7a',.3,{emissive:'#779d31',emissiveIntensity:.35}));g.add(marker);marker.visible=false;markers.push(marker);return g;
  }
  const thorax=new THREE.Group();thorax.name='Thorax';root.add(thorax);regions.thorax=thorax;
  ellipsoid(thorax,'Mesothorax',[0,1.02,0],[.38,.36,.52],cuticle);
  ellipsoid(thorax,'Dorsal_scutum',[0,1.18,.02],[.33,.24,.44],gold);
  ellipsoid(thorax,'Scutellum',[0,1.15,-.43],[.22,.16,.22],cuticle);
  const abdomen=pivot(root,'abdomen.pitch',[0,1,-.39],'x',[-.35,.35],'abdomen');regions.abdomen=abdomen;
  // Segments overlap to form a continuous tapered male abdomen with dark terminal tergites.
  for(let i=0;i<7;i++){
    const t=i/6, w=.35*Math.sqrt(1-t*.94), y=-.055-i*.027,z=-.1-i*.134;
    ellipsoid(abdomen,`Abdominal_tergite_${i+1}`,[0,y,z],[w,.265*(1-t*.67),.165],i>=4?dark:gold,'abdomen');
    if(i<4)ellipsoid(abdomen,`Tergite_band_${i+1}`,[0,y+.004,z-.095],[w*.97,.265*(1-t*.67)*1.01,.054],dark,'abdomen');
  }
  const head=pivot(root,'head.yaw',[0,1.07,.56],'y',[-.65,.65],'head');regions.head=head;
  const headPitch=pivot(head,'head.pitch',[0,0,0],'x',[-.4,.4],'head');
  ellipsoid(headPitch,'Head_capsule',[0,0,.08],[.34,.29,.26],gold,'head');
  for(const s of [-1,1]){
    const eye=ellipsoid(headPitch,`${s===1?'L':'R'}_compound_eye`,[s*.278,.035,.12],[.175,.251,.219],eyeMat,'head');eye.rotation.z=-s*.12;
    // Individual ommatidial facets follow the exposed ellipsoidal eye surface.
    const facets=new THREE.InstancedMesh(new THREE.SphereGeometry(1,5,4),mat('#b64a2c',.5),300);let n=0;const dummy=new THREE.Object3D();
    for(let row=1;row<17;row++){const phi=row*Math.PI/17;for(let col=0;col<18;col++){const theta=-Math.PI/2+(col+(row%2)*.5)*Math.PI/18;dummy.position.set(s*(.279+.178*Math.sin(phi)*Math.cos(theta)),.035+.252*Math.cos(phi),.12+.222*Math.sin(phi)*Math.sin(theta));dummy.scale.set(.012,.012,.012);dummy.updateMatrix();facets.setMatrixAt(n++,dummy.matrix);}}
    facets.count=n;facets.name=`${s===1?'L':'R'}_ommatidia`;facets.userData.region='head';headPitch.add(facets);surfaces.push(facets);
    const antenna=pivot(headPitch,`${s===1?'L':'R'}.antenna`,[s*.09,.08,.3],'y',[-.5,.5],'head');
    ellipsoid(antenna,'Pedicel',[0,0,0],[.055,.07,.055],cuticle,'head');ellipsoid(antenna,'Funiculus',[s*.022,-.06,.05],[.053,.094,.046],gold,'head');
    rod(antenna,[s*.025,-.02,.08],[s*.16,.16,.12],.009,.002,dark,'Arista','head');
    for(let j=0;j<6;j++)rod(antenna,[s*(.04+j*.019),j*.027,.086+j*.005],[s*(.10+j*.02),j*.027+.06,.1],.003,.0008,dark,'Arista_branch','head');
  }
  for(const [x,z] of [[-.065,0],[.065,0],[0,.08]])ellipsoid(headPitch,'Ocellus',[x,.28,z],[.023,.02,.024],dark,'head');
  const prob=pivot(headPitch,'proboscis.extension',[0,-.19,.22],'x',[-.8,.5],'head');rod(prob,[0,0,0],[0,-.13,.075],.047,.034,cuticle,'Proboscis','head');ellipsoid(prob,'Labellum',[0,-.14,.08],[.072,.032,.045],gold,'head');
  const legInfo=[];
  for(const s of [1,-1])for(let i=0;i<3;i++){
    const side=s===1?'L':'R',id=`${side}${['F','M','H'][i]}`, z=[.32,-.04,-.35][i];
    const hip=pivot(thorax,`${id}.coxa`,[s*.26,.9,z],'y',[-.65,.65],'legs');
    rod(hip,[0,0,0],[s*.13,-.1,0],.061,.043,legMat,'Coxa');
    const femur=pivot(hip,`${id}.femur`,[s*.13,-.1,0],'z',[-.65,.65],'legs');
    const knee=[s*.36,-.22,[.29,.04,-.29][i]];rod(femur,[0,0,0],knee,.046,.033,legMat,'Femur');
    const tibia=pivot(femur,`${id}.tibia`,knee,'z',[-.7,.7],'legs');
    const ankle=[s*.19,-.43,[.16,.03,-.19][i]];rod(tibia,[0,0,0],ankle,.027,.016,legMat,'Tibia');
    const tarsus=pivot(tibia,`${id}.tarsus`,ankle,'x',[-.8,.8],'legs');
    for(let k=0;k<5;k++)rod(tarsus,[s*.018*k,-.025*k,.023*k],[s*.018*(k+1),-.025*(k+1),.023*(k+1)],.016-k*.002,.013-k*.002,legMat,`Tarsomere_${k+1}`);
    for(const v of [-1,1])rod(tarsus,[s*.09,-.125,.115],[s*.09+v*.021,-.133,.151],.005,.001,dark,'Claw');
    for(let k=1;k<8;k++){const t=k/9;rod(tibia,ankle.map(v=>v*t),[ankle[0]*t+s*.048,ankle[1]*t+.026,ankle[2]*t],.004,.001,dark,'Tibial_bristle');}
    legInfo.push({id,side:s,index:i});
  }
  const wingShape=new THREE.Shape();wingShape.moveTo(0,0);wingShape.bezierCurveTo(.18,-.08,.68,-.48,.8,-1.08);wingShape.bezierCurveTo(.93,-1.67,.66,-1.85,.39,-1.67);wingShape.bezierCurveTo(.13,-1.43,.03,-.6,0,0);
  const wingGeometry=new THREE.ShapeGeometry(wingShape,36);wingGeometry.rotateX(Math.PI/2);
  for(const s of [1,-1]){
    const side=s===1?'L':'R';const wing=pivot(thorax,`${side}.wing.spread`,[s*.24,1.25,-.13],'y',[-1.48,1.48],'thorax');
    const flap=pivot(wing,`${side}.wing.flap`,[0,0,0],'z',[-1.3,1.3],'thorax');const shape=new THREE.Group();shape.scale.x=s;flap.add(shape);
    const wm=mat('#d4dfcc',.24,{transparent:true,opacity:.4,side:THREE.DoubleSide,metalness:.12,depthWrite:false});
    const mesh=new THREE.Mesh(wingGeometry,wm);mesh.name=`${side}_wing_membrane`;shape.add(mesh);membranes.push(mesh);
    function vein(points,r=.006){const curve=new THREE.CatmullRomCurve3(points.map(([x,z])=>new THREE.Vector3(x,.006,z)));const v=new THREE.Mesh(new THREE.TubeGeometry(curve,30,r,5,false),veinMat);v.name='Wing_vein';shape.add(v);membranes.push(v);}
    const outline=wingShape.getPoints(64).map(p=>[p.x,p.y]);vein(outline,.006);
    vein([[0,0],[.25,-.39],[.48,-.9],[.68,-1.66]]);vein([[0,0],[.17,-.55],[.27,-1.1],[.39,-1.67]]);vein([[.05,-.16],[.35,-.47],[.68,-.86],[.81,-1.22]]);vein([[.18,-.59],[.35,-.62],[.53,-.63]],.004);vein([[.28,-1.1],[.5,-1.08],[.79,-1.06]],.004);
    const h=pivot(thorax,`${side}.haltere`,[s*.32,1,-.42],'z',[-.5,.5],'thorax');rod(h,[0,0,0],[s*.22,.08,-.07],.019,.012,gold,'Haltere_stalk','thorax');ellipsoid(h,'Haltere_knob',[s*.22,.08,-.07],[.054,.047,.06],gold);
  }
  // Deterministic fine thoracic setae, with longer macrochaetae along the dorsal surface.
  for(let i=0;i<130;i++){const a=i*2.399963, y=.15+(i%17)/17*.8, r=Math.sqrt(1-y*y);const p=[.38*r*Math.cos(a),1.02+.36*y,.52*r*Math.sin(a)];const len=i%9===0?.13:.045;rod(thorax,p,[p[0]*(1+len*2),p[1]+len,p[2]*(1+len)],.003,.0006,dark,'Thoracic_seta','thorax');}
  function setPose(pose){
    for(const [name,value] of Object.entries(pose)){
      if(!joints[name])throw new Error(`Unknown joint: ${name}`);
      if(!Number.isFinite(value))throw new TypeError(`Non-finite angle: ${name}`);
    }
    for(const [name,value] of Object.entries(pose)){const [lo,hi]=limits[name];joints[name].node.rotation[joints[name].axis]=THREE.MathUtils.clamp(value,lo,hi);}
  }
  function resetPose(){for(const {node,axis} of Object.values(joints))node.rotation[axis]=0;}
  return {root,joints,limits,surfaces,membranes,markers,regions,legInfo,setPose,resetPose};
}

export function demonstrationPose(fly,mode,t,spread=25){
  const pose=Object.fromEntries(Object.keys(fly.joints).map(k=>[k,0]));
  for(const s of [1,-1]){const side=s===1?'L':'R';pose[`${side}.wing.spread`]=-s*spread*Math.PI/180;pose[`${side}.antenna`]=s*.06*Math.sin(t*2);if(mode==='flight'){pose[`${side}.wing.spread`]=-s*1.05;pose[`${side}.wing.flap`]=s*.95*Math.sin(t*18);pose[`${side}.haltere`]=s*.35*Math.sin(t*18+Math.PI);}}
  for(const {id,side,index} of fly.legInfo){const phase=t*5+((index+(side===1?0:1))%2)*Math.PI;if(mode==='walk'){pose[`${id}.coxa`]=side*.3*Math.sin(phase);pose[`${id}.femur`]=side*.23*Math.max(0,Math.cos(phase));pose[`${id}.tibia`]=-side*.3*Math.max(0,Math.cos(phase));}if(mode==='flight'){pose[`${id}.femur`]=-side*.5;pose[`${id}.tibia`]=side*.65;}if(mode==='groom'&&index===0){pose[`${id}.coxa`]=-side*.55;pose[`${id}.femur`]=side*(.5+.1*Math.sin(t*9));pose[`${id}.tibia`]=-side*.6;}}
  pose['head.pitch']=mode==='groom'?.15*Math.sin(t*4):.025*Math.sin(t);pose['abdomen.pitch']=mode==='flight'?-.12:0;return pose;
}
