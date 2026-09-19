// 3D fly stage for the dashboard: the Fly / Body Lab rig (fly.js) driven by the brain's
// behavior state. The fly walks on a table; the webcam is "in front" of it (+Z).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createFly } from './fly.js';

const ARENA = 3.2;            // mm, half-size of the walking area
const lerp = (a, b, k) => a + (b - a) * k;
const wrapAngle = a => Math.atan2(Math.sin(a), Math.cos(a));

export function createFlyStage(host) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.setClearColor(0x0d1018, 1); renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.25;
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight('#f4f8ff', '#3a3a46', 1.8));
  const key = new THREE.DirectionalLight('#fff5dc', 3.0); key.position.set(-3, 7, 5); key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024); Object.assign(key.shadow.camera, { left: -5, right: 5, top: 5, bottom: -5, near: .1, far: 20 }); key.shadow.normalBias = .025; scene.add(key);
  const fill = new THREE.DirectionalLight('#cfe0ff', 1.2); fill.position.set(4, 3, -4); scene.add(fill);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ color: '#171c2a', roughness: .95 }));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
  const grid = new THREE.GridHelper(2 * ARENA + 1, 15, '#2a3350', '#1f2740'); grid.position.y = .003; scene.add(grid);
  // the camera the fly is looking at: a small lens in front (+Z)
  const lens = new THREE.Group(); lens.position.set(0, .6, ARENA + 1.4);
  lens.add(new THREE.Mesh(new THREE.BoxGeometry(.9, .6, .3), new THREE.MeshStandardMaterial({ color: '#3a4360', roughness: .6 })));
  const glass = new THREE.Mesh(new THREE.CylinderGeometry(.18, .18, .12, 24), new THREE.MeshStandardMaterial({ color: '#0b0e14', roughness: .2, metalness: .6 }));
  glass.rotation.x = Math.PI / 2; glass.position.z = -.2; lens.add(glass); scene.add(lens);
  const objMarker = new THREE.Mesh(new THREE.SphereGeometry(.12, 16, 12), new THREE.MeshBasicMaterial({ color: '#4fd1ff', transparent: true, opacity: .8 }));
  objMarker.visible = false; scene.add(objMarker);

  const fly = createFly(); scene.add(fly.root);
  fly.root.castShadow = true;
  const camera = new THREE.PerspectiveCamera(36, 1, .01, 100); camera.position.set(4.5, 4.2, -5.0);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.minDistance = 2; controls.maxDistance = 16; controls.maxPolarAngle = Math.PI * .49;
  new ResizeObserver(() => { const { width, height } = host.getBoundingClientRect(); if (!width) return; renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix(); }).observe(host);

  // ---- behavior state --------------------------------------------------------------------
  const S = { state: 'idle', prev: 'idle', objX: 0, objSeen: 0, lateral: 0, t: 0, pos: new THREE.Vector3(0, 0, -.5), heading: 0, walkPhase: 0,
              jumpT: 0, jumpDir: 0, airborne: 0, songSide: 1, songT: 0, wander: 0, pose: {}, target: {} };
  for (const k of Object.keys(fly.joints)) { S.pose[k] = 0; S.target[k] = 0; }

  function onPacket(p) {
    S.prev = S.state; S.state = p.state;
    S.lateral = (p.lateral && p.lateral.track) || 0;
    const s = p.sees; S.objSeen = s ? s.obj[2] : 0; S.objX = s ? s.obj[0] : 0;
    if (S.state === 'escape' && S.prev !== 'escape' && S.jumpT <= 0) { S.jumpT = 0.9; S.jumpDir = wrapAngle(Math.PI + (S.objX || 0) * 0.9); }
  }

  // heading convention: 0 = facing +Z (toward the webcam); positive = turning toward the fly's left (+X)
  function targetPose(st, t) {
    const P = S.target; for (const k in P) P[k] = 0;
    const walking = st === 'track' || st === 'idle' || st === 'backward' || st === 'social';
    // rig convention (see demonstrationPose in fly.js): spreading = negative on L, positive on R
    const spread = (l, r = l) => { P['L.wing.spread'] = -l; P['R.wing.spread'] = r; };
    spread(25 * Math.PI / 180);                                   // rest: wings folded back with a small spread
    P['L.antenna'] = .06 * Math.sin(t * 2); P['R.antenna'] = -.06 * Math.sin(t * 2);
    P['head.pitch'] = .025 * Math.sin(t);
    if (S.objSeen > 0 && st !== 'sleep') P['head.yaw'] = THREE.MathUtils.clamp(S.objX * -.6, -.6, .6);
    for (const { id, side, index } of fly.legInfo) {
      const phase = S.walkPhase + ((index + (side === 1 ? 0 : 1)) % 2) * Math.PI;
      if (walking) { P[`${id}.coxa`] = side * .3 * Math.sin(phase); P[`${id}.femur`] = side * .23 * Math.max(0, Math.cos(phase)); P[`${id}.tibia`] = -side * .3 * Math.max(0, Math.cos(phase)); }
      if (st === 'escape' && S.airborne > 0) { P[`${id}.femur`] = -side * .5; P[`${id}.tibia`] = side * .65; }
      if (st === 'land') { P[`${id}.femur`] = side * .35; P[`${id}.tibia`] = -side * .5; P[`${id}.tarsus`] = .3; }
      if (st === 'groom' && index === 0) { P[`${id}.coxa`] = -side * .55; P[`${id}.femur`] = side * (.5 + .12 * Math.sin(t * 11)); P[`${id}.tibia`] = -side * .6; }
      if (st === 'sleep') { P[`${id}.femur`] = side * .3; P[`${id}.tibia`] = -side * .45; }
      if (st === 'freeze') { P[`${id}.femur`] = side * .12; P[`${id}.tibia`] = -side * .15; }
    }
    if (st === 'escape') {
      const flap = S.airborne > 0 ? Math.sin(t * 55) : .35 * Math.sin(t * 30);
      spread(1.05); P['L.wing.flap'] = .95 * flap; P['R.wing.flap'] = -.95 * flap;
      P['L.haltere'] = .35 * Math.sin(t * 55 + Math.PI); P['R.haltere'] = -.35 * Math.sin(t * 55 + Math.PI); P['abdomen.pitch'] = -.12;
    } else if (st === 'threat') {
      const q = .08 * Math.sin(t * 40);
      spread(1.35 + q); P['L.wing.flap'] = .35; P['R.wing.flap'] = -.35;
      P['abdomen.pitch'] = .3; P['head.pitch'] = -.25;
    } else if (st === 'social') {
      const ext = 1.4, vib = .18 * Math.sin(t * 70);          // courtship song: one wing extended and vibrating
      if (S.songSide > 0) { P['L.wing.spread'] = -ext; P['L.wing.flap'] = vib; } else { P['R.wing.spread'] = ext; P['R.wing.flap'] = -vib; }
    } else if (st === 'land') {
      spread(.6);
    } else if (st === 'sleep') {
      P['head.pitch'] = -.35; P['abdomen.pitch'] = .2; P['L.antenna'] = .3; P['R.antenna'] = -.3; P['proboscis.extension'] = .3;
    } else if (st === 'freeze') {
      P['head.pitch'] = -.1; P['L.antenna'] = 0; P['R.antenna'] = 0;
    } else if (st === 'groom') {
      P['head.pitch'] = .15 * Math.sin(t * 4) - .2;
    }
    return P;
  }

  function step(dt) {
    S.t += dt;
    const st = S.state;
    let speed = 0, turn = 0;
    if (st === 'track') { speed = 1.6; turn = (S.objSeen ? -S.objX : -S.lateral) * 2.4; }
    else if (st === 'social') { speed = .5; turn = -S.objX * 1.5; }
    else if (st === 'backward') { speed = -1.0; turn = .5 * Math.sin(S.t * 3); }
    else if (st === 'idle') { S.wander += (Math.random() - .5) * dt * 3; S.wander *= .985; turn = S.wander; speed = .35 + .15 * Math.sin(S.t * .7); }
    else if (st === 'land') { speed = .1; }
    // steer back toward the middle near the edges
    const toCenter = Math.atan2(-S.pos.x, -S.pos.z);   // heading that points at the origin
    const edge = Math.max(Math.abs(S.pos.x), Math.abs(S.pos.z)) - (ARENA - .8);
    if (edge > 0 && speed !== 0) { const want = speed > 0 ? toCenter : wrapAngle(toCenter + Math.PI); turn += wrapAngle(want - S.heading) * (2 + 6 * edge); }
    S.heading = wrapAngle(S.heading + turn * dt);
    if (S.jumpT > 0) {
      S.jumpT -= dt; const u = 1 - Math.max(0, S.jumpT) / .9;
      S.airborne = Math.sin(Math.PI * Math.min(1, u * 1.15));
      S.pos.y = 1.6 * S.airborne;
      const dir = wrapAngle(S.heading + S.jumpDir);
      S.pos.x += Math.sin(dir) * 3.2 * dt; S.pos.z += Math.cos(dir) * 3.2 * dt;
      if (S.jumpT <= 0) { S.airborne = 0; S.pos.y = 0; S.heading = dir; }
    } else {
      S.pos.y = lerp(S.pos.y, st === 'sleep' ? -.08 : 0, .1);
      S.pos.x += Math.sin(S.heading) * speed * dt; S.pos.z += Math.cos(S.heading) * speed * dt;
      S.walkPhase += speed * 4.5 * dt;
    }
    S.pos.x = THREE.MathUtils.clamp(S.pos.x, -ARENA, ARENA); S.pos.z = THREE.MathUtils.clamp(S.pos.z, -ARENA, ARENA);
    if (st === 'social') { S.songT += dt; if (S.songT > 2.5) { S.songT = 0; S.songSide = -S.songSide; } }

    // pose: ease toward the target pose (fast for flapping wings, slower for the rest)
    const T = targetPose(st, S.t);
    for (const k in S.pose) { const fast = k.includes('flap') || k.includes('haltere'); S.pose[k] = lerp(S.pose[k], T[k], fast ? .7 : .22); }
    fly.setPose(S.pose);
    fly.root.position.copy(S.pos); fly.root.rotation.y = S.heading;

    objMarker.visible = S.objSeen > 0;
    if (objMarker.visible) { objMarker.position.set(-S.objX * (ARENA - .3), .5, ARENA + .8); objMarker.scale.setScalar(.6 + S.objSeen); }
    controls.target.lerp(new THREE.Vector3(S.pos.x, .8 + S.pos.y * .5, S.pos.z), .08);
    controls.update();
    renderer.render(scene, camera);
  }

  return { onPacket, step, fly, scene };
}
