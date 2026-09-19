// 3D fly stage for the dashboard: the Fly / Body Lab rig (fly.js) driven by the brain.
// Nothing here decides what the fly does. Every frame the continuous motor channels decoded
// from descending neurons (forward, backward, turn, groom, threat, song, land, freeze, jump)
// set the walking speed, turning rate and joint poses; the state label is only a caption.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createFly } from './fly.js';

const ARENA = 3.2;            // mm, half-size of the table
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
  const lens = new THREE.Group(); lens.position.set(0, .6, ARENA + 1.4);
  lens.add(new THREE.Mesh(new THREE.BoxGeometry(.9, .6, .3), new THREE.MeshStandardMaterial({ color: '#3a4360', roughness: .6 })));
  const glass = new THREE.Mesh(new THREE.CylinderGeometry(.18, .18, .12, 24), new THREE.MeshStandardMaterial({ color: '#0b0e14', roughness: .2, metalness: .6 }));
  glass.rotation.x = Math.PI / 2; glass.position.z = -.2; lens.add(glass); scene.add(lens);
  const objMarker = new THREE.Mesh(new THREE.SphereGeometry(.12, 16, 12), new THREE.MeshBasicMaterial({ color: '#4fd1ff', transparent: true, opacity: .8 }));
  objMarker.visible = false; scene.add(objMarker);

  const fly = createFly(); scene.add(fly.root);
  const camera = new THREE.PerspectiveCamera(36, 1, .01, 100); camera.position.set(4.5, 4.2, -5.0);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.minDistance = 2; controls.maxDistance = 16; controls.maxPolarAngle = Math.PI * .49;
  new ResizeObserver(() => { const { width, height } = host.getBoundingClientRect(); if (!width) return; renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix(); }).observe(host);

  // ---- state from the brain ---------------------------------------------------------------
  const M = { forward: 0, backward: 0, turn: 0, groom: 0, threat: 0, song: 0, land: 0, freeze: 0, jump: 0 };
  const S = { state: 'idle', asleep: false, objX: 0, objSeen: 0, t: 0, pos: new THREE.Vector3(0, 0, -.5), heading: 0, walkPhase: 0,
              jumpT: 0, jumpDir: 0, airborne: 0, songSide: 1, songT: 0, pose: {}, target: {} };
  for (const k of Object.keys(fly.joints)) { S.pose[k] = 0; S.target[k] = 0; }

  function onPacket(p) {
    S.state = p.state; S.asleep = !!p.asleep;
    if (p.motor) for (const k in M) if (k in p.motor) M[k] = p.motor[k];
    const s = p.sees; S.objSeen = s ? s.obj[2] : 0; S.objX = s ? s.obj[0] : 0;
    // the Giant Fiber jump is the one discrete event: fire it when the channel crosses threshold
    if (M.jump > 0.5 && S.jumpT <= 0 && !S.jumpCooldown) { S.jumpT = 0.9; S.jumpDir = wrapAngle(Math.PI + (S.objX || 0) * 0.9); S.jumpCooldown = true; }
    if (M.jump < 0.2) S.jumpCooldown = false;
  }

  // rig convention (see demonstrationPose in fly.js): wing spread is negative on L, positive on R
  function targetPose(t, speed) {
    const P = S.target; for (const k in P) P[k] = 0;
    const spread = (l, r = l) => { P['L.wing.spread'] = -l; P['R.wing.spread'] = r; };
    const sleep = S.asleep ? 1 : 0, air = S.airborne;
    const gait = Math.min(1, Math.abs(speed) / 1.2);       // leg swing amplitude follows walking speed
    spread(25 * Math.PI / 180);
    P['L.antenna'] = .06 * Math.sin(t * 2) * (1 - M.freeze); P['R.antenna'] = -P['L.antenna'];
    P['head.pitch'] = .025 * Math.sin(t) * (1 - M.freeze);
    if (S.objSeen > 0 && !sleep) P['head.yaw'] = THREE.MathUtils.clamp(-S.objX * .6, -.6, .6);
    for (const { id, side, index } of fly.legInfo) {
      const phase = S.walkPhase + ((index + (side === 1 ? 0 : 1)) % 2) * Math.PI;
      let coxa = side * .3 * Math.sin(phase) * gait, femur = side * .23 * Math.max(0, Math.cos(phase)) * gait, tibia = -side * .3 * Math.max(0, Math.cos(phase)) * gait, tarsus = 0;
      // blend in postures by channel strength
      const g = index === 0 ? M.groom : 0;
      coxa = lerp(coxa, -side * .55, g); femur = lerp(femur, side * (.5 + .12 * Math.sin(t * 11)), g); tibia = lerp(tibia, -side * .6, g);
      femur = lerp(femur, side * .35, M.land); tibia = lerp(tibia, -side * .5, M.land); tarsus = lerp(tarsus, .3, M.land);
      femur = lerp(femur, side * .3, sleep); tibia = lerp(tibia, -side * .45, sleep);
      femur = lerp(femur, side * .12, M.freeze * .6); tibia = lerp(tibia, -side * .15, M.freeze * .6);
      femur = lerp(femur, -side * .5, air); tibia = lerp(tibia, side * .65, air);        // legs tucked in flight
      P[`${id}.coxa`] = coxa; P[`${id}.femur`] = femur; P[`${id}.tibia`] = tibia; P[`${id}.tarsus`] = tarsus;
    }
    // wings: flight while airborne, threat display, one-wing song; blended by strength
    let sp = 25 * Math.PI / 180, flapL = 0, flapR = 0;
    if (air > 0) { const f = Math.sin(t * 55); sp = lerp(sp, 1.05, air); flapL = .95 * f * air; flapR = -.95 * f * air; P['L.haltere'] = .35 * Math.sin(t * 55 + Math.PI) * air; P['R.haltere'] = -P['L.haltere']; }
    sp = lerp(sp, 1.35 + .08 * Math.sin(t * 40), M.threat); flapL = lerp(flapL, .35, M.threat); flapR = lerp(flapR, -.35, M.threat);
    sp = lerp(sp, .6, M.land);
    spread(sp);
    if (M.song > 0.05) { const ext = lerp(sp, 1.4, M.song), vib = .18 * Math.sin(t * 70) * M.song; if (S.songSide > 0) { P['L.wing.spread'] = -ext; flapL += vib; } else { P['R.wing.spread'] = ext; flapR -= vib; } }
    P['L.wing.flap'] = flapL; P['R.wing.flap'] = flapR;
    P['abdomen.pitch'] = -.12 * air + .3 * M.threat + .2 * sleep;
    P['head.pitch'] += -.25 * M.threat - .35 * sleep - .1 * M.freeze + (.15 * Math.sin(t * 4) - .2) * M.groom;
    P['L.antenna'] += .3 * sleep; P['R.antenna'] -= .3 * sleep;
    P['proboscis.extension'] = .3 * sleep;
    return P;
  }

  function step(dt) {
    S.t += dt;
    // locomotion straight from the motor channels: no scripted wandering
    const still = Math.max(M.freeze, S.asleep ? 1 : 0, M.groom * .8, M.threat * .6, M.song * .5);
    let speed = (1.8 * M.forward - 1.2 * M.backward) * (1 - still);
    let turn = 3.0 * M.turn * (1 - still * .5);
    // keep the fly on the table: steer toward the middle when close to the edge
    const toCenter = Math.atan2(-S.pos.x, -S.pos.z);
    const edge = Math.max(Math.abs(S.pos.x), Math.abs(S.pos.z)) - (ARENA - .8);
    if (edge > 0 && Math.abs(speed) > .01) { const want = speed > 0 ? toCenter : wrapAngle(toCenter + Math.PI); turn += wrapAngle(want - S.heading) * (2 + 6 * edge); }
    S.heading = wrapAngle(S.heading + turn * dt);
    if (S.jumpT > 0) {
      S.jumpT -= dt; const u = 1 - Math.max(0, S.jumpT) / .9;
      S.airborne = Math.sin(Math.PI * Math.min(1, u * 1.15));
      S.pos.y = 1.6 * S.airborne;
      const dir = wrapAngle(S.heading + S.jumpDir);
      S.pos.x += Math.sin(dir) * 3.2 * dt; S.pos.z += Math.cos(dir) * 3.2 * dt;
      if (S.jumpT <= 0) { S.airborne = 0; S.pos.y = 0; S.heading = dir; }
    } else {
      S.pos.y = lerp(S.pos.y, S.asleep ? -.08 : 0, .1);
      S.pos.x += Math.sin(S.heading) * speed * dt; S.pos.z += Math.cos(S.heading) * speed * dt;
      S.walkPhase += speed * 4.5 * dt;
    }
    S.pos.x = THREE.MathUtils.clamp(S.pos.x, -ARENA, ARENA); S.pos.z = THREE.MathUtils.clamp(S.pos.z, -ARENA, ARENA);
    if (M.song > 0.05) { S.songT += dt; if (S.songT > 2.5) { S.songT = 0; S.songSide = -S.songSide; } }

    const T = targetPose(S.t, speed);
    for (const k in S.pose) { const fast = k.includes('flap') || k.includes('haltere'); S.pose[k] = lerp(S.pose[k], T[k], fast ? .7 : .22); }
    fly.setPose(S.pose);
    fly.root.position.copy(S.pos); fly.root.rotation.y = S.heading;

    objMarker.visible = S.objSeen > 0;
    if (objMarker.visible) { objMarker.position.set(-S.objX * (ARENA - .3), .5, ARENA + .8); objMarker.scale.setScalar(.6 + S.objSeen); }
    controls.target.lerp(new THREE.Vector3(S.pos.x, .8 + S.pos.y * .5, S.pos.z), .08);
    controls.update();
    renderer.render(scene, camera);
  }

  return { onPacket, step, fly, scene, motor: M };
}
