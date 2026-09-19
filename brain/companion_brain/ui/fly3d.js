// The fly's world: a table with fruit, a flower, a water drop, a second fly and a window onto the
// human world (the webcam). The Fly / Body Lab rig (fly.js) is moved only by the brain's motor
// channels. The fly SEES this world through its own eyes: a camera on its head renders a small
// retina image that is posted to the brain, which is what the optic lobe then processes. The
// world also reports smell (fruit odor per antenna), taste (sugar on contact), touch (pollen dust)
// and humidity (near the water drop), which the brain receives on the matching sensory neurons.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createFly, demonstrationPose } from './fly.js';

const ARENA = 11;             // mm, half-size of the table
const bounds = { x: ARENA, z: ARENA };   // walkable half-extents of the current world
const RET_W = 80, RET_H = 60; // retina resolution (matches the optic lobe)
const lerp = (a, b, k) => a + (b - a) * k;
const wrapAngle = a => Math.atan2(Math.sin(a), Math.cos(a));
const mat = (color, roughness = .7, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness, ...extra });

export function createFlyStage(host, retinaCanvas, opts = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.setClearColor(0x0d1018, 1); renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.2;
  host.appendChild(renderer.domElement);

  // ---- scene & world ------------------------------------------------------------------------
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#c9d3e0');          // daylight: the fly's eye wants dark things on a bright world
  scene.fog = new THREE.Fog('#c9d3e0', 30, 70);
  scene.add(new THREE.HemisphereLight('#ffffff', '#7a7466', 2.2));
  const key = new THREE.DirectionalLight('#fff5dc', 2.8); key.position.set(-8, 16, 10); key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048); Object.assign(key.shadow.camera, { left: -16, right: 16, top: 16, bottom: -16, near: .1, far: 60 }); key.shadow.normalBias = .03; scene.add(key);
  const fill = new THREE.DirectionalLight('#cfe0ff', 1.0); fill.position.set(10, 6, -10); scene.add(fill);
  const camCanvas = document.createElement('canvas'); camCanvas.width = 320; camCanvas.height = 240;
  const camCtx = camCanvas.getContext('2d'); camCtx.fillStyle = '#20242c'; camCtx.fillRect(0, 0, 320, 240);
  const camTex = new THREE.CanvasTexture(camCanvas); camTex.colorSpace = THREE.SRGBColorSpace;
  const winW = 16, winH = 12;
  const windowMesh = new THREE.Mesh(new THREE.PlaneGeometry(winW, winH), new THREE.MeshBasicMaterial({ color: '#ffffff', map: camTex }));
  let worldGroup = new THREE.Group(); scene.add(worldGroup);
  const world = { name: 'table', fruit: [], flowers: [], water: [], others: [], lemon: [], sugar: [] };
  function clearWorld() { scene.remove(worldGroup); worldGroup = new THREE.Group(); scene.add(worldGroup); for (const k of ['fruit', 'flowers', 'water', 'lemon', 'sugar']) world[k] = []; other.root.visible = false; }

  function addGrape(x, z) {
    const g = new THREE.Group(); g.position.set(x, 0, z);
    const body = new THREE.Mesh(new THREE.SphereGeometry(1.5, 32, 24), mat('#6b1f4a', .35, { metalness: .05 })); body.position.y = 1.4; body.castShadow = true; body.receiveShadow = true; g.add(body);
    const shine = new THREE.Mesh(new THREE.SphereGeometry(.25, 12, 8), mat('#ffd6ea', .2, { emissive: '#ffb6dc', emissiveIntensity: .35 })); shine.position.set(-.55, 2.35, .6); g.add(shine);
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(.08, .12, 1.1, 8), mat('#5a6a2a', .8)); stem.position.set(.3, 3.2, 0); stem.rotation.z = .5; g.add(stem);
    // a split in the skin: juice (the sugar is here)
    const juice = new THREE.Mesh(new THREE.CircleGeometry(.5, 20), mat('#e8a0c8', .3, { emissive: '#c05080', emissiveIntensity: .25 })); juice.rotation.x = -Math.PI / 2; juice.position.set(-.9, .02, 1.2); g.add(juice);
    worldGroup.add(g); world.fruit.push({ group: g, pos: new THREE.Vector3(x, 0, z), radius: 1.5, mouth: new THREE.Vector3(x - .9, 0, z + 1.2) });
  }
  function addLemon(x, z) {
    const g = new THREE.Group(); g.position.set(x, 0, z);
    const slice = new THREE.Mesh(new THREE.CylinderGeometry(1.5, 1.5, .5, 28), mat('#f4d03f', .5)); slice.position.y = .25; slice.castShadow = true; slice.receiveShadow = true; g.add(slice);
    const flesh = new THREE.Mesh(new THREE.CylinderGeometry(1.25, 1.25, .52, 28), mat('#fdf2a8', .7, { emissive: '#e8d070', emissiveIntensity: .15 })); flesh.position.y = .25; g.add(flesh);
    for (let i = 0; i < 8; i++) { const seg = new THREE.Mesh(new THREE.BoxGeometry(1.2, .02, .06), mat('#f4d03f', .6)); seg.position.set(0, .52, 0); seg.rotation.y = i * Math.PI / 8; seg.geometry.translate(0, 0, 0); seg.position.set(Math.cos(i * Math.PI / 8) * .6 * 0, .52, 0); g.add(seg); }
    worldGroup.add(g); world.lemon.push({ group: g, pos: new THREE.Vector3(x, 0, z), radius: 1.5 });
  }
  function addFlower(x, z) {
    const g = new THREE.Group(); g.position.set(x, 0, z);
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(.09, .14, 3.2, 8), mat('#3f7a2f', .8)); stem.position.y = 1.6; stem.castShadow = true; g.add(stem);
    const leaf = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), mat('#4c8f3a', .8)); leaf.scale.set(1.2, .08, .55); leaf.position.set(.9, .12, .4); leaf.rotation.y = .6; leaf.castShadow = true; g.add(leaf);
    const head = new THREE.Group(); head.position.y = 3.2; head.rotation.x = .35; g.add(head);
    for (let i = 0; i < 7; i++) { const p = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), mat('#f2f0ff', .6)); p.scale.set(.55, .08, 1.15); const a = i / 7 * Math.PI * 2; p.position.set(Math.sin(a) * 1.1, 0, Math.cos(a) * 1.1); p.rotation.y = a; p.castShadow = true; head.add(p); }
    const disc = new THREE.Mesh(new THREE.SphereGeometry(.6, 16, 12), mat('#e8b41c', .9, { emissive: '#c08a00', emissiveIntensity: .2 })); disc.scale.y = .45; head.add(disc);
    const pollen = new THREE.InstancedMesh(new THREE.SphereGeometry(.05, 6, 4), mat('#ffd54a', .9), 60); const d = new THREE.Object3D();
    for (let i = 0; i < 60; i++) { const a = Math.random() * Math.PI * 2, r = Math.random(); d.position.set(Math.sin(a) * r * .7, .2 + Math.random() * .12, Math.cos(a) * r * .7); d.updateMatrix(); pollen.setMatrixAt(i, d.matrix); } head.add(pollen);
    // pollen also falls on the table around the flower
    const dustRing = new THREE.Mesh(new THREE.RingGeometry(.4, 1.6, 32), mat('#c9a83a', 1, { transparent: true, opacity: .35 })); dustRing.rotation.x = -Math.PI / 2; dustRing.position.y = .005; g.add(dustRing);
    worldGroup.add(g); world.flowers.push({ group: g, pos: new THREE.Vector3(x, 0, z), radius: 2.2 });
  }
  function addWater(x, z) {
    const w = new THREE.Mesh(new THREE.SphereGeometry(1.3, 32, 20), new THREE.MeshPhysicalMaterial({ color: '#9fd3ff', roughness: .05, transmission: .85, thickness: 1, transparent: true, opacity: .7 }));
    w.scale.y = .45; w.position.set(x, .55, z); w.castShadow = true; worldGroup.add(w);
    world.water.push({ pos: new THREE.Vector3(x, 0, z), radius: 1.3 });
  }

  function buildTable() {
  world.name = 'table'; bounds.x = ARENA; bounds.z = ARENA;
  const table = new THREE.Mesh(new THREE.PlaneGeometry(2 * ARENA + 6, 2 * ARENA + 6), mat('#a08a6a', .95));
  table.rotation.x = -Math.PI / 2; table.receiveShadow = true; worldGroup.add(table);
  // wood grain lines
  for (let i = -ARENA - 3; i < ARENA + 3; i += 1.7) { const l = new THREE.Mesh(new THREE.PlaneGeometry(2 * ARENA + 6, .06), mat('#8a7458', .95)); l.rotation.x = -Math.PI / 2; l.position.set(0, .002, i + Math.sin(i) * .3); worldGroup.add(l); }
  const rim = new THREE.Mesh(new THREE.BoxGeometry(2 * ARENA + 6, .6, 2 * ARENA + 6), mat('#4a3c2c', .9)); rim.position.y = -.31; rim.receiveShadow = true; worldGroup.add(rim);

  // fruit: a grape at the far right, the fly's favourite smell (and sugar on contact)
  addGrape(6.5, 3.5);
  // the lemon: a second smell (different glomeruli) with no sugar. Present only in the two-odor experiment.
  if (opts.twoOdor !== false) addLemon(-6.5, -3.5);
  // flower with pollen: walking into it dusts the fly
  addFlower(-6.5, 2.5);
  // water drop: humid air around it
  addWater(-2.5, -6);
  // pebbles and a leaf for the eye to have edges to look at
  for (const [x, z, r] of [[3, -7, .5], [4.1, -6.4, .35], [-8, -5, .6], [8.5, -2, .4]]) { const p = new THREE.Mesh(new THREE.SphereGeometry(r, 14, 10), mat('#8d8a80', .9)); p.scale.y = .6; p.position.set(x, r * .6, z); p.castShadow = true; worldGroup.add(p); }
  { const leaf = new THREE.Mesh(new THREE.SphereGeometry(1, 18, 12), mat('#5f8a3e', .85)); leaf.scale.set(2.6, .05, 1.3); leaf.position.set(2, .05, 7.5); leaf.rotation.y = -.5; leaf.receiveShadow = true; worldGroup.add(leaf); }

  // the window onto the human world: the webcam plays on it
  windowMesh.position.set(0, winH / 2 + .6, ARENA + 3.5); windowMesh.rotation.y = Math.PI; worldGroup.add(windowMesh);
  const frame = new THREE.Mesh(new THREE.BoxGeometry(winW + 1, winH + 1, .4), mat('#2b3247', .7)); frame.position.set(0, winH / 2 + .6, ARENA + 3.7); worldGroup.add(frame);
  const sill = new THREE.Mesh(new THREE.BoxGeometry(winW + 1.6, .5, 1.2), mat('#3a4360', .7)); sill.position.set(0, .25, ARENA + 3.3); worldGroup.add(sill);
  other.root.visible = true; O.pos.set(-3, 0, 5.5);
  S.pos.set(0, 0, -2); S.heading = 0;
  }
  function setWebcam(img) { if (!img.complete || !img.naturalWidth) return; camCtx.drawImage(img, 0, 0, 320, 240); camTex.needsUpdate = true; }
  function closeWindow() {   // the human world is gone: draw drawn curtains on the window
    camCtx.fillStyle = '#3a3f55'; camCtx.fillRect(0, 0, 320, 240);
    for (let x = 0; x < 320; x += 20) { camCtx.fillStyle = (x / 20) % 2 ? '#454b66' : '#32374b'; camCtx.fillRect(x, 0, 20, 240); }
    camCtx.fillStyle = '#8a93a8'; camCtx.font = '14px sans-serif'; camCtx.fillText('webcam closed', 112, 124);
    camTex.needsUpdate = true;
  }

  // the other fly: part of the world, so it follows a simple script (wander, stop, groom)
  const other = createFly(); other.root.scale.setScalar(.92); scene.add(other.root);
  const O = { pos: new THREE.Vector3(-3, 0, 5.5), heading: 1.2, t: 0, mode: 'walk', modeT: 0, phase: Math.random() * 10 };
  function stepOther(dt) {
    O.t += dt; O.modeT -= dt;
    if (O.modeT <= 0) { O.mode = ['walk', 'walk', 'rest', 'groom'][Math.floor(Math.random() * 4)]; O.modeT = 1.5 + Math.random() * 3; O.turn = (Math.random() - .5) * 1.6; }
    if (O.mode === 'walk') { O.heading += O.turn * dt; const toC = Math.atan2(-O.pos.x, -O.pos.z); if (Math.max(Math.abs(O.pos.x), Math.abs(O.pos.z)) > ARENA - 1.5) O.heading += wrapAngle(toC - O.heading) * 3 * dt; O.pos.x += Math.sin(O.heading) * 1.6 * dt; O.pos.z += Math.cos(O.heading) * 1.6 * dt; }
    other.setPose(demonstrationPose(other, O.mode === 'walk' ? 'walk' : O.mode === 'groom' ? 'groom' : 'rest', O.t + O.phase, 25));
    other.root.position.copy(O.pos); other.root.rotation.y = O.heading;
  }

  // ---- our fly --------------------------------------------------------------------------------
  const fly = createFly(); scene.add(fly.root);
  const camera = new THREE.PerspectiveCamera(36, 1, .05, 200); camera.position.set(7, 7, -9);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.minDistance = 2; controls.maxDistance = 40; controls.maxPolarAngle = Math.PI * .49;
  new ResizeObserver(() => { const { width, height } = host.getBoundingClientRect(); if (!width) return; renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix(); }).observe(host);

  // the fly's own eyes: a wide camera on the head, rendered to a tiny retina
  const eyeCam = new THREE.PerspectiveCamera(95, RET_W / RET_H, .08, 80);
  const headNode = fly.joints['head.yaw'].node; headNode.add(eyeCam); eyeCam.position.set(0, .12, .55); eyeCam.rotation.y = Math.PI;   // rig: +Z anterior; camera looks down -Z
  const retinaRT = new THREE.WebGLRenderTarget(RET_W, RET_H, { depthBuffer: true });
  const retinaPixels = new Uint8Array(RET_W * RET_H * 4), retinaGray = new Uint8Array(RET_W * RET_H);
  const retinaCtx = retinaCanvas ? retinaCanvas.getContext('2d') : null;
  const retinaImg = retinaCtx ? retinaCtx.createImageData(RET_W, RET_H) : null;
  let lastRetina = 0, lastWorldPost = 0;

  const M = { forward: 0, backward: 0, turn: 0, groom: 0, threat: 0, song: 0, land: 0, freeze: 0, jump: 0, feed: 0 };
  const S = { state: 'idle', asleep: false, objX: 0, objSeen: 0, t: 0, pos: new THREE.Vector3(0, 0, -2), heading: 0, walkPhase: 0,
              flight: 0, flightT: 0, airborne: 0, alt: 0, landingT: 0, restUntil: 0, songSide: 1, songT: 0, pose: {}, target: {},
              // body state that lives in the world, not the brain
              satiety: 0.2, dust: 0, sugar: 0, odor: [0, 0], odorB: [0, 0], moist: 0, feeding: 0, selfMotion: 0, valence: 0 };
  for (const k of Object.keys(fly.joints)) { S.pose[k] = 0; S.target[k] = 0; }

  // ---- the second world: a two-odor choice arena ------------------------------------------
  // A plain chamber. Grape scent (odor A) comes from the left end, lemon scent (odor B) from the
  // right end, nothing else to see or smell. During training a sugar drop sits at the grape end.
  // The assay measures which half the fly spends its time in: PI = (t_A - t_B) / (t_A + t_B).
  const AX = 12, AZ = 4.5;
  const assay = { phase: 'idle', t: 0, sideA: 0, sideB: 0, fed: 0, naive: null, trained: null, log: [] };
  function buildArena() {
    world.name = 'arena'; bounds.x = AX; bounds.z = AZ;
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(2 * AX + 2, 2 * AZ + 2), mat('#d9d6cc', .95)); floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; worldGroup.add(floor);
    const wallMat = mat('#c7c3b8', .8, { transparent: true, opacity: .55 });
    for (const [w, h, x, z] of [[2 * AX + 2, 1.2, 0, -AZ - 1], [2 * AX + 2, 1.2, 0, AZ + 1]]) { const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, .2), wallMat); m.position.set(x, h / 2, z); worldGroup.add(m); }
    for (const x of [-AX - 1, AX + 1]) { const m = new THREE.Mesh(new THREE.BoxGeometry(.2, 1.2, 2 * AZ + 2), wallMat); m.position.set(x, .6, 0); worldGroup.add(m); }
    const mid = new THREE.Mesh(new THREE.PlaneGeometry(.08, 2 * AZ + 2), mat('#8a8578', 1)); mid.rotation.x = -Math.PI / 2; mid.position.y = .003; worldGroup.add(mid);
    // scent sources at the ends: a grape (no sugar on its own here) and a lemon, behind mesh screens
    addGrape(-AX - .2, 0); world.fruit[0].hasSugar = false; world.fruit[0].odorScale = 12;   // broad plumes: mid-arena is half strength
    addLemon(AX + .2, 0); world.lemon[0].odorScale = 12;
    // scent plumes drawn on the floor so you can see the gradient
    for (const [x0, col] of [[-AX, '#8a3a6a'], [AX, '#c9b24a']]) for (let i = 1; i <= 5; i++) { const ring = new THREE.Mesh(new THREE.RingGeometry(i * 2.2 - .05, i * 2.2, 48, 1, 0, Math.PI * 2), mat(col, 1, { transparent: true, opacity: .18 - i * .025 })); ring.rotation.x = -Math.PI / 2; ring.position.set(x0, .004, 0); worldGroup.add(ring); }
    // sugar: during training the whole grape half is coated with sugar water (as in a training tube)
    const drops = new THREE.Group(); drops.visible = false; worldGroup.add(drops);
    const dropMat = new THREE.MeshPhysicalMaterial({ color: '#ffe9a8', roughness: .1, transmission: .6, thickness: .5, transparent: true, opacity: .85 });
    for (let i = 0; i < 14; i++) { const d = new THREE.Mesh(new THREE.SphereGeometry(.35 + Math.random() * .3, 14, 10), dropMat); d.scale.y = .4; d.position.set(-AX + 1 + Math.random() * (AX - 3), .14, (Math.random() - .5) * 2 * (AZ - 1)); drops.add(d); }
    world.sugar.push({ mesh: drops, zone: { xmax: -5 } });
    S.pos.set(0, 0, 0); S.heading = Math.PI / 2 * (Math.random() < .5 ? 1 : -1);   // start in the middle, facing along the chamber
    windowMesh.visible = false;
  }
  function setWorld(name) {
    clearWorld(); windowMesh.visible = true; assay.phase = 'idle';
    if (name === 'arena') buildArena(); else buildTable();
    S.flight = 0; S.flightT = 0; S.landingT = 0; S.airborne = 0; S.alt = 0; S.pos.y = 0; S.dust = 0;
    controls.target.set(S.pos.x, .8, S.pos.z); camera.position.set(S.pos.x + 7, 7, S.pos.z - 9);
  }
  function startAssay() {
    if (world.name !== 'arena') setWorld('arena');
    Object.assign(assay, { phase: 'naive test', t: 0, sideA: 0, sideB: 0, fed: 0, naive: null, trained: null, log: [] });
    S.satiety = 0;                                    // a hungry fly learns
    S.pos.set(0, 0, 0); S.heading = Math.random() * Math.PI * 2;
  }
  const TEST_S = 60, TRAIN_S = 75, TRAIN_FED_S = 15;
  function stepAssay(dt) {
    if (world.name !== 'arena' || assay.phase === 'idle' || assay.phase === 'done') { if (world.sugar[0]) world.sugar[0].mesh.visible = false; return; }
    assay.t += dt;
    const drop = world.sugar[0];
    if (assay.phase === 'training') {
      drop.mesh.visible = true;
      if (S.pos.x > -5) S.pos.x = -5;                  // confined with the grape's scent and the sugar, as in a training tube
      if (S.flight > 0) { S.flight = 0; S.airborne = 0; S.alt = 0; S.pos.y = 0; }
      if (S.feeding > 0.2) assay.fed += dt;
      if (assay.fed >= TRAIN_FED_S || assay.t >= TRAIN_S) { assay.log.push(`training done: fed ${assay.fed.toFixed(1)} s of ${assay.t.toFixed(0)}`); assay.phase = 'test'; assay.t = 0; assay.sideA = assay.sideB = 0; S.pos.set(0, 0, 0); S.heading = Math.random() * Math.PI * 2; S.satiety = 0; }
    } else {                                          // a test: count time on each side
      drop.mesh.visible = false;
      if (S.pos.x < -1) assay.sideA += dt; else if (S.pos.x > 1) assay.sideB += dt;
      if (assay.t >= TEST_S) {
        const pi = (assay.sideA - assay.sideB) / Math.max(1e-6, assay.sideA + assay.sideB);
        if (assay.phase === 'naive test') { assay.naive = pi; assay.log.push(`naive PI ${pi.toFixed(2)}`); assay.phase = 'training'; assay.t = 0; assay.fed = 0; S.pos.set(-AX + 3, 0, 0); S.satiety = 0; }
        else { assay.trained = pi; assay.log.push(`trained PI ${pi.toFixed(2)}`); assay.phase = 'done'; }
      }
    }
  }

  function onPacket(p) {
    S.state = p.state; S.asleep = !!p.asleep; S.valence = p.valence || 0;
    if (p.motor) for (const k in M) if (k in p.motor) M[k] = p.motor[k];
    const s = p.sees; S.objSeen = s ? s.obj[2] : 0; S.objX = s ? s.obj[0] : 0;
    if (M.jump > 0.5 && S.flight <= 0 && !S.jumpCooldown && S.t > S.restUntil) takeoff();
    if (M.jump < 0.2) S.jumpCooldown = false;
  }
  function takeoff() { S.flight = 1; S.flightT = 0; S.jumpCooldown = true; S.jumpDir = wrapAngle(S.heading + Math.PI + (S.objX || 0) * .9); }

  // ---- senses computed by the world ---------------------------------------------------------
  const tmp = new THREE.Vector3();
  function senseWorld(dt) {
    const hunger = 1 - S.satiety;
    // antennae positions in world space (head is ~0.6 mm ahead of the root)
    const ant = side => { tmp.set(side * .12, 0, .75); tmp.applyAxisAngle(new THREE.Vector3(0, 1, 0), S.heading); return tmp.clone().add(S.pos); };
    const aL = ant(1), aR = ant(-1);
    let oL = 0, oR = 0, sugar = 0;
    for (const f of world.fruit) {
      const dL = aL.distanceTo(f.pos), dR = aR.distanceTo(f.pos);
      const sc = f.odorScale || 4; oL += 1 / (1 + Math.pow(dL / sc, 2)); oR += 1 / (1 + Math.pow(dR / sc, 2));
      if (f.hasSugar !== false && S.airborne <= 0 && S.pos.distanceTo(f.pos) < f.radius + 1.6) sugar = 1;   // at the grape's skin: juice
    }
    S.odor = [Math.min(1, oL * hunger), Math.min(1, oR * hunger)];
    let bL = 0, bR = 0;
    for (const l of world.lemon) { const sc = l.odorScale || 4; bL += 1 / (1 + Math.pow(aL.distanceTo(l.pos) / sc, 2)); bR += 1 / (1 + Math.pow(aR.distanceTo(l.pos) / sc, 2)); }
    S.odorB = [Math.min(1, bL * hunger), Math.min(1, bR * hunger)];
    for (const d of world.sugar) if (d.mesh.visible && S.airborne <= 0 && (d.zone ? S.pos.x <= d.zone.xmax : S.pos.distanceTo(d.pos) < d.radius + 1.2)) sugar = 1;
    S.sugar = sugar * hunger;
    // eating: proboscis out on sugar -> satiety rises, hunger falls
    S.feeding = sugar > 0 ? M.feed : 0;
    S.satiety = Math.min(1, S.satiety + S.feeding * dt * 0.06);
    S.satiety = Math.max(0, S.satiety - dt / 240);            // gets hungry again over ~4 minutes
    // pollen: walking through a flower dusts the fly; grooming cleans it
    for (const fl of world.flowers) if (S.airborne <= 0 && S.pos.distanceTo(fl.pos) < fl.radius) S.dust = Math.min(1, S.dust + dt * .6);
    S.dust = Math.max(0, S.dust - M.groom * dt * .25 - dt * .003);
    // humidity near the water
    S.moist = 0; for (const w of world.water) S.moist = Math.max(S.moist, 1 / (1 + Math.pow(S.pos.distanceTo(w.pos) / 2.5, 2)));
  }

  function postSenses(now) {
    if (now - lastRetina > 100) {           // ~10 Hz retina
      lastRetina = now;
      renderer.setRenderTarget(retinaRT); renderer.render(scene, eyeCam); renderer.setRenderTarget(null);
      renderer.readRenderTargetPixels(retinaRT, 0, 0, RET_W, RET_H, retinaPixels);
      for (let y = 0; y < RET_H; y++) for (let x = 0; x < RET_W; x++) {
        const src = ((RET_H - 1 - y) * RET_W + x) * 4, dst = y * RET_W + x;     // flip: GL origin is bottom-left
        retinaGray[dst] = (retinaPixels[src] * 77 + retinaPixels[src + 1] * 151 + retinaPixels[src + 2] * 28) >> 8;
        if (retinaImg) { const o = dst * 4; retinaImg.data[o] = retinaPixels[src]; retinaImg.data[o + 1] = retinaPixels[src + 1]; retinaImg.data[o + 2] = retinaPixels[src + 2]; retinaImg.data[o + 3] = 255; }
      }
      if (retinaCtx) retinaCtx.putImageData(retinaImg, 0, 0);
      const body = new Uint8Array(4 + RET_W * RET_H); body[0] = RET_W & 255; body[1] = RET_W >> 8; body[2] = RET_H & 255; body[3] = RET_H >> 8; body.set(retinaGray, 4);
      fetch('/retina', { method: 'POST', body }).catch(() => {});
      fetch('/world', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ odor: S.odor, odor_b: S.odorB, sugar: +S.sugar.toFixed(3), dust: +S.dust.toFixed(3), moist: +S.moist.toFixed(3),
                               satiety: +S.satiety.toFixed(3), airborne: S.airborne > 0, feeding: +S.feeding.toFixed(2),
                               self_motion: +S.selfMotion.toFixed(3), world: world.name,
                               assay: { phase: assay.phase, t: +assay.t.toFixed(1), sideA: +assay.sideA.toFixed(1), sideB: +assay.sideB.toFixed(1), fed: +assay.fed.toFixed(1), naive: assay.naive, trained: assay.trained } }) }).catch(() => {});
    }
  }

  // ---- posture from the motor channels ------------------------------------------------------
  function targetPose(t, speed) {
    const P = S.target; for (const k in P) P[k] = 0;
    const spread = (l, r = l) => { P['L.wing.spread'] = -l; P['R.wing.spread'] = r; };
    const sleep = S.asleep ? 1 : 0, air = S.airborne, landing = S.landingT > 0 ? 1 : 0;
    const gait = Math.min(1, Math.abs(speed) / 3.0);
    spread(25 * Math.PI / 180);
    P['L.antenna'] = .06 * Math.sin(t * 2) * (1 - M.freeze); P['R.antenna'] = -P['L.antenna'];
    P['head.pitch'] = .025 * Math.sin(t) * (1 - M.freeze);
    if (S.objSeen > 0 && !sleep) P['head.yaw'] = THREE.MathUtils.clamp(-S.objX * .6, -.6, .6);
    for (const { id, side, index } of fly.legInfo) {
      const phase = S.walkPhase + ((index + (side === 1 ? 0 : 1)) % 2) * Math.PI;
      let coxa = side * .3 * Math.sin(phase) * gait, femur = side * .23 * Math.max(0, Math.cos(phase)) * gait, tibia = -side * .3 * Math.max(0, Math.cos(phase)) * gait, tarsus = 0;
      const g = index === 0 ? M.groom : 0;
      coxa = lerp(coxa, -side * .55, g); femur = lerp(femur, side * (.5 + .12 * Math.sin(t * 11)), g); tibia = lerp(tibia, -side * .6, g);
      const ld = Math.max(M.land, landing);
      femur = lerp(femur, side * .35, ld); tibia = lerp(tibia, -side * .5, ld); tarsus = lerp(tarsus, .3, ld);
      femur = lerp(femur, side * .3, sleep); tibia = lerp(tibia, -side * .45, sleep);
      femur = lerp(femur, side * .12, M.freeze * .6); tibia = lerp(tibia, -side * .15, M.freeze * .6);
      femur = lerp(femur, -side * .5, air * (1 - landing)); tibia = lerp(tibia, side * .65, air * (1 - landing));
      P[`${id}.coxa`] = coxa; P[`${id}.femur`] = femur; P[`${id}.tibia`] = tibia; P[`${id}.tarsus`] = tarsus;
    }
    let sp = 25 * Math.PI / 180, flapL = 0, flapR = 0;
    if (air > 0) { const f = Math.sin(t * 55); sp = lerp(sp, 1.05, air); flapL = .95 * f * air; flapR = -.95 * f * air; P['L.haltere'] = .35 * Math.sin(t * 55 + Math.PI) * air; P['R.haltere'] = -P['L.haltere']; }
    sp = lerp(sp, 1.35 + .08 * Math.sin(t * 40), M.threat * (1 - air)); flapL = lerp(flapL, .35, M.threat * (1 - air)); flapR = lerp(flapR, -.35, M.threat * (1 - air));
    sp = lerp(sp, .6, M.land * (1 - air));
    spread(sp);
    if (M.song > 0.05 && air <= 0) { const ext = lerp(sp, 1.4, M.song), vib = .18 * Math.sin(t * 70) * M.song; if (S.songSide > 0) { P['L.wing.spread'] = -ext; flapL += vib; } else { P['R.wing.spread'] = ext; flapR -= vib; } }
    P['L.wing.flap'] = flapL; P['R.wing.flap'] = flapR;
    P['abdomen.pitch'] = -.12 * air + .3 * M.threat + .2 * sleep;
    P['head.pitch'] += -.25 * M.threat - .35 * sleep - .1 * M.freeze + (.15 * Math.sin(t * 4) - .2) * M.groom - .3 * S.feeding;
    P['L.antenna'] += .3 * sleep; P['R.antenna'] -= .3 * sleep;
    P['proboscis.extension'] = .3 * sleep + .5 * M.feed;   // proboscis motor neurons
    return P;
  }

  // ---- per frame ------------------------------------------------------------------------------
  function step(dt) {
    S.t += dt;
    if (other.root.visible) stepOther(dt);
    stepAssay(dt);
    senseWorld(dt);
    const still = Math.max(M.freeze, S.asleep ? 1 : 0, M.groom * .8, M.threat * .6, M.song * .5, S.feeding);
    let speed, turn;
    if (S.flight > 0) {
      // flight: takeoff hop, then cruise until the landing neurons fire (or the fly tires)
      S.flightT += dt;
      const tired = S.flightT > 5;
      if (S.landingT <= 0 && S.flightT > 1.2 && (M.land > 0.35 || tired)) S.landingT = 0.8;
      if (S.landingT > 0) { S.landingT -= dt; S.alt = lerp(S.alt, 0, dt * 4); if (S.landingT <= 0) { S.flight = 0; S.airborne = 0; S.alt = 0; S.pos.y = 0; S.restUntil = S.t + 2.5; } }
      else S.alt = lerp(S.alt, 2.6, dt * 3);
      S.airborne = S.flight > 0 ? Math.min(1, S.alt / 1.0) : 0;
      S.pos.y = S.alt;
      const dir = S.flightT < 0.6 ? S.jumpDir : S.heading;     // the escape hop goes away from the threat, then it steers
      if (S.flightT < 0.6) S.heading = lerp(S.heading, S.jumpDir, dt * 6);
      speed = S.landingT > 0 ? 2 : 7; turn = 2.2 * M.turn;
      S.pos.x += Math.sin(dir) * speed * dt; S.pos.z += Math.cos(dir) * speed * dt;
    } else {
      speed = (6.0 * M.forward - 3.0 * M.backward) * (1 - still);   // mm/s; a fly walks 10-20 mm/s at full tilt
      turn = 3.0 * M.turn * (1 - still * .5);
      // mushroom-body output biases steering: approach valence turns toward the stronger-smelling side,
      // avoidance turns away (MBON -> descending pathways; Aso et al. 2014b). Strength: learning.valence_steering
      const gradA = S.odor[1] - S.odor[0], gradB = S.odorB[1] - S.odorB[0];
      const grad = gradA + gradB;                      // right antenna minus left: +ve = smell is on the fly's right
      const vs = opts.valenceSteering ?? 1.0;
      turn += -vs * S.valence * grad * 6.0 * (1 - still);
      // klinotaxis on the learned value itself (Gomez-Marin & Louis): the mushroom-body valence rises toward
      // smells the fly has learned to like. Falling value -> turn (persistent random bias); rising -> run.
      S.smoothVal = (S.smoothVal ?? S.valence) + 0.3 * (S.valence - (S.smoothVal ?? S.valence));
      const dV = (S.smoothVal - (S.lastVal ?? S.smoothVal)) / Math.max(dt, 1e-3); S.lastVal = S.smoothVal;
      const trend = THREE.MathUtils.clamp(dV * 4, -1, 1);
      if (trend > 0.05) S.turnBias = Math.random() < .5 ? 1 : -1;
      turn += vs * Math.max(0, -trend) * 2.5 * (S.turnBias ?? 1) * (1 - still);
      speed += vs * (Math.max(0, trend) * 3.0 + Math.max(0, S.valence) * 1.5) * (1 - still);
      S.pos.y = lerp(S.pos.y, S.asleep ? -.08 : 0, .1);
      S.pos.x += Math.sin(S.heading) * speed * dt; S.pos.z += Math.cos(S.heading) * speed * dt;
      S.walkPhase += speed * 3.0 * dt;
    }
    // stay on the table: steer toward the middle near the edge (walking or flying)
    const toCenter = Math.atan2(-S.pos.x, -S.pos.z);
    const edge = Math.max(Math.abs(S.pos.x) - (bounds.x - 1.2), Math.abs(S.pos.z) - (bounds.z - 1.2));
    if (edge > 0 && Math.abs(speed) > .01) { const want = speed > 0 ? toCenter : wrapAngle(toCenter + Math.PI); turn += wrapAngle(want - S.heading) * (2 + 6 * edge); }
    S.heading = wrapAngle(S.heading + turn * dt);
    S.pos.x = THREE.MathUtils.clamp(S.pos.x, -bounds.x, bounds.x); S.pos.z = THREE.MathUtils.clamp(S.pos.z, -bounds.z, bounds.z);
    if (S.airborne <= 0) for (const o of [...world.fruit, ...world.water, ...world.lemon]) {   // solid things: stay outside them
      const dx = S.pos.x - o.pos.x, dz = S.pos.z - o.pos.z, d = Math.hypot(dx, dz), min = o.radius + .6;
      if (d < min && d > 1e-4) { S.pos.x = o.pos.x + dx / d * min; S.pos.z = o.pos.z + dz / d * min; }
    }
    // efference copy: how much of what the eyes see is the fly's own movement
    S.selfMotion = Math.min(1, Math.abs(speed) / 5.0 + Math.abs(turn) / 2.0 + (S.flight > 0 ? 1 : 0));
    if (M.song > 0.05) { S.songT += dt; if (S.songT > 2.5) { S.songT = 0; S.songSide = -S.songSide; } }

    const T = targetPose(S.t, S.flight > 0 ? 0 : speed);
    for (const k in S.pose) { const fast = k.includes('flap') || k.includes('haltere'); S.pose[k] = lerp(S.pose[k], T[k], fast ? .7 : .22); }
    fly.setPose(S.pose);
    fly.root.position.copy(S.pos); fly.root.rotation.y = S.heading;
    fly.root.rotation.x = S.flight > 0 ? -.15 * S.airborne : 0;

    controls.target.lerp(new THREE.Vector3(S.pos.x, .8 + S.pos.y * .5, S.pos.z), .08);
    controls.update();
    postSenses(performance.now());
    renderer.render(scene, camera);
  }

  setWorld(opts.world || 'table');
  return { onPacket, step, fly, scene, motor: M, state: S, setWebcam, closeWindow, setWorld, startAssay, assay, world };
}
