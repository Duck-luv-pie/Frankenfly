// The fly's world: a table with fruit, a flower, a water drop, a second fly and a window onto the
// human world (the webcam). The Fly / Body Lab rig (fly.js) is moved only by the brain's motor
// channels. The fly SEES this world through its own eyes: a camera on its head renders a small
// retina image that is posted to the brain, which is what the optic lobe then processes. The
// world also reports smell (fruit odor per antenna), taste (sugar on contact), touch (pollen dust)
// and humidity (near the water drop), which the brain receives on the matching sensory neurons.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { createFly, demonstrationPose } from './fly.js';

const ARENA = 11;             // mm, half-size of the table
const bounds = { x: ARENA, z: ARENA };   // walkable half-extents of the current world
let RET_W = 80, RET_H = 60;   // retina resolution (the optic lobe adapts to it; the hunt arena's wide eye uses 160x120)
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
  let retinaRT = new THREE.WebGLRenderTarget(RET_W, RET_H, { depthBuffer: true });
  let retinaPixels = new Uint8Array(RET_W * RET_H * 4), retinaGray = new Uint8Array(RET_W * RET_H);
  const retinaCtx = retinaCanvas ? retinaCanvas.getContext('2d') : null;
  let retinaImg = retinaCtx ? retinaCtx.createImageData(RET_W, RET_H) : null;
  function setRetina(w, h) {
    if (w === RET_W && h === RET_H) return;
    RET_W = w; RET_H = h; retinaRT.dispose(); retinaRT = new THREE.WebGLRenderTarget(w, h, { depthBuffer: true });
    retinaPixels = new Uint8Array(w * h * 4); retinaGray = new Uint8Array(w * h);
    if (retinaCanvas) { retinaCanvas.width = w; retinaCanvas.height = h; }
    retinaImg = retinaCtx ? retinaCtx.createImageData(w, h) : null;
    eyeCam.aspect = w / h; eyeCam.updateProjectionMatrix();
  }
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

  // ---- the third world: the hunt arena ----------------------------------------------------
  // A meter-scale room (the Fly / People Lab arena) with walking humans loaded from that project's
  // GLB exports. The fly is a 32 cm ground vehicle: forward / backward / turn only, no walking, no
  // flight. Touching a person -> sugar + reward dopamine; running out of time -> bitter + punishment.
  // Senses: two heat sensors (45° left / right cosine lobes) -> the arista's hot cells; vision is the
  // retina as in every world. Layouts come from the same seeded generator as sim/hunt_arena.py, so a
  // training seed shows the same people, patrols and start pose here. Physics mirrors that file.
  const HUNT = { arena: { x: 7.4, z: 6.1 }, people: 4, episode_s: 45, reward_s: 2, punish_s: 2, person_r: .2, nose_m: .16, touch_m: .06, frontal_deg: 30, side_reward: .5, freeze_brakes: false, speed_max: 1.6, speed_reverse: 0, turn_max: 2.5, turn_gain: 1, turn_sign: -1, heat_gain: 1, heat_scale_m: 3, heat_lobe_deg: 45, valence_steer: 6, fov_deg: 150 };
  const H = { phase: 'idle', t: 0, phaseT: 0, episode: 0, seed: 2048, touches: 0, frontal: 0, misses: 0, log: [], heat: [0, 0], nearest: 0, sugar: 0, bitter: 0, reward: 0, people: [] };
  const humanFiles = ['human-shorts.glb', 'human-jeans.glb', 'human-chinos.glb', 'human-joggers.glb'];
  const gltfLoader = new GLTFLoader();
  let mixers = [], huntFx = null;
  // seededRandom from the fly project's human.js (mulberry32), verbatim, so layouts match Python
  function mulberry32(seed) { let state = Number(seed) >>> 0; return () => { state += 0x6D2B79F5; let t = state; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function huntLayout(seed) {
    const rng = mulberry32(seed); for (let i = 0; i < 7; i++) rng();
    const n = Math.max(1, Math.min(8, HUNT.people | 0)), people = [];
    for (let i = 0; i < n; i++) { const p = { slot: i, anchorX: ((i % 4) - 1.5) * 3.25, anchorZ: Math.floor(i / 4) * 4 - 1.8, offset: rng() * Math.PI * 2, pace: .66 + rng() * .17, x: 0, z: 0, yaw: 0, speed: 0 }; rng(); people.push(p); }
    rng();
    return { people, rng };
  }
  function placePeople(t) { for (const p of H.people) { const th = t * p.pace + p.offset, ox = p.x, oz = p.z; p.x = Math.max(-7.1, Math.min(7.1, p.anchorX + 1.14 * Math.cos(th))); p.z = Math.max(-5.8, Math.min(5.8, p.anchorZ + 1.05 * Math.sin(th))); p.yaw = Math.atan2(-1.14 * Math.sin(th), 1.05 * Math.cos(th)); p.speed = Math.hypot(p.x - ox, p.z - oz); } }
  function huntBearing(px, pz) { const dx = px - S.pos.x, dz = pz - S.pos.z; const fwd = dx * Math.sin(S.heading) + dz * Math.cos(S.heading), left = dx * Math.cos(S.heading) - dz * Math.sin(S.heading); return [Math.hypot(dx, dz), Math.atan2(left, fwd)]; }
  function huntNearest() { let best = [1e9, 0]; for (const p of H.people) { const b = huntBearing(p.x, p.z); if (b[0] < best[0]) best = b; } return best; }
  function startEpisode(seed) {
    H.seed = seed >>> 0; const { people, rng } = huntLayout(H.seed);
    for (let i = 0; i < H.people.length && i < people.length; i++) Object.assign(H.people[i], { offset: people[i].offset, pace: people[i].pace });   // keep the models, new patrols
    H.t = 0; placePeople(0);
    for (let k = 0; k < 20; k++) { S.pos.set((rng() * 2 - 1) * (HUNT.arena.x - 1), 0, (rng() * 2 - 1) * (HUNT.arena.z - 1)); S.heading = rng() * Math.PI * 2 - Math.PI; if (huntNearest()[0] > 2) break; }
    H.phase = 'hunt'; H.phaseT = 0; H.sugar = 0; H.bitter = 0; H.reward = 0; H.episode++;
  }
  function buildHunt() {
    world.name = 'hunt'; bounds.x = HUNT.arena.x; bounds.z = HUNT.arena.z; mixers = [];
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(2 * HUNT.arena.x + 2, 2 * HUNT.arena.z + 2), mat('#d8d2c4', .95)); floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; worldGroup.add(floor);
    const grid = new THREE.GridHelper(16, 16, '#b9b2a4', '#c9c2b4'); grid.position.y = .003; grid.material.transparent = true; grid.material.opacity = .5; worldGroup.add(grid);
    const railMat = mat('#8a8578', .9);
    for (const side of [-1, 1]) { const rail = new THREE.Mesh(new THREE.BoxGeometry(.16, .22, 2 * HUNT.arena.z + 2), railMat); rail.position.set(side * (HUNT.arena.x + .6), .11, 0); rail.castShadow = true; worldGroup.add(rail); const back = new THREE.Mesh(new THREE.BoxGeometry(2 * HUNT.arena.x + 2, .22, .16), railMat); back.position.set(0, .11, side * (HUNT.arena.z + .6)); back.castShadow = true; worldGroup.add(back); }
    huntFx = new THREE.Mesh(new THREE.RingGeometry(.5, .6, 48), mat('#7fd68a', .5, { transparent: true, opacity: .8, side: THREE.DoubleSide, emissive: '#3c8a45', emissiveIntensity: .6 })); huntFx.rotation.x = -Math.PI / 2; huntFx.position.y = .01; huntFx.visible = false; worldGroup.add(huntFx);
    const { people } = huntLayout(H.seed);
    H.people = people.map(p => ({ ...p, model: null, mixer: null, glow: null }));
    for (const p of H.people) {
      const glow = new THREE.Mesh(new THREE.CircleGeometry(1.0, 40), mat('#ff9a4a', 1, { transparent: true, opacity: .22, emissive: '#ff6a00', emissiveIntensity: .5, depthWrite: false })); glow.rotation.x = -Math.PI / 2; glow.position.y = .006; worldGroup.add(glow); p.glow = glow;
      gltfLoader.load('./models/humans/' + humanFiles[p.slot % humanFiles.length], g => {
        if (world.name !== 'hunt' || !H.people.includes(p)) return;
        g.scene.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
        worldGroup.add(g.scene); p.model = g.scene;
        if (g.animations.length) { const mixer = new THREE.AnimationMixer(g.scene); mixer.clipAction(g.animations[0]).play(); p.mixer = mixer; mixers.push(mixer); }
      }, undefined, err => console.warn('human model failed to load', err));
    }
    windowMesh.visible = false;
    fly.root.scale.setScalar(.135); other.root.visible = false;
    eyeCam.far = 600;                                   // the head camera lives in the scaled rig: 600 x 0.135 = 81 m
    eyeCam.fov = HUNT.fov_deg; eyeCam.updateProjectionMatrix();   // the hunter's wide eye ...
    setRetina(160, 120);                                          // ... needs more pixels: a person 7 m away is still 2 px wide
    H.touches = 0; H.frontal = 0; H.misses = 0; H.episode = 0; H.log = [];
    startEpisode(H.seed);
  }
  function senseHunt() {
    S.odor = [0, 0]; S.odorB = [0, 0]; S.dust = 0; S.moist = 0; S.feeding = 0; S.satiety = 0;
    let hl = 0, hr = 0; const lobe = HUNT.heat_lobe_deg * Math.PI / 180;
    for (const p of H.people) { const [d, b] = huntBearing(p.x, p.z); const w = 1 / (1 + Math.pow(d / HUNT.heat_scale_m, 2)); hl += w * Math.max(0, Math.cos(b - lobe)); hr += w * Math.max(0, Math.cos(b + lobe)); }
    H.heat = [Math.min(1, hl * HUNT.heat_gain), Math.min(1, hr * HUNT.heat_gain)];
    H.nearest = huntNearest()[0];
    H.sugar = H.phase === 'reward' ? H.reward : 0; H.bitter = H.phase === 'punish' ? 1 : 0;
    S.sugar = H.sugar;
  }
  function stepHunt(dt) {
    if (H.phase !== 'idle') { H.t += dt; placePeople(H.t); }
    for (const p of H.people) { if (p.model) { p.model.position.set(p.x, 0, p.z); p.model.rotation.y = p.yaw; } if (p.mixer) p.mixer.timeScale = Math.min(1.6, p.speed / Math.max(dt, 1e-3) / .9); if (p.glow) p.glow.position.set(p.x, .006, p.z); }
    for (const m of mixers) m.update(dt);
    let speed = 0, turn = 0;
    if (H.phase === 'reward' || H.phase === 'punish') {
      H.phaseT += dt; huntFx.visible = true; huntFx.position.set(S.pos.x, .01, S.pos.z);
      const col = H.phase === 'punish' ? ['#e06060', '#8a2f2f'] : H.reward >= 1 ? ['#7fd68a', '#3c8a45'] : ['#e8d060', '#8a7a20'];   // green head-on, yellow glancing, red timeout
      huntFx.material.color.set(col[0]); huntFx.material.emissive.set(col[1]);
      if (H.phaseT >= (H.phase === 'reward' ? HUNT.reward_s : HUNT.punish_s)) { huntFx.visible = false; startEpisode(H.seed + 1); }
    } else if (H.phase === 'hunt') {
      const brake = HUNT.freeze_brakes ? 1 - M.freeze : 1;
      speed = (HUNT.speed_max * M.forward - HUNT.speed_reverse * M.backward) * brake;
      turn = HUNT.turn_sign * HUNT.turn_max * HUNT.turn_gain * M.turn;                     // heading rate, +ve = left
      // learned steering: mushroom-body valence x heat gradient (as the other worlds do with smell)
      const vs = opts.valenceSteering ?? 1.0, grad = H.heat[1] - H.heat[0];
      turn += -vs * S.valence * grad * HUNT.valence_steer;
      S.smoothVal = (S.smoothVal ?? S.valence) + 0.3 * (S.valence - (S.smoothVal ?? S.valence));
      const dV = (S.smoothVal - (S.lastVal ?? S.smoothVal)) / Math.max(dt, 1e-3); S.lastVal = S.smoothVal;
      const trend = THREE.MathUtils.clamp(dV * 4, -1, 1);
      if (trend > 0.05) S.turnBias = Math.random() < .5 ? 1 : -1;
      turn += vs * Math.max(0, -trend) * 2.5 * (S.turnBias ?? 1);
      speed += vs * (Math.max(0, trend) * .5 + Math.max(0, S.valence) * .3) * HUNT.speed_max * brake;
      speed = THREE.MathUtils.clamp(speed, -HUNT.speed_reverse, HUNT.speed_max);
      const toCenter = Math.atan2(-S.pos.x, -S.pos.z), edge = Math.max(Math.abs(S.pos.x) - (bounds.x - 1.2), Math.abs(S.pos.z) - (bounds.z - 1.2));
      if (edge > 0 && Math.abs(speed) > .01) { const want = speed > 0 ? toCenter : wrapAngle(toCenter + Math.PI); turn += wrapAngle(want - S.heading) * (2 + 6 * edge); }
      S.heading = wrapAngle(S.heading + turn * dt);
      let nx = THREE.MathUtils.clamp(S.pos.x + Math.sin(S.heading) * speed * dt, -bounds.x, bounds.x), nz = THREE.MathUtils.clamp(S.pos.z + Math.cos(S.heading) * speed * dt, -bounds.z, bounds.z);
      const solid = HUNT.person_r + .1;                                 // people are solid: the flank stops at their skin
      for (const p of H.people) { const d = Math.hypot(p.x - nx, p.z - nz); if (d < solid && d > 1e-6) { nx = p.x + (nx - p.x) / d * solid; nz = p.z + (nz - p.z) / d * solid; } }
      S.pos.x = nx; S.pos.z = nz;
      const d = huntNearest()[0];
      const noseX = S.pos.x + Math.sin(S.heading) * HUNT.nose_m, noseZ = S.pos.z + Math.cos(S.heading) * HUNT.nose_m;   // the nose must touch the skin
      let hit = null; for (const p of H.people) if (Math.hypot(p.x - noseX, p.z - noseZ) <= HUNT.person_r + HUNT.touch_m) { const b = huntBearing(p.x, p.z)[1]; if (hit === null || Math.abs(b) < Math.abs(hit)) hit = b; }
      if (hit !== null) { const frontal = Math.abs(hit) < HUNT.frontal_deg * Math.PI / 180; H.reward = frontal ? 1 : HUNT.side_reward; H.phase = 'reward'; H.phaseT = 0; H.touches++; if (frontal) H.frontal++; H.log.push(`episode ${H.episode}: ${frontal ? 'head-on' : 'glancing'} touch at ${H.t.toFixed(1)} s`); }
      else if (H.t >= HUNT.episode_s) { H.phase = 'punish'; H.phaseT = 0; H.misses++; H.log.push(`episode ${H.episode}: timeout, closest ${d.toFixed(2)} m`); }
    }
    S.selfMotion = Math.min(1, Math.abs(speed) / HUNT.speed_max);
    const T = targetPose(S.t, 0);
    for (const k in S.pose) S.pose[k] = lerp(S.pose[k], T[k], .22);
    fly.setPose(S.pose);
    fly.root.position.copy(S.pos); fly.root.rotation.y = S.heading;
    fly.root.rotation.z = -.15 * (turn / HUNT.turn_max);          // lean into the turn
    fly.root.rotation.x = -.05 * (speed / HUNT.speed_max);
    controls.target.lerp(new THREE.Vector3(S.pos.x, .3, S.pos.z), .08);
    controls.update();
    postSenses(performance.now());
    renderer.render(scene, camera);
  }
  function setHuntConfig(cfg) { if (cfg) for (const k in cfg) if (k in HUNT) HUNT[k] = (k === 'arena') ? { ...HUNT.arena, ...cfg.arena } : cfg[k]; }
  function setWorld(name) {
    clearWorld(); windowMesh.visible = true; assay.phase = 'idle'; H.phase = 'idle'; mixers = []; H.people = [];
    fly.root.scale.setScalar(1); fly.root.rotation.z = 0; eyeCam.far = 80; eyeCam.fov = 95; eyeCam.updateProjectionMatrix(); setRetina(80, 60);
    if (name === 'arena') buildArena(); else if (name === 'hunt') buildHunt(); else buildTable();
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
    if (M.jump > 0.5 && S.flight <= 0 && !S.jumpCooldown && S.t > S.restUntil && world.name !== 'hunt') takeoff();
    if (M.jump < 0.2) S.jumpCooldown = false;
  }
  function takeoff() { S.flight = 1; S.flightT = 0; S.jumpCooldown = true; S.jumpDir = wrapAngle(S.heading + Math.PI + (S.objX || 0) * .9); }

  // ---- senses computed by the world ---------------------------------------------------------
  const tmp = new THREE.Vector3();
  function senseWorld(dt) {
    if (world.name === 'hunt') { senseHunt(); return; }
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
                               heat: [+H.heat[0].toFixed(3), +H.heat[1].toFixed(3)], bitter: H.bitter,
                               hunt: { phase: H.phase, t: +H.t.toFixed(1), episode: H.episode, touches: H.touches, frontal: H.frontal, misses: H.misses, nearest: +H.nearest.toFixed(2), seed: H.seed },
                               assay: { phase: assay.phase, t: +assay.t.toFixed(1), sideA: +assay.sideA.toFixed(1), sideB: +assay.sideB.toFixed(1), fed: +assay.fed.toFixed(1), naive: assay.naive, trained: assay.trained } }) }).catch(() => {});
    }
  }

  // ---- posture from the motor channels ------------------------------------------------------
  function targetPose(t, speed) {
    const P = S.target; for (const k in P) P[k] = 0;
    const spread = (l, r = l) => { P['L.wing.spread'] = -l; P['R.wing.spread'] = r; };
    const sleep = S.asleep ? 1 : 0, air = S.airborne, landing = S.landingT > 0 ? 1 : 0, veh = world.name === 'hunt' ? 1 : 0;
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
      femur = lerp(femur, side * .45, veh); tibia = lerp(tibia, -side * .6, veh); tarsus = lerp(tarsus, .3, veh);   // the vehicle: legs tucked
      P[`${id}.coxa`] = coxa; P[`${id}.femur`] = femur; P[`${id}.tibia`] = tibia; P[`${id}.tarsus`] = tarsus;
    }
    let sp = 25 * Math.PI / 180, flapL = 0, flapR = 0;
    if (air > 0) { const f = Math.sin(t * 55); sp = lerp(sp, 1.05, air); flapL = .95 * f * air; flapR = -.95 * f * air; P['L.haltere'] = .35 * Math.sin(t * 55 + Math.PI) * air; P['R.haltere'] = -P['L.haltere']; }
    sp = lerp(sp, 1.35 + .08 * Math.sin(t * 40), M.threat * (1 - air)); flapL = lerp(flapL, .35, M.threat * (1 - air)); flapR = lerp(flapR, -.35, M.threat * (1 - air));
    sp = lerp(sp, .6, M.land * (1 - air));
    sp = lerp(sp, .12, veh);
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
    if (world.name === 'hunt') { stepHunt(dt); return; }
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
  return { onPacket, step, fly, scene, motor: M, state: S, setWebcam, closeWindow, setWorld, startAssay, assay, world, hunt: H, setHuntConfig };
}
