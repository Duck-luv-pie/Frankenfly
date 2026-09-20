// MaleCNS neuron atlas (from the Fly / Neural Atlas: real neuron skeletons and cell-body
// positions, HHMI Janelia / Google Research, CC BY 4.0), driven live by our simulation.
// Rendering follows the atlas viewer (neural.js): every skeleton vertex and every cell body
// reads its pulse from a 512x512 activity texture indexed by catalogue slot.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CATEGORIES, PALETTE, ActivityState, parseBundle } from './atlas-data.js';

const vertexShader = `
  attribute float cell; attribute float category; attribute vec3 tint;
  uniform sampler2D uActivity; uniform float uTime; uniform float uDecay;
  uniform float uSelected; uniform float uBrightness; uniform vec4 uClasses; uniform float uPixelRatio;
  varying vec3 vColor; varying float vAlpha;
  void main(){
    vec2 uv=(vec2(mod(cell,512.0),floor(cell/512.0))+.5)/512.0;
    vec4 state=texture2D(uActivity,uv);
    float pulse=state.g*exp(-max(0.0,uTime-state.r)/uDecay);
    float selected=1.0-step(.1,abs(cell-uSelected));
    float enabled=category<.5?uClasses.x:category<1.5?uClasses.y:category<2.5?uClasses.z:uClasses.w;
    float visible=max(enabled,selected);
    vColor=mix(tint,vec3(.87,1.0,.94),min(1.0,pulse*.95));
    #ifdef SOMA
      vAlpha=(.05+selected*.2+pulse*.55)*visible;
      gl_PointSize=(1.3+selected*1.5+pulse*3.0)*uPixelRatio;
    #else
      vAlpha=(.004+uBrightness*.045+selected*.10+pulse*.6)*visible;
      gl_PointSize=1.0;
    #endif
    gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);
  }`;
const fragmentShader = `varying vec3 vColor; varying float vAlpha;
  void main(){
    float alpha=vAlpha;
    #ifdef SOMA
      float d=length(gl_PointCoord-.5)*2.0;if(d>1.0)discard;alpha*=1.0-d*d;
    #endif
    if(alpha<.001)discard;gl_FragColor=vec4(vColor,alpha);
  }`;

export function createAtlas(host, opts = {}) {
  const onStatus = opts.onStatus || (() => {});
  const onSelect = opts.onSelect || (() => {});
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.7)); renderer.setClearColor(0x05070b, 1); renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(43, 1, .01, 200);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.minDistance = 1.2; controls.maxDistance = 40;
  controls.target.set(0, -.45, 0); camera.position.set(0, -.45, 10.5);
  new ResizeObserver(() => { const { width, height } = host.getBoundingClientRect(); if (!width || !height) return; renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix(); }).observe(host);
  const root = new THREE.Group(); scene.add(root);
  const clock = new THREE.Clock();

  const activity = new ActivityState();
  const texture = new THREE.DataTexture(activity.data, activity.size, activity.size, THREE.RGBAFormat, THREE.FloatType);
  texture.minFilter = texture.magFilter = THREE.NearestFilter; texture.needsUpdate = true;
  const uniforms = { uActivity: { value: texture }, uTime: { value: 0 }, uDecay: { value: .3 }, uSelected: { value: -1 }, uBrightness: { value: .4 }, uClasses: { value: new THREE.Vector4(1, 1, 1, 1) }, uPixelRatio: { value: Math.min(devicePixelRatio, 1.7) } };
  const lineMaterial = new THREE.ShaderMaterial({ uniforms, vertexShader, fragmentShader, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending });
  const somaMaterial = new THREE.ShaderMaterial({ uniforms, vertexShader, fragmentShader, defines: { SOMA: 1 }, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending });
  const palette = PALETTE.map(c => new THREE.Color(c));
  const catalog = new Map(); let catalogRows = [], manifest, center = [0, 0, 0], somas, somaRecords = [], loaded = 0;
  const worldScale = .008;
  const transformed = (out, o, x, y, z) => { out[o] = (x - center[0]) * worldScale; out[o + 1] = (center[2] - z) * worldScale; out[o + 2] = (y - center[1]) * worldScale; };
  const colorFor = r => { const c = palette[r.region]; const v = .72 + (Number(r.id) % 41) / 100; return [Math.min(255, c.r * 255 * v), Math.min(255, c.g * 255 * v), Math.min(255, c.b * 255 * v)]; };

  async function fetchGz(path) {
    const res = await fetch(path); if (!res.ok) throw new Error(`${path}: ${res.status}`);
    return new Response(res.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  }
  function addSkeletons(skeletons) {
    skeletons = skeletons.filter(s => catalog.has(s.id)); if (!skeletons.length) return;
    const nv = skeletons.reduce((n, s) => n + s.positions.length / 3, 0), ne = skeletons.reduce((n, s) => n + s.edges.length, 0);
    const positions = new Float32Array(nv * 3), cells = new Float32Array(nv), categories = new Float32Array(nv), colors = new Uint8Array(nv * 3), edges = new Uint32Array(ne);
    let vo = 0, eo = 0;
    for (const sk of skeletons) {
      const r = catalog.get(sk.id), count = sk.positions.length / 3, color = colorFor(r);
      for (let i = 0; i < count; i++) { const a = i * 3, b = (vo + i) * 3; transformed(positions, b, sk.positions[a], sk.positions[a + 1], sk.positions[a + 2]); colors.set(color, b); }
      cells.fill(r.slot, vo, vo + count); categories.fill(r.region, vo, vo + count);
      for (let i = 0; i < sk.edges.length; i++) edges[eo + i] = sk.edges[i] + vo;
      vo += count; eo += sk.edges.length; loaded++;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3)); g.setAttribute('cell', new THREE.BufferAttribute(cells, 1));
    g.setAttribute('category', new THREE.BufferAttribute(categories, 1)); g.setAttribute('tint', new THREE.BufferAttribute(colors, 3, true));
    g.setIndex(new THREE.BufferAttribute(edges, 1)); root.add(new THREE.LineSegments(g, lineMaterial));
  }
  function addSomas() {
    somaRecords = catalogRows.filter(r => r.soma);
    const n = somaRecords.length, positions = new Float32Array(n * 3), cells = new Float32Array(n), categories = new Float32Array(n), colors = new Uint8Array(n * 3);
    somaRecords.forEach((r, i) => { transformed(positions, i * 3, ...r.soma); cells[i] = r.slot; categories[i] = r.region; colors.set(colorFor(r), i * 3); });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3)); g.setAttribute('cell', new THREE.BufferAttribute(cells, 1));
    g.setAttribute('category', new THREE.BufferAttribute(categories, 1)); g.setAttribute('tint', new THREE.BufferAttribute(colors, 3, true));
    somas = new THREE.Points(g, somaMaterial); root.add(somas);
  }

  // click a cell body to name it
  const raycaster = new THREE.Raycaster(); raycaster.params.Points.threshold = .03; let pointerStart;
  renderer.domElement.addEventListener('pointerdown', e => { pointerStart = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', e => {
    if (!somas || !pointerStart || Math.hypot(e.clientX - pointerStart[0], e.clientY - pointerStart[1]) > 4) return;
    const rect = renderer.domElement.getBoundingClientRect();
    raycaster.setFromCamera(new THREE.Vector2((e.clientX - rect.left) / rect.width * 2 - 1, -(e.clientY - rect.top) / rect.height * 2 + 1), camera);
    const hit = raycaster.intersectObject(somas)[0];
    if (hit) { const r = somaRecords[hit.index]; uniforms.uSelected.value = r.slot; onSelect(r); }
  });

  const api = {
    ready: false, catalog, stats: () => ({ loaded, somas: somaRecords.length, catalog: catalog.size }),
    fire(slots, amp = .45) { const now = clock.elapsedTime; for (const s of slots) if (s >= 0 && s < 512 * 512) activity.fire(s, now, amp); texture.needsUpdate = true; },
    preset(name) { controls.target.set(0, -.45, 0); camera.position.set(...({ front: [0, -.45, 10.5], side: [10.5, -.45, 0], top: [0, 10, .01] }[name] || [0, -.45, 14.2])); camera.up.set(0, 1, 0); controls.update(); },
    setBrightness(v) { uniforms.uBrightness.value = v; },
    setClass(i, on) { uniforms.uClasses.value.setComponent(i, on ? 1 : 0); },
    record(slot) { return catalogRows[slot]; },
    step() { uniforms.uTime.value = clock.getElapsedTime(); controls.update(); renderer.render(scene, camera); },
  };

  (async () => {
    try {
      onStatus('reading the MaleCNS catalogue…');
      manifest = await (await fetch('/atlas/manifest.json')).json();
      center = manifest.bounds[0].map((v, i) => (v + manifest.bounds[1][i]) / 2);
      const rows = JSON.parse(new TextDecoder().decode(await fetchGz('/atlas/catalog.json.gz')));
      catalogRows = rows.map((row, slot) => ({ id: row[0], type: row[1], instance: row[2], superclass: row[3], side: row[4], region: row[5], soma: row[6], slot }));
      for (const r of catalogRows) catalog.set(r.id, r);
      addSomas();
      onStatus(`${somaRecords.length.toLocaleString()} cell bodies · loading skeletons…`);
      for (let i = 0; i < manifest.chunks.length; i += 2) {
        const batch = await Promise.all(manifest.chunks.slice(i, i + 2).map(async name => parseBundle(await fetchGz('/atlas/' + name))));
        for (const sk of batch) addSkeletons(sk);
        onStatus(`${loaded.toLocaleString()} real neuron skeletons · ${somaRecords.length.toLocaleString()} cell bodies`);
        await new Promise(r => requestAnimationFrame(r));
      }
      api.ready = true;
      onStatus(`${loaded.toLocaleString()} skeletons · ${somaRecords.length.toLocaleString()} cell bodies · ${manifest.dataset}`);
    } catch (e) { console.error(e); onStatus('atlas unavailable: ' + e.message); }
  })();
  return api;
}
