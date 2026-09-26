// Interactive 3D POS terminal for the home page hero.
// Build: `npm run build:3d` → website/assets/js/hero-3d.js
//
// Interactions: drag to rotate (with inertia), slow auto-rotate when idle,
// subtle tilt toward the mouse, receipt feeds out of the printer on load, tap the device to
// tear the receipt off and print a new one.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const MODEL_URL = 'assets/models/dynamo-pos.glb';
const FALLBACK_IMG = 'assets/img/835cd9a5.webp';

const ACCENT = 0xa6d57f;
const START_YAW = -0.45;
const START_PITCH = 0.12;
const AUTO_SPEED = 0.32; // rad/s, ~18°/s
const IDLE_BEFORE_AUTO = 2.5; // s
const PRINT_TIME = 2.2; // s
const TEAR_TIME = 0.6; // s


const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const canHover = matchMedia('(hover: hover) and (pointer: fine)').matches;

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const damp = (cur, target, lambda, dt) => cur + (target - cur) * (1 - Math.exp(-lambda * dt));
const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

// Receipt feed. The paper is one curled sheet rising out of the printer slot.
// We trace its centre line (in the model's y/z plane), give every vertex a
// distance along that line, then slide vertices back along the curve to
// "un-print" it. Vertices pushed past the slot continue straight into the
// printer and are hidden by a clip plane across the slot, so the torn top edge
// and the printed text travel out together, like real thermal paper.
function buildFeed(receipt, model) {
  model.updateMatrixWorld(true);
  const toModel = new THREE.Matrix4().copy(model.matrixWorld).invert();
  const v = new THREE.Vector3();
  const parts = [];
  const ys = [];
  const zs = [];

  receipt.traverse((o) => {
    if (!o.isMesh) return;
    const rel = new THREE.Matrix4().multiplyMatrices(toModel, o.matrixWorld);
    const relN = new THREE.Matrix3().getNormalMatrix(rel);
    const inv = rel.clone().invert();
    const invN = new THREE.Matrix3().getNormalMatrix(inv);
    const g = o.geometry;
    const src = g.attributes.position;
    const srcN = g.attributes.normal;
    const n = src.count;
    const base = new Float32Array(n * 3);
    const baseN = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      v.fromBufferAttribute(src, i).applyMatrix4(rel);
      base[i * 3] = v.x; base[i * 3 + 1] = v.y; base[i * 3 + 2] = v.z;
      ys.push(v.y); zs.push(v.z);
      if (srcN) {
        v.fromBufferAttribute(srcN, i).applyMatrix3(relN).normalize();
        baseN[i * 3] = v.x; baseN[i * 3 + 1] = v.y; baseN[i * 3 + 2] = v.z;
      }
    }
    // Quantized attributes can't hold positions outside the original bounds,
    // so switch to plain floats we can rewrite every frame.
    const pos = new THREE.BufferAttribute(new Float32Array(n * 3), 3).setUsage(THREE.DynamicDrawUsage);
    const nrm = new THREE.BufferAttribute(new Float32Array(n * 3), 3).setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', pos);
    if (srcN) g.setAttribute('normal', nrm);
    o.frustumCulled = false;
    parts.push({ g, pos, nrm: srcN ? nrm : null, base, baseN, inv, invN, s: new Float32Array(n), off: new Float32Array(n), th: new Float32Array(n) });
  });

  // Centre line: average z per height band (the curl rises monotonically).
  const BINS = 40;
  let yMin = Infinity, yMax = -Infinity;
  for (const y of ys) { if (y < yMin) yMin = y; if (y > yMax) yMax = y; }
  const sumZ = new Float64Array(BINS), cnt = new Uint32Array(BINS), sumY = new Float64Array(BINS);
  for (let i = 0; i < ys.length; i++) {
    const b = Math.min(BINS - 1, Math.floor(((ys[i] - yMin) / (yMax - yMin)) * BINS));
    sumZ[b] += zs[i]; sumY[b] += ys[i]; cnt[b]++;
  }
  let path = [];
  for (let b = 0; b < BINS; b++) if (cnt[b]) path.push([sumY[b] / cnt[b], sumZ[b] / cnt[b]]);
  // Light smoothing so shading along the sheet stays even.
  path = path.map((p, i, a) => (i === 0 || i === a.length - 1 ? p : [(a[i - 1][0] + 2 * p[0] + a[i + 1][0]) / 4, (a[i - 1][1] + 2 * p[1] + a[i + 1][1]) / 4]));

  const P = path.length;
  const S = new Float64Array(P);   // arc length at each point
  const TH = new Float64Array(P);  // tangent angle at each point, atan2(dz, dy)
  for (let i = 1; i < P; i++) S[i] = S[i - 1] + Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
  for (let i = 0; i < P; i++) {
    const a = path[Math.max(0, i - 1)], b = path[Math.min(P - 1, i + 1)];
    TH[i] = Math.atan2(b[1] - a[1], b[0] - a[0]);
  }
  const length = S[P - 1];

  // Point on the curve at arc length s (straight extension beyond both ends).
  const at = (s, out) => {
    let i0, i1, t;
    if (s <= 0) { i0 = 0; t = s; out.th = TH[0]; }
    else if (s >= length) { i0 = P - 1; t = s - length; out.th = TH[P - 1]; }
    else {
      i1 = 1;
      while (S[i1] < s) i1++;
      i0 = i1 - 1;
      const f = (s - S[i0]) / (S[i1] - S[i0]);
      out.y = path[i0][0] + (path[i1][0] - path[i0][0]) * f;
      out.z = path[i0][1] + (path[i1][1] - path[i0][1]) * f;
      out.th = TH[i0] + (TH[i1] - TH[i0]) * f;
      return out;
    }
    out.y = path[i0][0] + Math.cos(out.th) * t;
    out.z = path[i0][1] + Math.sin(out.th) * t;
    return out;
  };

  // Project each vertex onto the curve: arc length + signed offset from it.
  const q = { y: 0, z: 0, th: 0 };
  for (const part of parts) {
    const n = part.s.length;
    for (let i = 0; i < n; i++) {
      const y = part.base[i * 3 + 1], z = part.base[i * 3 + 2];
      let best = Infinity, bs = 0, bo = 0;
      for (let k = 0; k < P - 1; k++) {
        const ay = path[k][0], az = path[k][1];
        const dy = path[k + 1][0] - ay, dz = path[k + 1][1] - az;
        const len2 = dy * dy + dz * dz;
        let f = ((y - ay) * dy + (z - az) * dz) / len2;
        if (k > 0) f = Math.max(0, f);
        if (k < P - 2) f = Math.min(1, f);
        const py = ay + dy * f, pz = az + dz * f;
        const d2 = (y - py) ** 2 + (z - pz) ** 2;
        if (d2 < best) {
          best = d2;
          bs = S[k] + f * Math.sqrt(len2);
          const len = Math.sqrt(len2);
          bo = ((z - pz) * dy - (y - py) * dz) / len; // offset along the curve normal
        }
      }
      part.s[i] = bs;
      part.off[i] = bo;
      part.th[i] = at(bs, q).th;
    }
  }

  // Clip plane across the slot: keep only paper that has come out.
  const t0y = Math.cos(TH[0]), t0z = Math.sin(TH[0]);
  const plane = new THREE.Plane(new THREE.Vector3(0, t0y, t0z), -(t0y * path[0][0] + t0z * path[0][1]) + 0.0004);

  const m = new THREE.Vector3();
  let applied = -1;
  function apply(level) {
    if (level === applied) return;
    applied = level;
    const feed = (1 - level) * (length + 0.002);
    for (const part of parts) {
      const n = part.s.length;
      const pa = part.pos.array, na = part.nrm && part.nrm.array;
      for (let i = 0; i < n; i++) {
        const o = part.off[i];
        at(part.s[i] - feed, q);
        const c = Math.cos(q.th), s = Math.sin(q.th);
        m.set(part.base[i * 3], q.y - s * o, q.z + c * o).applyMatrix4(part.inv);
        pa[i * 3] = m.x; pa[i * 3 + 1] = m.y; pa[i * 3 + 2] = m.z;
        if (na) {
          const d = q.th - part.th[i], cd = Math.cos(d), sd = Math.sin(d);
          const ny = part.baseN[i * 3 + 1], nz = part.baseN[i * 3 + 2];
          m.set(part.baseN[i * 3], ny * cd - nz * sd, ny * sd + nz * cd).applyMatrix3(part.invN).normalize();
          na[i * 3] = m.x; na[i * 3 + 1] = m.y; na[i * 3 + 2] = m.z;
        }
      }
      part.pos.needsUpdate = true;
      if (part.nrm) part.nrm.needsUpdate = true;
    }
  }

  apply(1);
  for (const part of parts) part.g.computeBoundingSphere();
  return { apply, plane };
}

function showFallback(host) {
  host.querySelector('canvas')?.remove();
  const img = document.createElement('img');
  img.src = FALLBACK_IMG;
  img.alt = 'Dynamo Core handheld POS terminal with built-in receipt printer';
  img.style.cssText =
    'position:absolute;left:50%;bottom:8%;height:84%;width:auto;max-width:none;transform:translateX(-50%) rotate(-6deg);' +
    'filter:drop-shadow(0 40px 50px rgba(0,0,0,0.55))';
  host.appendChild(img);
}

function makeShadowTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(0,0,0,0.55)');
  grad.addColorStop(0.5, 'rgba(0,0,0,0.22)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function mount(host) {
  host.dataset.hero3d = 'mounted';

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  } catch (e) {
    showFallback(host);
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.localClippingEnabled = true;

  const canvas = renderer.domElement;
  canvas.style.cssText =
    'position:absolute;inset:0;width:100%;height:100%;display:block;opacity:0;transition:opacity .9s ease;' +
    'touch-action:pan-y;cursor:grab;outline:none';
  host.appendChild(canvas);

  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.85;
  pmrem.dispose();

  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(0.5, 0.8, 1);
  const rim = new THREE.DirectionalLight(ACCENT, 4);
  rim.position.set(-0.8, 0.5, -0.9);
  const rim2 = new THREE.DirectionalLight(ACCENT, 2);
  rim2.position.set(0.9, 0.2, -0.6);
  scene.add(key, rim, rim2);

  const camera = new THREE.PerspectiveCamera(26, 1, 0.01, 10);

  // yaw → tilt → model. Yaw is user/auto rotation, tilt is pitch + mouse lean.
  const yawGroup = new THREE.Group();
  const tiltGroup = new THREE.Group();
  yawGroup.add(tiltGroup);
  scene.add(yawGroup);

  const shadow = new THREE.Mesh(
    new THREE.PlaneGeometry(0.16, 0.16),
    new THREE.MeshBasicMaterial({ map: makeShadowTexture(), transparent: true, depthWrite: false })
  );
  shadow.rotation.x = -Math.PI / 2;
  scene.add(shadow);

  // ---- state ----
  const st = {
    yaw: START_YAW,
    yawVel: 0,
    pitch: START_PITCH,
    leanX: 0,
    leanY: 0,
    targetLeanX: 0,
    targetLeanY: 0,
    dragging: false,
    lastInput: -Infinity,
    time: 0,
    visible: true,
    running: false,
  };

  let model = null;
  let receipt = null;
  const receiptHome = new THREE.Vector3();
  const paperMats = [];
  let feed = null;
  const clipWorld = new THREE.Plane();
  // phase: 'idle' | 'print' | 'tear'
  const paper = { phase: 'idle', t: 0, level: reducedMotion ? 1 : 0 };

  function fit() {
    const w = host.clientWidth || 1;
    const h = host.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // Fit the full device height (plus receipt) and its rotating width.
    const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
    const distH = 0.128 / Math.tan(halfFov);
    const distW = 0.062 / (Math.tan(halfFov) * camera.aspect);
    camera.position.set(0, 0.018, Math.max(distH, distW));
    camera.lookAt(0, 0.004, 0);
    camera.updateProjectionMatrix();
  }
  fit();
  new ResizeObserver(fit).observe(host);

  // ---- load ----
  const loader = new GLTFLoader();
  loader.setMeshoptDecoder(MeshoptDecoder);
  loader.load(
    MODEL_URL,
    (gltf) => {
      model = gltf.scene;
      // Centre the device body on the pivot so it spins about its own axis.
      model.position.set(0, -0.1066, 0);
      tiltGroup.add(model);
      shadow.position.y = -0.1066 - 0.02;

      receipt = model.getObjectByName('Receipt');
      if (receipt) receiptHome.copy(receipt.position);
      receipt?.traverse((o) => {
        if (!o.isMesh) return;
        o.material = o.material.clone();
        o.material.clippingPlanes = [clipWorld];
        paperMats.push(o.material);
      });
      if (receipt) feed = buildFeed(receipt, model);

      canvas.style.opacity = '1';
      if (!reducedMotion) startPrint(0.5);
      start();
    },
    undefined,
    (err) => {
      console.warn('[hero-3d] model failed to load', err);
      renderer.dispose();
      showFallback(host);
    }
  );

  // ---- receipt ----
  function startPrint(delay = 0) {
    paper.phase = 'print';
    paper.t = -delay;
    paper.level = 0;
    if (receipt) receipt.position.copy(receiptHome), receipt.rotation.set(0, 0, 0);
    setPaperOpacity(1);
  }

  function setPaperOpacity(a) {
    for (const m of paperMats) {
      const transparent = a < 1;
      if (m.transparent !== transparent) {
        m.transparent = transparent;
        m.needsUpdate = true;
      }
      m.opacity = a;
    }
  }

  function tearAndReprint() {
    if (!receipt || paper.phase !== 'idle') return;
    if (reducedMotion) return;
    paper.phase = 'tear';
    paper.t = 0;
  }

  function updatePaper(dt) {
    if (!feed) return;
    if (paper.phase === 'print') {
      paper.t += dt;
      const t = clamp(paper.t / PRINT_TIME, 0, 1);
      // Thermal printers feed in quick pulses: a small ripple on a steady feed.
      paper.level = clamp(t + Math.sin(t * Math.PI * 14) * 0.012 * (1 - t), 0, 1);
      if (paper.t >= PRINT_TIME) {
        paper.phase = 'idle';
        paper.level = 1;
      }
    } else if (paper.phase === 'tear') {
      paper.t += dt;
      const t = clamp(paper.t / TEAR_TIME, 0, 1);
      const e = easeInOut(t);
      receipt.position.set(receiptHome.x, receiptHome.y + e * 0.03, receiptHome.z + e * 0.012);
      receipt.rotation.set(-e * 0.35, 0, e * 0.12);
      paper.level = 1;
      setPaperOpacity(1 - e);
      if (t >= 1) startPrint(0.25);
    }
    feed.apply(paper.level);
    clipWorld.copy(feed.plane).applyMatrix4(model.matrixWorld);
  }

  // ---- input ----
  let downX = 0, downY = 0, lastX = 0, lastT = 0, moved = 0, activeId = null;
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();

  function markInteracted() {
    st.lastInput = st.time;
  }

  canvas.addEventListener('pointerdown', (e) => {
    if (activeId !== null) return;
    activeId = e.pointerId;
    st.dragging = true;
    st.yawVel = 0;
    downX = lastX = e.clientX;
    downY = e.clientY;
    lastT = performance.now();
    moved = 0;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = 'grabbing';
    markInteracted();
  });

  canvas.addEventListener('pointermove', (e) => {
    if (e.pointerId !== activeId) return;
    const now = performance.now();
    const dx = e.clientX - lastX;
    const dy = e.movementY || 0;
    moved = Math.max(moved, Math.hypot(e.clientX - downX, e.clientY - downY));
    st.yaw += dx * 0.011;
    if (e.pointerType === 'mouse') st.pitch = clamp(st.pitch + dy * 0.004, -0.3, 0.5);
    const dtm = Math.max(now - lastT, 1) / 1000;
    st.yawVel = clamp((dx * 0.011) / dtm, -12, 12);
    lastX = e.clientX;
    lastT = now;
    st.lastInput = st.time;
  });

  function endDrag(e) {
    if (e.pointerId !== activeId) return;
    activeId = null;
    st.dragging = false;
    canvas.style.cursor = 'grab';
    if (performance.now() - lastT > 80) st.yawVel = 0;
    st.lastInput = st.time;
    if (e.type === 'pointerup' && moved < 6 && model) {
      const r = canvas.getBoundingClientRect();
      ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      if (raycaster.intersectObject(model, true).length) tearAndReprint();
    }
  }
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);

  if (canHover && !reducedMotion) {
    window.addEventListener(
      'pointermove',
      (e) => {
        if (e.pointerType !== 'mouse') return;
        const r = host.getBoundingClientRect();
        const nx = clamp(((e.clientX - (r.left + r.width / 2)) / window.innerWidth) * 2, -1, 1);
        const ny = clamp(((e.clientY - (r.top + r.height / 2)) / window.innerHeight) * 2, -1, 1);
        st.targetLeanY = nx * 0.22;
        st.targetLeanX = ny * 0.12;
      },
      { passive: true }
    );
    document.documentElement.addEventListener('mouseleave', () => {
      st.targetLeanX = st.targetLeanY = 0;
    });
  }

  // ---- loop ----
  const timer = new THREE.Timer();
  function frame() {
    if (!st.running) return;
    requestAnimationFrame(frame);
    timer.update();
    const dt = Math.min(timer.getDelta(), 1 / 20);
    st.time += dt;

    if (!st.dragging) {
      const idle = st.time - st.lastInput > IDLE_BEFORE_AUTO;
      const auto = reducedMotion ? 0 : idle ? AUTO_SPEED : 0;
      // Inertia decays towards the auto-rotate speed.
      st.yawVel = damp(st.yawVel, auto, idle ? 1.2 : 3.5, dt);
      st.yaw += st.yawVel * dt;
      if (idle) st.pitch = damp(st.pitch, START_PITCH, 1.5, dt);
    }
    st.leanX = damp(st.leanX, st.targetLeanX, 4, dt);
    st.leanY = damp(st.leanY, st.targetLeanY, 4, dt);

    const bob = reducedMotion ? 0 : Math.sin(st.time * 1.3) * 0.003;
    const buzz = paper.phase === 'print' && paper.t > 0 ? Math.sin(st.time * 90) * 0.0012 : 0;
    yawGroup.rotation.y = st.yaw + st.leanY;
    tiltGroup.rotation.set(st.pitch + st.leanX, 0, buzz);
    tiltGroup.position.y = bob;
    shadow.scale.setScalar(1 - bob * 25);
    shadow.material.opacity = 0.9 - bob * 40;

    scene.updateMatrixWorld();
    updatePaper(dt);
    renderer.render(scene, camera);
  }

  function start() {
    if (st.running || !st.visible || !model) return;
    st.running = true;
    timer.update();
    requestAnimationFrame(frame);
  }
  function stop() {
    st.running = false;
  }

  new IntersectionObserver(([entry]) => {
    st.visible = entry.isIntersecting && !document.hidden;
    st.visible ? start() : stop();
  }).observe(host);
  document.addEventListener('visibilitychange', () => {
    st.visible = !document.hidden;
    st.visible ? start() : stop();
  });
}

// The page is rendered by a client-side runtime that reads the raw <x-dc>
// template and replaces it with live DOM. Only mount into the live copy: if we
// touched the template, the runtime would clone our canvas as dead markup.
function watch() {
  const tryMount = () => {
    const host = document.getElementById('hero-3d');
    if (host && !host.dataset.hero3d && !host.closest('x-dc')) mount(host);
  };
  tryMount();
  new MutationObserver(tryMount).observe(document.documentElement, { childList: true, subtree: true });
}

watch();
