// Homepage background: the KITTI 00 map building up along the drive, SLAM-viewer style
// (camera frustum, keyframes, loop closures), then slowly orbiting.
// Data comes from scripts/build_hero_map.py (media/hero/map.bin).
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js';

const BUILD_SECONDS = 22;
const KEYFRAME_EVERY = 12;   // trajectory samples between keyframes
const PULSE_SECONDS = 1.6;
const BLUE = 0x2563eb, KEYFRAME = 0x93c5fd, LOOP = 0x16a34a;
const host = document.getElementById('hero-map');
const still = matchMedia('(prefers-reduced-motion: reduce)').matches;

const buf = await (await fetch('media/hero/map.bin')).arrayBuffer();
const [n, m, nl] = new Uint32Array(buf, 0, 3);
const s = new Float32Array(buf, 12, 1)[0];
const shown = new Uint32Array(buf, 16, m);
const loops = new Uint32Array(buf, 16 + m * 4, nl * 2);
const off = 16 + (m + nl * 2) * 4;
const pts = Float32Array.from(new Int16Array(buf, off, n * 3), v => v / s);
const traj = Float32Array.from(new Int16Array(buf, off + n * 6, m * 3), v => v / s);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
host.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 1, 5000);
const lines = (positions, color, opts = {}) => {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, ...opts }));
};

const mapGeo = new THREE.BufferGeometry();
mapGeo.setAttribute('position', new THREE.BufferAttribute(pts, 3));
scene.add(new THREE.Points(mapGeo, new THREE.PointsMaterial({ size: 2, color: 0x64748b })));

const pathGeo = new THREE.BufferGeometry();
pathGeo.setAttribute('position', new THREE.BufferAttribute(traj, 3));
scene.add(new THREE.Line(pathGeo, new THREE.LineBasicMaterial({ color: BLUE })));

// camera frustum at pose i: apex->corner edges + image rectangle = 8 segments = 48 floats
const UP = new THREE.Vector3(0, 1, 0);
function frustum(i, size, out, o) {
  const c = new THREE.Vector3().fromArray(traj, i * 3);
  const f = new THREE.Vector3().fromArray(traj, Math.min(i + 1, m - 1) * 3).sub(c);
  if (f.lengthSq() < 1e-6) f.copy(c).sub(new THREE.Vector3().fromArray(traj, Math.max(i - 1, 0) * 3));
  f.y = 0;
  if (f.lengthSq() < 1e-6) f.set(0, 0, 1);
  f.normalize();
  const r = new THREE.Vector3().crossVectors(f, UP);
  const k = [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([a, b]) =>
    c.clone().addScaledVector(f, size).addScaledVector(r, a * size * 0.7).addScaledVector(UP, b * size * 0.45));
  for (let q = 0; q < 4; q++) {
    c.toArray(out, o + q * 6); k[q].toArray(out, o + q * 6 + 3);
    k[q].toArray(out, o + 24 + q * 6); k[(q + 1) % 4].toArray(out, o + 24 + q * 6 + 3);
  }
}

const nk = Math.ceil(m / KEYFRAME_EVERY);
const kfPos = new Float32Array(nk * 48);
for (let q = 0; q < nk; q++) frustum(q * KEYFRAME_EVERY, 20, kfPos, q * 48);
const keyframes = lines(kfPos, KEYFRAME);
scene.add(keyframes);

const camPos = new Float32Array(48);
const cam = lines(camPos, BLUE);
scene.add(cam);

// loop closures: a ring stays where the drive revisits itself, a pulse expands when it happens
const circle = new Float32Array(64 * 3).map((_, i) => (i % 3 === 1 ? 0 : (i % 3 === 0 ? Math.cos : Math.sin)(Math.floor(i / 3) / 64 * Math.PI * 2)));
const ringGeo = new THREE.BufferGeometry();
ringGeo.setAttribute('position', new THREE.BufferAttribute(circle, 3));
const loopRings = [...Array(nl)].map((_, q) => {
  const ring = new THREE.LineLoop(ringGeo, new THREE.LineBasicMaterial({ color: LOOP }));
  ring.position.fromArray(traj, loops[q * 2] * 3);
  ring.scale.setScalar(22);
  scene.add(ring);
  return ring;
});
const pulse = new THREE.LineLoop(ringGeo, new THREE.LineBasicMaterial({ color: LOOP, transparent: true }));
scene.add(pulse);

function resize() {
  const { clientWidth: w, clientHeight: h } = host;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  // shift the map left on wide screens so it sits beside the profile photo
  camera.setViewOffset(w, h, w > 900 ? w * 0.12 : 0, 0, w, h);
  camera.updateProjectionMatrix();
}

function draw(t) {
  const k = Math.min(m - 1, Math.floor((t / BUILD_SECONDS) * m));
  const done = k === m - 1;
  mapGeo.setDrawRange(0, shown[k]);
  pathGeo.setDrawRange(0, k + 1);
  keyframes.geometry.setDrawRange(0, 16 * (Math.floor(k / KEYFRAME_EVERY) + 1));

  frustum(k, 42, camPos, 0);
  cam.geometry.attributes.position.needsUpdate = true;
  cam.visible = !done;

  pulse.visible = false;
  loopRings.forEach((ring, q) => {
    const i = loops[q * 2];
    ring.visible = k >= i;
    const age = t - (i / m) * BUILD_SECONDS;
    if (age >= 0 && age < PULSE_SECONDS) {
      pulse.visible = true;
      pulse.position.copy(ring.position);
      pulse.scale.setScalar(22 + age * 90);
      pulse.material.opacity = 1 - age / PULSE_SECONDS;
    }
  });

  const a = 0.6 + t * 0.04;
  camera.position.set(Math.sin(a) * 1050, 880, Math.cos(a) * 1050);
  camera.lookAt(0, -40, 0);
  renderer.render(scene, camera);
}

new ResizeObserver(() => { resize(); if (still) draw(BUILD_SECONDS); }).observe(host);
resize();

if (still) {
  draw(BUILD_SECONDS);
} else {
  // only animate while the hero is on screen
  let visible = true, t = 0, last = performance.now();
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(host);
  renderer.setAnimationLoop(now => {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    if (!visible) return;
    t += dt;
    draw(t);
  });
}
host.classList.add('ready');
