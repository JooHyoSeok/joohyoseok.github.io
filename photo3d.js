// Profile photo -> 3D: on hover (tap on touch screens) the photo turns into its
// Depth Anything 3 point cloud and sways. Data comes from scripts/build_photo_cloud.py.
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js';

const box = document.querySelector('.photo-wrap');
const canvas = box.querySelector('canvas');
const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
let renderer, scene, camera, pivot, start, active = false;

async function setup() {
  const buf = await (await fetch('media/hero/photo.bin')).arrayBuffer();
  const n = new Uint32Array(buf, 0, 1)[0];
  const [s, fovY, aspect, subjectZ] = new Float32Array(buf, 4, 4);
  const pts = Float32Array.from(new Int16Array(buf, 20, n * 3), v => v / s);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pts, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(buf, 20 + n * 6, n * 3), 3, true));

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;   // colours are already sRGB bytes
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  scene = new THREE.Scene();
  scene.add(new THREE.Points(geo, new THREE.PointsMaterial({ size: 2.6, sizeAttenuation: false, vertexColors: true })));
  // same intrinsics as the photo, so the first frame lines up with it
  camera = new THREE.PerspectiveCamera(fovY, aspect, 0.01, 100);
  pivot = new THREE.Vector3(0, 0, subjectZ);   // orbit around the person, not the background
}

function frame(now) {
  const t = still ? 1.2 : (now - start) / 1000;
  const yaw = 0.22 * Math.sin(t * 1.1), pitch = 0.08 * Math.sin(t * 0.7);
  const r = -pivot.z;
  camera.position.set(r * Math.sin(yaw), r * Math.sin(pitch), pivot.z + r * Math.cos(yaw) * Math.cos(pitch));
  camera.lookAt(pivot);
  renderer.render(scene, camera);
}

let ready;
async function on() {
  ready ??= setup();
  await ready;
  active = true;
  start = performance.now();
  box.classList.add('on');
  renderer.setAnimationLoop(still ? null : frame);
  frame(start);
}
function off() {
  active = false;
  box.classList.remove('on');
  renderer?.setAnimationLoop(null);
}

if (matchMedia('(hover: hover)').matches) {
  box.addEventListener('pointerenter', on);
  box.addEventListener('pointerleave', off);
} else {
  box.addEventListener('click', () => (active ? off() : on()));
}
