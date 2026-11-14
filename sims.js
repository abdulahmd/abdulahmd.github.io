// Live research demos: H-SLAM (CIrENs Lab) and Lunabotics (Comet Aerobotics).
// Both are simplified, scripted re-enactments of the real pipelines, not recordings.
import * as THREE from 'three';

const V3 = THREE.Vector3;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const ACCENT = new THREE.Color(0x8fb4ff); // drone data, the only colour in the demos
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const ease = t => t < .5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
const R = (x0, z0, x1, z1, h = 0, y0 = 0) => ({ x0, z0, x1, z1, h, y0 });
const inRect = (r, x, z, m = 0) => x > r.x0 - m && x < r.x1 + m && z > r.z0 - m && z < r.z1 + m;

function seeded(s) {
  return () => {
    s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// ---------- drawing helpers ----------
function wire(geom, color = 0xffffff) {
  const g = new THREE.Group();
  g.add(new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ color: 0x050505, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 })));
  const mat = new THREE.LineBasicMaterial({ color });
  g.add(new THREE.LineSegments(new THREE.EdgesGeometry(geom, 25), mat));
  g.userData.mat = mat;
  return g;
}
function dashedLoop(pts, color = 0x5a5a5a, opacity = .9) {
  const l = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineDashedMaterial({ color, dashSize: .16, gapSize: .12, transparent: true, opacity }));
  l.computeLineDistances();
  return l;
}
const rectPts = (r, y) => [new V3(r.x0, y, r.z0), new V3(r.x1, y, r.z0), new V3(r.x1, y, r.z1), new V3(r.x0, y, r.z1)];
const circlePts = (x, z, r, y, n = 40) => Array.from({ length: n }, (_, i) => new V3(x + Math.cos(i / n * Math.PI * 2) * r, y, z + Math.sin(i / n * Math.PI * 2) * r));

class Trail {
  constructor(parent, color, opacity = 1, max = 5000) {
    this.pos = new Float32Array(max * 3); this.max = max; this.n = 0;
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.geo.setDrawRange(0, 0);
    this.line = new THREE.Line(this.geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
    this.line.frustumCulled = false;
    parent.add(this.line);
  }
  push(x, y, z) {
    if (this.n >= this.max) return;
    if (this.n) {
      const k = (this.n - 1) * 3;
      if ((x - this.pos[k]) ** 2 + (y - this.pos[k + 1]) ** 2 + (z - this.pos[k + 2]) ** 2 < .01) return;
    }
    this.pos.set([x, y, z], this.n++ * 3);
    this.geo.setDrawRange(0, this.n);
    this.geo.attributes.position.needsUpdate = true;
  }
  set(path, y) { this.clear(); for (const p of path) this.push(p.x, y, p.z); }
  clear() { this.n = 0; this.geo.setDrawRange(0, 0); }
}

// Fading RRT search tree drawn flat on the ground
class TreeViz {
  constructor(parent, y, max = 6000) {
    this.y = y; this.max = max; this.t = 9;
    this.pos = new Float32Array(max * 6);
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.mat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 });
    this.obj = new THREE.LineSegments(this.geo, this.mat);
    this.obj.frustumCulled = false;
    parent.add(this.obj);
  }
  show(edges) {
    const n = Math.min(edges.length / 4, this.max);
    for (let i = 0; i < n; i++) {
      this.pos.set([edges[i * 4], this.y, edges[i * 4 + 1], edges[i * 4 + 2], this.y, edges[i * 4 + 3]], i * 6);
    }
    this.geo.setDrawRange(0, n * 2);
    this.geo.attributes.position.needsUpdate = true;
    this.t = 0;
  }
  update(dt) {
    this.t += dt;
    this.mat.opacity = .45 * Math.max(0, 1 - this.t / 1.6);
    this.obj.visible = this.mat.opacity > 0;
  }
}

// ---------- planning ----------
const segFreeOf = free => (ax, az, bx, bz) => {
  const n = Math.max(1, Math.ceil(Math.hypot(bx - ax, bz - az) / .08));
  for (let i = 1; i <= n; i++) if (!free(ax + (bx - ax) * i / n, az + (bz - az) * i / n)) return false;
  return true;
};

function rrt(sx, sz, gx, gz, segFree, [bx0, bz0, bx1, bz1], step = .55, iters = 3000) {
  const nodes = [{ x: sx, z: sz, p: -1 }], edges = [];
  for (let i = 0; i < iters; i++) {
    const goal = Math.random() < .12;
    const qx = goal ? gx : bx0 + Math.random() * (bx1 - bx0);
    const qz = goal ? gz : bz0 + Math.random() * (bz1 - bz0);
    let best = 0, bd = Infinity;
    for (let k = 0; k < nodes.length; k++) {
      const d = (nodes[k].x - qx) ** 2 + (nodes[k].z - qz) ** 2;
      if (d < bd) { bd = d; best = k; }
    }
    const n = nodes[best], d = Math.sqrt(bd);
    if (d < 1e-6) continue;
    const s = Math.min(step, d), nx = n.x + (qx - n.x) / d * s, nz = n.z + (qz - n.z) / d * s;
    if (!segFree(n.x, n.z, nx, nz)) continue;
    nodes.push({ x: nx, z: nz, p: best });
    edges.push(n.x, n.z, nx, nz);
    if (Math.hypot(gx - nx, gz - nz) < step && segFree(nx, nz, gx, gz)) {
      nodes.push({ x: gx, z: gz, p: nodes.length - 1 });
      edges.push(nx, nz, gx, gz);
      const path = [];
      for (let k = nodes.length - 1; k !== -1; k = nodes[k].p) path.push(nodes[k]);
      path.reverse();
      // shortcut smoothing, as a planner's path smoother would
      const out = [path[0]];
      for (let a = 0; a < path.length - 1;) {
        let b = path.length - 1;
        while (b > a + 1 && !segFree(path[a].x, path[a].z, path[b].x, path[b].z)) b--;
        out.push(path[b]); a = b;
      }
      return { path: out, edges };
    }
  }
  return { path: null, edges };
}

// Hilbert curve index -> cell (classic d2xy)
function hilbert(n, d) {
  let x = 0, y = 0, t = d;
  for (let s = 1; s < n; s *= 2) {
    const rx = 1 & (t >> 1), ry = 1 & (t ^ rx);
    if (ry === 0) {
      if (rx === 1) { x = s - 1 - x; y = s - 1 - y; }
      [x, y] = [y, x];
    }
    x += s * rx; y += s * ry; t >>= 2;
  }
  return [x, y];
}

function segHitsRect(ax, az, bx, bz, r, tmax = .92) {
  let t0 = 0, t1 = tmax;
  const dx = bx - ax, dz = bz - az;
  const p = [-dx, dx, -dz, dz], q = [ax - r.x0, r.x1 - ax, az - r.z0, r.z1 - az];
  for (let i = 0; i < 4; i++) {
    if (p[i] === 0) { if (q[i] < 0) return false; continue; }
    const t = q[i] / p[i];
    if (p[i] < 0) { if (t > t1) return false; if (t > t0) t0 = t; }
    else { if (t < t0) return false; if (t < t1) t1 = t; }
  }
  return true;
}

// ---------- scripted motion (generators advance one frame per yield) ----------
function* wait(S, s) { for (let t = 0; t < s; t += S.dt) yield; }

function* drive(S, a, path, speed, turnRate = 5) {
  for (let k = 1; k < path.length; k++) {
    for (;;) {
      const dx = path[k].x - a.x, dz = path[k].z - a.z, d = Math.hypot(dx, dz);
      if (d < .04) break;
      const err = wrap(Math.atan2(-dz, dx) - a.yaw);
      a.yaw += Math.sign(err) * Math.min(Math.abs(err), turnRate * S.dt);
      if (Math.abs(err) < .5) { // skid-steer: turn in place first, then drive
        const s = Math.min(d, speed * S.dt * (1 - Math.abs(err) / .6));
        a.x += dx / d * s; a.z += dz / d * s;
      }
      yield;
    }
  }
}
function* turnTo(S, a, yaw, rate = 3) {
  for (;;) {
    const err = wrap(yaw - a.yaw);
    if (Math.abs(err) < .01) return;
    a.yaw += Math.sign(err) * Math.min(Math.abs(err), rate * S.dt);
    yield;
  }
}

// ---------- stage scaffold: renderer, orbit camera, HTML labels ----------
function makeSim(face, { target, radius, el, az }) {
  const canvas = face.querySelector('canvas');
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true }); }
  catch { face.classList.add('no-webgl'); return null; }
  const dpr = Math.min(devicePixelRatio, 2);
  renderer.setPixelRatio(dpr);
  renderer.setClearColor(0x050505, 1);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, .1, 200);
  const labelsEl = face.querySelector('.stage-labels');
  const statusEl = face.querySelector('[data-status]');
  const subEl = face.querySelector('[data-sub]');
  const S = { face, scene, camera, dpr, target, radius, az, el, dt: 0, time: 0, progress: 0, finished: false, labels: [], w: 1, h: 1 };

  let drag = null;
  face.addEventListener('pointerdown', e => {
    if (e.target.closest('button, a')) return;
    drag = { x: e.clientX, y: e.clientY };
    face.setPointerCapture(e.pointerId);
    face.classList.add('is-grabbing', 'is-touched');
  });
  face.addEventListener('pointermove', e => {
    if (!drag) return;
    S.az -= (e.clientX - drag.x) * .008;
    S.el = clamp(S.el + (e.clientY - drag.y) * .006, .25, 1.45);
    drag.x = e.clientX; drag.y = e.clientY;
  });
  const end = () => { drag = null; face.classList.remove('is-grabbing'); };
  face.addEventListener('pointerup', end);
  face.addEventListener('pointercancel', end);

  S.resize = () => {
    S.w = face.clientWidth || 1; S.h = face.clientHeight || 1;
    renderer.setSize(S.w, S.h, false);
    camera.aspect = S.w / S.h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(S.resize).observe(face);
  S.resize();

  S.label = (text, pos, cls = '') => {
    const el = document.createElement('span');
    el.className = 'lbl ' + cls;
    el.textContent = text;
    el.hidden = true;
    labelsEl.append(el);
    const L = { el, pos, shown: false };
    S.labels.push(L);
    return L;
  };
  let lastStatus = '', lastSub = '';
  S.phase = (title, detail) => {
    const s = detail ? `${title} · ${detail}` : title;
    if (s !== lastStatus) statusEl.textContent = lastStatus = s;
  };
  S.sub = s => { if (s !== lastSub) subEl.textContent = lastSub = s; };

  const v = new V3();
  S.render = dt => {
    if (!drag && !reduce) S.az += dt * .035;
    const c = Math.cos(S.el);
    camera.position.set(target.x + c * Math.sin(S.az) * radius, target.y + Math.sin(S.el) * radius, target.z + c * Math.cos(S.az) * radius);
    camera.lookAt(target);
    renderer.render(scene, camera);
    for (const L of S.labels) {
      if (!L.shown) { if (!L.el.hidden) L.el.hidden = true; continue; }
      v.copy(L.pos).project(camera);
      L.el.hidden = v.z > 1;
      L.el.style.transform = `translate(${(v.x * .5 + .5) * S.w}px, ${(-v.y * .5 + .5) * S.h}px) translate(-50%, -100%)`;
    }
  };
  S.start = () => { S.reset(); S.task = S.mission(); S.finished = false; S.progress = 0; };
  S.tick = dt => {
    S.dt = dt; S.time += dt;
    if (S.task && S.task.next().done) { S.task = null; S.finished = true; }
    S.update(dt);
  };
  return S;
}

// Shared ghost-to-mapped point cloud
function makeCloud(S, P, ghost) {
  const N = ghost.length;
  const pos = new Float32Array(P), col = new Float32Array(N * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const pts = new THREE.Points(geo, new THREE.PointsMaterial({ size: 2.1 * S.dpr, sizeAttenuation: false, vertexColors: true }));
  S.scene.add(pts);
  const cloud = {
    N, pos, col, geo, state: new Uint8Array(N), dirty: false, count: 0,
    reset() {
      this.state.fill(0); this.count = 0;
      for (let i = 0; i < N; i++) col.fill(ghost[i], i * 3, i * 3 + 3);
      geo.attributes.color.needsUpdate = true;
    },
    paint(i, b, tint) {
      if (tint) col.set([tint.r * b, tint.g * b, tint.b * b], i * 3); else col.fill(b, i * 3, i * 3 + 3);
      this.dirty = true;
    },
    flush() {
      if (this.dirty) { geo.attributes.color.needsUpdate = true; this.dirty = false; }
    },
  };
  return cloud;
}

// ---------- models ----------
function makeJackal() {
  const g = new THREE.Group();
  const body = wire(new THREE.BoxGeometry(1, .3, .7)); body.position.y = .32; g.add(body);
  for (const sx of [-.33, .33]) for (const sz of [-.4, .4]) {
    const w = wire(new THREE.CylinderGeometry(.16, .16, .12, 12));
    w.rotation.x = Math.PI / 2; w.position.set(sx, .16, sz); g.add(w);
  }
  const lidar = wire(new THREE.CylinderGeometry(.12, .12, .12, 14)); lidar.position.set(.3, .53, 0); g.add(lidar);
  const cam = wire(new THREE.BoxGeometry(.08, .08, .24)); cam.position.set(.5, .42, 0); g.add(cam);
  const deck = wire(new THREE.BoxGeometry(.5, .02, .62)); deck.position.set(-.2, .48, 0); g.add(deck);
  return g;
}
function makeDrone() {
  const g = new THREE.Group();
  const s = .14, seg = [-s, 0, -s, s, 0, s, -s, 0, s, s, 0, -s];
  for (const [cx, cz] of [[-s, -s], [s, s], [-s, s], [s, -s]]) {
    for (let i = 0; i < 12; i++) {
      const a = i / 12 * Math.PI * 2, b = (i + 1) / 12 * Math.PI * 2;
      seg.push(cx + Math.cos(a) * .065, .01, cz + Math.sin(a) * .065, cx + Math.cos(b) * .065, .01, cz + Math.sin(b) * .065);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(seg, 3));
  g.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0xffffff })));
  const hub = wire(new THREE.BoxGeometry(.07, .035, .07)); g.add(hub);
  // downward camera frustum, scaled by altitude each frame
  const c = [];
  const corners = [[1, 1], [1, -1], [-1, -1], [-1, 1]];
  corners.forEach(([x, z], i) => { const [nx, nz] = corners[(i + 1) % 4]; c.push(0, 0, 0, x, -1, z, x, -1, z, nx, -1, nz); });
  const cg = new THREE.BufferGeometry();
  cg.setAttribute('position', new THREE.Float32BufferAttribute(c, 3));
  const cone = new THREE.LineSegments(cg, new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: .28 }));
  g.add(cone);
  g.userData.cone = cone;
  return g;
}
function makeExcavator() {
  const g = new THREE.Group();
  const body = wire(new THREE.BoxGeometry(1, .32, .72)); body.position.y = .4; g.add(body);
  for (const sx of [-.34, .34]) for (const sz of [-.44, .44]) {
    const w = wire(new THREE.CylinderGeometry(.2, .2, .14, 14));
    w.rotation.x = Math.PI / 2; w.position.set(sx, .2, sz); g.add(w);
  }
  const mast = wire(new THREE.BoxGeometry(.04, .4, .04)); mast.position.set(.12, .76, 0); g.add(mast);
  const cam = wire(new THREE.BoxGeometry(.1, .08, .2)); cam.position.set(.16, .98, 0); g.add(cam);
  const pivot = new THREE.Group(); pivot.position.set(.42, .5, 0); g.add(pivot);
  for (const z of [-.28, .28]) { const arm = wire(new THREE.BoxGeometry(.5, .05, .06)); arm.position.set(.25, 0, z); pivot.add(arm); }
  const bucket = new THREE.Group(); bucket.position.set(.52, 0, 0); pivot.add(bucket);
  const shell = wire(new THREE.BoxGeometry(.24, .2, .66)); shell.position.set(.08, -.08, 0); bucket.add(shell);
  const load = wire(new THREE.BoxGeometry(.18, .07, .56), 0x8f8f8f); load.position.set(.08, .03, 0); load.visible = false; bucket.add(load);
  g.userData = { pivot, bucket, load };
  return g;
}

// ======================================================================
// H-SLAM: Jackal covers the floor on a Hilbert curve, launches Crazyflies
// over ground it can't drive, and a VLM labels the merged map.
// ======================================================================
function buildHslam(face) {
  const S = makeSim(face, { target: new V3(8, 0, 8), radius: 24, el: .95, az: -.55 });
  if (!S) return null;
  const { scene } = S;
  const rnd = seeded(11);
  const SPEED = 4, ALT = 2.6, INF = .38;

  const outer = [R(-.2, -.2, 16.2, 0, 1.2), R(-.2, 16, 16.2, 16.2, 1.2), R(-.2, 0, 0, 16, 1.2), R(16, 0, 16.2, 16, 1.2)];
  const inner = [R(7.85, 3, 8.15, 9.5, 1.2), R(1, 7.85, 5.5, 8.15, 1.2)];
  const furniture = [R(3, 3.2, 5, 4.2, .75), R(10.8, 5, 11.4, 8, 1.6), R(6, 11, 7, 12, .8), R(9.2, 1.4, 10.2, 2.4, .8)];
  const zones = [
    { name: 'mezzanine', ...R(10, 12, 16, 16, 1.4) },
    { name: 'rubble field', ...R(0, 12, 4, 16, .6) },
    { name: 'stairwell', ...R(12, 8, 16, 12, 1.4) },
  ];
  const blocked = [...inner, ...furniture, ...zones];
  const free = (x, z) => {
    if (x < INF || z < INF || x > 16 - INF || z > 16 - INF) return false;
    for (const r of blocked) if (inRect(r, x, z, INF)) return false;
    return true;
  };
  const segFree = segFreeOf(free);
  const occluders = [...inner, furniture[1], zones[0]];
  const los = (ax, az, bx, bz) => !occluders.some(r => segHitsRect(ax, az, bx, bz, r));

  // ----- point cloud of the whole space; dim until something maps it -----
  const FLOOR = 1, DRONE = 2;
  const P = [], F = [];
  for (let x = .125; x < 16; x += .25) for (let z = .125; z < 16; z += .25) {
    if (inRect(zones[0], x, z) || inRect(zones[2], x, z)) continue;
    P.push(x, 0, z); F.push(FLOOR | (inRect(zones[1], x, z) ? DRONE : 0));
  }
  function sampleBox(r, sideFlag, topFlag, s) {
    const ny = Math.max(1, Math.round(r.h / s));
    for (let j = r.y0 ? 0 : 1; j <= ny; j++) {
      const y = r.y0 + r.h * j / ny;
      for (let x = r.x0; x <= r.x1 + 1e-6; x += s) { P.push(x, y, r.z0, x, y, r.z1); F.push(sideFlag, sideFlag); }
      for (let z = r.z0 + s; z < r.z1; z += s) { P.push(r.x0, y, z, r.x1, y, z); F.push(sideFlag, sideFlag); }
    }
    for (let x = r.x0 + s; x < r.x1; x += s) for (let z = r.z0 + s; z < r.z1; z += s) { P.push(x, r.y0 + r.h, z); F.push(topFlag); }
  }
  outer.forEach(r => sampleBox(r, 0, 0, .25));
  inner.forEach(r => sampleBox(r, 0, 0, .2));
  furniture.forEach(r => sampleBox(r, 0, 0, .18));
  sampleBox(zones[0], 0, DRONE, .2);
  for (let i = 0; i < 4; i++) sampleBox(R(12, 8 + i, 16, 9 + i, .35 * (i + 1)), 0, DRONE, .2);
  for (let i = 0; i < 16; i++) {
    const s = .25 + rnd() * .45, x = .3 + rnd() * (3.4 - s), z = 12.3 + rnd() * (3.4 - s);
    sampleBox(R(x, z, x + s, z + s, s * (.6 + rnd() * .6)), DRONE, DRONE, .12);
  }
  const flags = Uint8Array.from(F);
  const ghost = Float32Array.from(F, f => f & FLOOR ? .05 : .11);
  const cloud = makeCloud(S, P, ghost);
  const { pos, state } = cloud;
  const reveal = (i, who) => {
    state[i] = who; cloud.count++;
    cloud.paint(i, flags[i] & FLOOR ? .4 : .95, who === 2 ? ACCENT : null);
  };

  // ----- objects the VLM can label -----
  const objects = [
    { name: 'chair', x: 2.4, y: 0, z: 5.6 },
    { name: 'monitor', x: 4, y: .75, z: 3.7 },
    { name: 'fire extinguisher', x: 8.45, y: 0, z: 4.4 },
    { name: 'backpack', x: 13.4, y: 0, z: 7.4 },
    { name: 'toolbox', x: 13, y: 1.4, z: 14, drone: true },
    { name: 'helmet', x: 2.1, y: .4, z: 14.1, drone: true },
    { name: 'valve', x: 14.8, y: 1.4, z: 11.5, drone: true },
  ];
  for (const o of objects) {
    o.glyph = wire(new THREE.BoxGeometry(.28, .28, .28), 0x333333);
    o.glyph.position.set(o.x, o.y + .14, o.z);
    scene.add(o.glyph);
    o.label = S.label(o.name, new V3(o.x, o.y + .45, o.z));
  }
  const see = (o, by) => {
    if (o.seen) return;
    o.seen = true; S.labeled++;
    o.glyph.userData.mat.color.set(by === 2 ? ACCENT : 0xffffff);
    o.label.shown = true;
    o.label.el.classList.add('new');
  };

  // ----- coverage plan: order-3 Hilbert curve over 2 m cells -----
  const NUDGE = [[.8, 0], [-.8, 0], [0, .8], [0, -.8], [.8, .8], [-.8, .8], [.8, -.8], [-.8, -.8]];
  const wps = [];
  for (let d = 0; d < 64; d++) {
    const [cx, cz] = hilbert(8, d);
    let x = cx * 2 + 1, z = cz * 2 + 1;
    const zone = zones.findIndex(r => inRect(r, x, z));
    let ok = true;
    if (zone < 0 && !free(x, z)) {
      ok = false;
      for (const [ox, oz] of NUDGE) if (free(x + ox, z + oz)) { x += ox; z += oz; ok = true; break; }
    }
    wps.push({ x, z, zone, ok });
  }
  const planLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(wps.map(w => new V3(w.x, .03, w.z))),
    new THREE.LineDashedMaterial({ color: 0x4a4a4a, dashSize: .2, gapSize: .16 }));
  planLine.computeLineDistances();
  scene.add(planLine);
  const wpCol = new Float32Array(64 * 3);
  const wpGeo = new THREE.BufferGeometry();
  wpGeo.setAttribute('position', new THREE.Float32BufferAttribute(wps.flatMap(w => [w.x, .05, w.z]), 3));
  wpGeo.setAttribute('color', new THREE.BufferAttribute(wpCol, 3));
  scene.add(new THREE.Points(wpGeo, new THREE.PointsMaterial({ size: 5 * S.dpr, sizeAttenuation: false, vertexColors: true })));
  const markWp = (i, s) => {
    if (s === 2) wpCol.set([ACCENT.r * .8, ACCENT.g * .8, ACCENT.b * .8], i * 3); else wpCol.fill(s ? 1 : .3, i * 3, i * 3 + 3);
    wpGeo.attributes.color.needsUpdate = true;
  };

  // ----- agents -----
  const rover = { x: 1, z: 1, yaw: 0, obj: makeJackal() };
  scene.add(rover.obj);
  const roverTrail = new Trail(scene, 0xffffff, .55);
  const legPlan = new Trail(scene, 0xffffff, .9, 400);
  const tree = new TreeViz(scene, .02);
  const ring = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(circlePts(0, 0, 1, 0, 64)),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 }));
  scene.add(ring);

  const pods = [new V3(-.34, .54, -.2), new V3(-.34, .54, .2), new V3(.02, .54, -.22)];
  const drones = pods.map((pod, id) => {
    const obj = makeDrone();
    scene.add(obj);
    return { id, pod, obj, pos: new V3(), yaw: 0, state: 'docked', task: null, msg: '', trail: new Trail(scene, ACCENT, .6, 3000) };
  });
  const podWorld = (d, out = new V3()) => out.copy(d.pod).applyMatrix4(rover.obj.matrixWorld);

  function* flyTo(d, tx, ty, tz, speed) {
    for (;;) {
      const dx = tx - d.pos.x, dy = ty - d.pos.y, dz = tz - d.pos.z, dist = Math.hypot(dx, dy, dz);
      if (dist < .04) return;
      const s = Math.min(dist, speed * S.dt);
      d.pos.x += dx / dist * s; d.pos.y += dy / dist * s; d.pos.z += dz / dist * s;
      if (Math.hypot(dx, dz) > .05) d.yaw += wrap(Math.atan2(-dz, dx) - d.yaw) * Math.min(1, S.dt * 4);
      yield;
    }
  }
  function* returnHome(d) {
    d.msg = `drone ${d.id + 1} returning`;
    const home = new V3();
    for (;;) {
      podWorld(d, home);
      const dx = home.x - d.pos.x, dz = home.z - d.pos.z, dh = Math.hypot(dx, dz);
      if (dh < .08) break;
      const s = Math.min(dh, 4.8 * S.dt);
      d.pos.x += dx / dh * s; d.pos.z += dz / dh * s;
      d.pos.y += (ALT - d.pos.y) * Math.min(1, S.dt * 3);
      yield;
    }
    for (;;) {
      podWorld(d, home);
      d.pos.x = home.x; d.pos.z = home.z;
      const dy = home.y - d.pos.y;
      if (Math.abs(dy) < .02) break;
      d.pos.y += Math.sign(dy) * Math.min(Math.abs(dy), 1.6 * S.dt);
      yield;
    }
    d.state = 'docked'; d.msg = '';
  }
  const lawnmower = (r, m = .5, sp = 1) => {
    const pts = [];
    let flip = false;
    for (let z = r.z0 + m; z <= r.z1 - m + 1e-6; z += sp) {
      const a = { x: r.x0 + m, z }, b = { x: r.x1 - m, z };
      pts.push(...(flip ? [b, a] : [a, b]));
      flip = !flip;
    }
    return pts;
  };
  function* recon(d, zone) {
    d.state = 'flying'; d.trail.clear();
    d.msg = `drone ${d.id + 1} launched → ${zone.name}`;
    yield* flyTo(d, d.pos.x, ALT, d.pos.z, 2.2);
    const path = lawnmower(zone);
    yield* flyTo(d, path[0].x, ALT, path[0].z, 4.5);
    d.msg = `drone ${d.id + 1} mapping the ${zone.name}`;
    for (const p of path.slice(1)) yield* flyTo(d, p.x, ALT, p.z, 2.6);
    yield* returnHome(d);
  }

  // ----- sensing -----
  function senseRover() {
    const rx = rover.x, rz = rover.z;
    for (let i = 0; i < cloud.N; i++) {
      if (state[i] || flags[i] & DRONE) continue;
      const x = pos[i * 3], z = pos[i * 3 + 2], d2 = (x - rx) ** 2 + (z - rz) ** 2;
      if (d2 > (flags[i] & FLOOR ? 9 : 25)) continue; // depth camera ~3 m on the floor, LiDAR ~5 m on structure
      if (los(rx, rz, x, z)) reveal(i, 1);
    }
    for (const o of objects) if (!o.drone && (o.x - rx) ** 2 + (o.z - rz) ** 2 < 20 && los(rx, rz, o.x, o.z)) see(o, 1);
  }
  function senseDrone(d) {
    const fr2 = (1.5 * Math.min(1, d.pos.y / ALT)) ** 2;
    for (let i = 0; i < cloud.N; i++) {
      if (state[i] || pos[i * 3 + 1] > d.pos.y - .3) continue;
      if ((pos[i * 3] - d.pos.x) ** 2 + (pos[i * 3 + 2] - d.pos.z) ** 2 < fr2) reveal(i, 2);
    }
    for (const o of objects) if ((o.x - d.pos.x) ** 2 + (o.z - d.pos.z) ** 2 < fr2) see(o, 2);
  }

  // ----- prompts -----
  const promptEl = face.querySelector('.stage-prompt');
  function* prompt(text, o) {
    promptEl.hidden = false;
    for (let i = 0; i <= text.length; i++) { promptEl.textContent = '> ' + text.slice(0, i); yield* wait(S, .045); }
    yield* wait(S, .45);
    for (const x of objects) x.label.el.classList.remove('is-target');
    o.label.el.classList.add('is-target');
  }
  const approach = o => {
    for (let k = 0; k < 16; k++) {
      const a = Math.atan2(rover.z - o.z, rover.x - o.x) + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * .4;
      const x = o.x + Math.cos(a) * .9, z = o.z + Math.sin(a) * .9;
      if (free(x, z)) return { x, z };
    }
    return { x: rover.x, z: rover.z };
  };
  function* goTo(x, z) {
    const res = rrt(rover.x, rover.z, x, z, segFree, [.4, .4, 15.6, 15.6]);
    tree.show(res.edges);
    if (!res.path) return false;
    legPlan.set(res.path, .06);
    yield* drive(S, rover, res.path, SPEED);
    legPlan.clear();
    return true;
  }

  function* mission() {
    yield* wait(S, .6);
    for (let i = 0; i < wps.length; i++) {
      const w = wps[i];
      S.progress = .78 * i / wps.length;
      if (w.zone >= 0) {
        markWp(i, 2);
        const z = zones[w.zone];
        if (!z.sent) { z.sent = true; const d = drones[w.zone]; d.state = 'flying'; d.task = recon(d, z); }
        continue;
      }
      if (!w.ok) { markWp(i, 2); continue; }
      S.phase('Coverage', `Hilbert waypoint ${i + 1}/64 · move_base + RRT`);
      if (!(yield* goTo(w.x, w.z))) { markWp(i, 2); continue; }
      markWp(i, 1);
    }
    S.phase('Coverage done', 'waiting for drones to dock');
    while (drones.some(d => d.state !== 'docked')) yield;
    S.progress = .8;
    S.phase('Map merged', `ORB-SLAM · ${S.labeled} objects labeled by the VLM`);
    yield* wait(S, 2);

    yield* prompt('go to the backpack', objects[3]);
    S.phase('Prompt', 'backpack is reachable · sending the rover');
    const g = approach(objects[3]);
    yield* goTo(g.x, g.z);
    S.progress = .9;
    yield* wait(S, .8);

    yield* prompt('inspect the toolbox', objects[4]);
    S.phase('Prompt', 'toolbox is on the mezzanine · sending a drone');
    const d = drones[0];
    d.state = 'flying'; d.trail.clear();
    d.task = (function* () {
      d.msg = 'drone 1 → toolbox';
      yield* flyTo(d, d.pos.x, ALT, d.pos.z, 2.2);
      yield* flyTo(d, objects[4].x, ALT, objects[4].z, 4.5);
      yield* flyTo(d, objects[4].x, 2.1, objects[4].z, 1.2);
      d.msg = 'drone 1 hovering at the toolbox';
      yield* wait(S, 1.4);
      yield* returnHome(d);
    })();
    while (d.state !== 'docked') yield;
    promptEl.hidden = true;
    S.progress = 1;
    S.phase('Mission complete', `${cloud.count.toLocaleString()} map points · ${S.labeled} labels`);
    yield* wait(S, 3.5);
  }

  let senseAcc = 0;
  S.update = dt => {
    rover.obj.position.set(rover.x, 0, rover.z);
    rover.obj.rotation.y = rover.yaw;
    rover.obj.updateMatrixWorld();
    roverTrail.push(rover.x, .04, rover.z);
    for (const d of drones) {
      if (d.task && d.task.next().done) d.task = null;
      if (d.state === 'docked') { podWorld(d, d.pos); d.yaw = rover.yaw; }
      else d.trail.push(d.pos.x, d.pos.y, d.pos.z);
      d.obj.position.copy(d.pos);
      d.obj.rotation.y = d.yaw;
      const cone = d.obj.userData.cone;
      cone.visible = d.state !== 'docked' && d.pos.y > 1;
      cone.scale.set(d.pos.y * .55, d.pos.y, d.pos.y * .55);
    }
    senseAcc += dt;
    if (senseAcc > .08) {
      senseAcc = 0;
      senseRover();
      for (const d of drones) if (d.state !== 'docked' && d.pos.y > 1) senseDrone(d);
      cloud.flush();
    }
    const ping = (S.time * .8) % 1;
    ring.position.set(rover.x, .05, rover.z);
    ring.scale.setScalar(.3 + ping * 4.7);
    ring.material.opacity = .4 * (1 - ping);
    tree.update(dt);
    S.sub(drones.map(d => d.msg).filter(Boolean).join(' · ') || `${cloud.count.toLocaleString()} map points · ${S.labeled} labels`);
  };

  S.reset = () => {
    cloud.reset();
    S.labeled = 0;
    for (const o of objects) {
      o.seen = false; o.label.shown = false;
      o.label.el.classList.remove('is-target', 'new');
      o.glyph.userData.mat.color.set(0x333333);
    }
    zones.forEach(z => { z.sent = false; });
    Object.assign(rover, { x: 1, z: 1, yaw: 0 });
    rover.obj.position.set(1, 0, 1); rover.obj.rotation.y = 0; rover.obj.updateMatrixWorld();
    roverTrail.clear(); legPlan.clear(); tree.t = 9;
    for (const d of drones) { d.state = 'docked'; d.task = null; d.msg = ''; d.trail.clear(); }
    wps.forEach((_, i) => markWp(i, 0));
    promptEl.hidden = true;
    S.phase('Coverage', 'planning a Hilbert curve over the floor');
  };
  S.mission = mission;
  return S;
}

// ======================================================================
// Lunabotics: RTAB-Map + Nav2 (RRT) through craters and boulders,
// dig in the excavation zone, build the berm in the construction zone.
// ======================================================================
function buildLuna(face) {
  const S = makeSim(face, { target: new V3(7, 0, 3.5), radius: 14, el: .92, az: -.45 });
  if (!S) return null;
  const { scene } = S;
  const W = 14, D = 7, SPEED = 1.8;
  const craters = [{ x: 5.6, z: 2, r: .75 }, { x: 8, z: 5, r: .6 }, { x: 8.6, z: 1.5, r: .5 }, { x: 6, z: 5.6, r: .4 }];
  const boulders = [{ x: 4.9, z: 4, r: .3 }, { x: 6.9, z: 3.3, r: .28 }, { x: 9.4, z: 3.6, r: .34 }, { x: 7.2, z: .7, r: .24 }];
  const Z = {
    start: { name: 'Start', ...R(.2, .2, 3, 3) },
    berm: { name: 'Construction · berm', ...R(.4, 4.1, 3.4, 6.8) },
    obstacles: { name: 'Obstacle zone', ...R(4, .2, 10, 6.8) },
    dig: { name: 'Excavation', ...R(10.2, .4, 13.8, 6.6) },
  };
  const bermBlock = R(1.9, 4.4, 3, 6.6);
  const free = (x, z) => {
    if (x < .55 || z < .55 || x > W - .55 || z > D - .55) return false;
    for (const c of craters) if ((x - c.x) ** 2 + (z - c.z) ** 2 < (c.r + .4) ** 2) return false;
    for (const b of boulders) if ((x - b.x) ** 2 + (z - b.z) ** 2 < (b.r + .45) ** 2) return false;
    return !inRect(bermBlock, x, z, .55);
  };
  const segFree = segFreeOf(free);

  // ----- regolith heightfield -----
  const groundH = (x, z) => {
    let h = .03 * Math.sin(x * 1.7 + z * .9) + .02 * Math.sin(x * 3.1 - z * 2.3) + .012 * Math.sin(x * 7.3 + z * 5.1);
    for (const c of craters) {
      const d = Math.hypot(x - c.x, z - c.z) / c.r;
      if (d < 1) h -= .32 * c.r * (1 - d * d);
      else if (d < 1.4) h += .08 * c.r * Math.sin((d - 1) / .4 * Math.PI);
    }
    return h;
  };
  const P = [];
  for (let x = .06; x < W; x += .12) for (let z = .06; z < D; z += .12) P.push(x, groundH(x, z), z);
  const N = P.length / 3;
  const base = Float32Array.from({ length: N }, (_, i) => P[i * 3 + 1]);
  const cloud = makeCloud(S, P, new Float32Array(N).fill(.07));
  const { pos, state } = cloud;
  const shade = i => clamp(.5 + pos[i * 3 + 1] * 2.2, .16, .95);
  const reveal = i => { state[i] = 1; cloud.count++; cloud.paint(i, shade(i)); };

  // arena + zones
  scene.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(rectPts(R(0, 0, W, D), .02)), new THREE.LineBasicMaterial({ color: 0x5a5a5a })));
  for (const z of Object.values(Z)) {
    scene.add(dashedLoop(rectPts(z, .03), 0x4a4a4a));
    S.label(z.name, new V3((z.x0 + z.x1) / 2, .05, z.z0 + .1), 'zone').shown = true;
  }
  // hazards: ghosted until the camera sees them, then their costmap inflation appears
  const hazards = [
    ...craters.map(c => ({ ...c, inf: c.r + .4, kind: 'crater' })),
    ...boulders.map(b => ({ ...b, inf: b.r + .45, kind: 'boulder' })),
  ];
  const rnd = seeded(5);
  for (const h of hazards) {
    if (h.kind === 'boulder') {
      h.obj = wire(new THREE.DodecahedronGeometry(h.r, 0), 0x2a2a2a);
      h.obj.scale.y = .75; h.obj.position.set(h.x, h.r * .45, h.z);
      h.obj.rotation.set(rnd() * 3, rnd() * 3, rnd() * 3);
      scene.add(h.obj);
    }
    h.ring = dashedLoop(circlePts(h.x, h.z, h.inf, .06, 48), 0xffffff, .55);
    scene.add(h.ring);
  }

  const rover = { x: 1.5, z: 1.5, yaw: 0, obj: makeExcavator() };
  scene.add(rover.obj);
  const { pivot, bucket, load } = rover.obj.userData;
  const trail = new Trail(scene, 0xffffff, .5);
  const legPlan = new Trail(scene, 0xffffff, .9, 400);
  const tree = new TreeViz(scene, .04);

  function senseRover() {
    const rx = rover.x, rz = rover.z, yaw = rover.yaw;
    const inView = (x, z, range) => {
      const dx = x - rx, dz = z - rz, d2 = dx * dx + dz * dz;
      return d2 < 1 || (d2 < range * range && Math.abs(wrap(Math.atan2(-dz, dx) - yaw)) < .7);
    };
    for (let i = 0; i < N; i++) if (!state[i] && inView(pos[i * 3], pos[i * 3 + 2], 4)) reveal(i);
    for (const h of hazards) if (!h.seen && inView(h.x, h.z, 4.5)) {
      h.seen = true; S.hazards++;
      h.ring.visible = true;
      if (h.obj) h.obj.userData.mat.color.set(0xffffff);
    }
  }
  // dig a hollow or pile up the berm around a point
  function sculpt(cx, cz, radius, amount) {
    for (let i = 0; i < N; i++) {
      const d2 = (pos[i * 3] - cx) ** 2 + (pos[i * 3 + 2] - cz) ** 2;
      if (d2 > radius * radius) continue;
      pos[i * 3 + 1] += amount * (1 - d2 / (radius * radius));
      if (state[i]) cloud.paint(i, shade(i));
    }
    cloud.geo.attributes.position.needsUpdate = true;
  }
  const tip = () => ({ x: rover.x + Math.cos(rover.yaw) * 1.05, z: rover.z - Math.sin(rover.yaw) * 1.05 });
  function* tween(obj, to, dur, step) {
    const from = obj.rotation.z;
    for (let t = 0; t < dur; t += S.dt) { obj.rotation.z = from + (to - from) * ease(t / dur); step?.(S.dt / dur); yield; }
    obj.rotation.z = to;
  }
  function* goTo(x, z) {
    for (let tries = 0; tries < 3; tries++) {
      const res = rrt(rover.x, rover.z, x, z, segFree, [.55, .55, W - .55, D - .55], .5, 4000);
      tree.show(res.edges);
      if (!res.path) { yield* wait(S, .3); continue; }
      legPlan.set(res.path, .08);
      yield* drive(S, rover, res.path, SPEED, 3);
      legPlan.clear();
      return;
    }
  }

  const digs = [{ x: 11, z: 1.8 }, { x: 11.6, z: 4.6 }, { x: 10.9, z: 3.2 }];
  const dumps = [4.9, 5.5, 6.1];
  function* mission() {
    S.phase('Localizing', 'RTAB-Map sweep from the start zone');
    yield* wait(S, .5);
    yield* turnTo(S, rover, Math.PI, 2.2);
    yield* turnTo(S, rover, -Math.PI / 2 + .01, 2.2);
    yield* turnTo(S, rover, 0, 2.2);
    for (let c = 0; c < 3; c++) {
      const cyc = `Cycle ${c + 1}/3`;
      S.progress = c / 3;
      S.phase(cyc, 'Nav2 → excavation zone · RRT around hazards');
      yield* goTo(digs[c].x, digs[c].z);
      yield* turnTo(S, rover, 0);
      S.phase(cyc, 'excavating regolith');
      yield* tween(pivot, -.5, .6);
      const t = tip();
      for (let k = 0; k < 1.8; k += S.dt) {
        pivot.rotation.z = -.5 + .05 * Math.sin(k * 14);
        sculpt(t.x, t.z, .55, -.16 * S.dt / 1.8);
        yield;
      }
      load.visible = true;
      yield* tween(pivot, .15, .6);
      S.progress = (c + .5) / 3;
      S.phase(cyc, 'Nav2 → construction zone');
      yield* goTo(3.75, dumps[c]);
      yield* turnTo(S, rover, Math.PI);
      S.phase(cyc, 'depositing regolith on the berm');
      yield* tween(pivot, .7, .6);
      const b = tip();
      yield* tween(bucket, -1.3, .9, f => sculpt(b.x - .1, b.z, .8, .3 * f));
      load.visible = false;
      S.loads++;
      yield* tween(bucket, 0, .5);
      yield* tween(pivot, .15, .5);
    }
    S.progress = 1;
    legPlan.clear();
    S.phase('Mission complete', 'berm built from 3 loads');
    yield* wait(S, 3.5);
  }

  let senseAcc = 0;
  S.update = dt => {
    rover.obj.position.set(rover.x, 0, rover.z);
    rover.obj.rotation.y = rover.yaw;
    trail.push(rover.x, .06, rover.z);
    senseAcc += dt;
    if (senseAcc > .08) { senseAcc = 0; senseRover(); }
    cloud.flush();
    tree.update(dt);
    S.sub(`${cloud.count.toLocaleString()} map points · ${S.hazards}/${hazards.length} hazards in costmap · berm ${S.loads}/3`);
  };
  S.reset = () => {
    pos.forEach((_, k) => { if (k % 3 === 1) pos[k] = base[(k - 1) / 3]; });
    cloud.geo.attributes.position.needsUpdate = true;
    cloud.reset();
    S.hazards = 0; S.loads = 0;
    for (const h of hazards) { h.seen = false; h.ring.visible = false; if (h.obj) h.obj.userData.mat.color.set(0x2a2a2a); }
    Object.assign(rover, { x: 1.5, z: 1.5, yaw: 0 });
    pivot.rotation.z = .15; bucket.rotation.z = 0; load.visible = false;
    trail.clear(); legPlan.clear(); tree.t = 9;
    S.phase('Localizing', 'RTAB-Map sweep from the start zone');
  };
  S.mission = mission;
  return S;
}

// ======================================================================
// Flip card controller
// ======================================================================
const stage = document.getElementById('stage');
if (stage) {
  const faces = [...stage.querySelectorAll('.stage-face')];
  const sims = [buildHslam(faces[0]), buildLuna(faces[1])];
  const flipBtn = document.getElementById('stage-flip');
  const flipLabel = flipBtn.querySelector('.stage-flip-label');
  const notes = [...document.querySelectorAll('.sim-note')];
  const names = ['H-SLAM', 'Lunabotics'];
  let active = 0, auto = !reduce, visible = false, flipping = 0;
  sims.forEach(s => s?.start());

  const show = i => {
    active = i;
    stage.classList.toggle('is-flipped', i === 1);
    faces.forEach((f, k) => { f.inert = k !== i; });
    notes.forEach((n, k) => { n.hidden = k !== i; });
    flipLabel.textContent = names[1 - i];
    flipBtn.setAttribute('aria-label', `Show the ${names[1 - i]} demo`);
    if (sims[i]?.finished) sims[i].start();
    flipping = 1.1;
  };
  flipBtn.addEventListener('click', () => { auto = false; stage.classList.add('is-held'); show(1 - active); });
  show(0);

  new IntersectionObserver(([e]) => { visible = e.isIntersecting; }, { threshold: .15 }).observe(stage);

  let last = performance.now();
  const loop = now => {
    requestAnimationFrame(loop);
    const dt = Math.min(.05, (now - last) / 1000);
    last = now;
    if (!visible) return;
    const s = sims[active];
    if (s) {
      s.tick(dt);
      s.render(dt);
      flipBtn.style.setProperty('--flip-progress', s.progress);
      if (s.finished) { if (auto && sims[1 - active]) show(1 - active); else s.start(); }
    }
    if (flipping > 0) { flipping -= dt; sims[1 - active]?.render(0); }
  };
  requestAnimationFrame(loop);
}
