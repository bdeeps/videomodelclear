// VideoModelClear's shared models. Everything here is real, tiny and computed in your browser, with no
// libraries, no downloads and no pretrained weights:
//  - Procedural pictures: a coloured shape (circle, square or triangle) on a dark gradient, drawn with
//    4×4 supersampling. Pixel values live in [-1, 1], the range diffusion models use (Ho et al. 2020).
//  - Banks of training pictures and clips: every picture or 8-frame clip the toy "was trained on".
//  - The exact denoiser for a bank. For Gaussian noise, the best possible guess of the clean picture x0
//    from a noisy x_t = a·x0 + s·noise is the posterior mean: a softmax-weighted average of the training
//    pictures, with weights ∝ exp(−|x_t − a·x_i|² / 2s²) (Tweedie's formula; see Karras et al. 2022,
//    "Elucidating the design space of diffusion-based generative models", the "ideal denoiser"). A real
//    model trains a neural network to approximate this guess. Ours is exact, so it can only ever produce
//    copies of its training pictures: real networks generalise because they approximate it smoothly
//    (Kadkhodaie et al., ICLR 2024).
//  - The DDIM sampler (Song, Meng & Ermon 2021, deterministic, eta = 0) with a cosine noise schedule
//    a = cos(πt/2), s = sin(πt/2) (Nichol & Dhariwal 2021, in its continuous form).
//  - Classifier-free guidance (Ho & Salimans 2021): guess = unconditional + w × (conditional − unconditional).
//  - A linear autoencoder: PCA of 4×4-pixel blocks (the best possible linear autoencoder; Baldi & Hornik
//    1989), found with a Jacobi eigen-solver written here.
//  - Board, tile and picture helpers for drawing in 3D.
import { THREE, M, box, canvasTexture, clamp } from './kit.js';

// ---------------------------------------------------------------- random numbers
export function rng(seed) {
  let a = (seed >>> 0) || 1;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function gaussArr(n, seed) {
  const r = rng(seed * 7919 + 13), a = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(1e-12, r()), v = r(), m = Math.sqrt(-2 * Math.log(u));
    a[i] = m * Math.cos(2 * Math.PI * v); if (i + 1 < n) a[i + 1] = m * Math.sin(2 * Math.PI * v);
  }
  return a;
}

// ---------------------------------------------------------------- procedural pictures
export const COLOURS = {
  red: { rgb: [1.0, 0.36, 0.28], css: '#ff5c47' },
  blue: { rgb: [0.3, 0.72, 1.0], css: '#4cb8ff' },
  yellow: { rgb: [1.0, 0.84, 0.25], css: '#ffd640' },
};
export const CNAMES = Object.keys(COLOURS);
export const SHAPES = ['circle', 'square', 'triangle'];
const BG0 = [0.06, 0.09, 0.19], BG1 = [0.17, 0.18, 0.27];

function inside(shape, dx, dy, r) {
  if (shape === 'circle') return dx * dx + dy * dy <= r * r;
  if (shape === 'square') return Math.max(Math.abs(dx), Math.abs(dy)) <= r * 0.8;
  const t = (dy + r) / (1.75 * r);            // triangle, point up
  return t >= 0 && t <= 1 && Math.abs(dx) <= t * r;
}
// One S×S picture: gradient background plus zero or more shapes {shape, colour, x, y, r} in pixel units.
export function renderFrame(S, objs = [], H = S) {
  const out = new Float32Array(S * H * 3);
  for (let j = 0; j < H; j++) for (let i = 0; i < S; i++) {
    const k = j / (H - 1), p = (j * S + i) * 3;
    let c = [BG0[0] + (BG1[0] - BG0[0]) * k, BG0[1] + (BG1[1] - BG0[1]) * k, BG0[2] + (BG1[2] - BG0[2]) * k];
    for (const o of objs) {
      let cov = 0;
      for (let sy = 0; sy < 4; sy++) for (let sx = 0; sx < 4; sx++) if (inside(o.shape, i + (sx + 0.5) / 4 - o.x, j + (sy + 0.5) / 4 - o.y, o.r)) cov++;
      if (!cov) continue;
      cov /= 16;
      const shade = 1.08 - 0.22 * clamp((j + 0.5 - o.y) / o.r * 0.5 + 0.5, 0, 1), col = COLOURS[o.colour].rgb;
      c = c.map((v, q) => v + (Math.min(1, col[q] * shade) - v) * cov);
    }
    out[p] = c[0] * 2 - 1; out[p + 1] = c[1] * 2 - 1; out[p + 2] = c[2] * 2 - 1;
  }
  return out;
}

// ---------------------------------------------------------------- banks and the exact denoiser
export class Bank {
  constructor(D) { this.D = D; this.meta = []; this.rows = []; }
  add(x, meta) { this.rows.push(x); this.meta.push(meta); }
  finish() {
    const N = (this.N = this.rows.length), D = this.D;
    this.X = new Float32Array(N * D); this.n2 = new Float32Array(N);
    this.rows.forEach((r, i) => { this.X.set(r, i * D); let s = 0; for (let d = 0; d < D; d++) s += r[d] * r[d]; this.n2[i] = s; });
    this.rows = null; this.all = [...Array(N).keys()];
    return this;
  }
  where(f) { return this.all.filter((i) => f(this.meta[i])); }
  // Posterior mean E[x0 | x_t] over the pictures in idx (all if null). prior: optional log-weights per
  // picture (a soft prompt). Writes into out; returns weights.
  denoise(xt, a, s, idx, out, prior = null) {
    const D = this.D, X = this.X, ids = idx || this.all, n = ids.length, L = new Float64Array(n);
    const inv = 1 / (2 * s * s);
    let mx = -Infinity;
    for (let q = 0; q < n; q++) {
      const i = ids[q], o = i * D; let dot = 0;
      for (let d = 0; d < D; d++) dot += xt[d] * X[o + d];
      L[q] = (2 * a * dot - a * a * this.n2[i]) * inv + (prior ? prior[i] : 0); if (L[q] > mx) mx = L[q];
    }
    let z = 0; for (let q = 0; q < n; q++) { L[q] = Math.exp(L[q] - mx); z += L[q]; }
    out.fill(0);
    for (let q = 0; q < n; q++) {
      const w = (L[q] /= z); if (w < 1e-7) continue;
      const o = ids[q] * D; for (let d = 0; d < D; d++) out[d] += w * X[o + d];
    }
    return { ids, w: L };
  }
}
export const alphaOf = (t) => Math.cos((Math.PI / 2) * t);
export const sigmaOf = (t) => Math.sin((Math.PI / 2) * t);

// DDIM from pure noise (t = 1) to a picture (t = 0) in `steps` equal steps of t.
// cond: indices of the pictures that match the prompt (null = no prompt), or prior: soft log-weights
// for a prompt the model has only partly learned. w: guidance strength.
export function ddim(bank, noise, steps, { cond = null, prior = null, w = 1, record = false } = {}) {
  const D = bank.D, x = Float32Array.from(noise), x0 = new Float32Array(D), xu = new Float32Array(D), xc = new Float32Array(D);
  const rec = { xs: record ? [x.slice()] : null, x0s: [], ws: [], t: [1], xus: [], xcs: [] };
  for (let k = 0; k < steps; k++) {
    const t = 1 - k / steps, t2 = 1 - (k + 1) / steps, a = alphaOf(t), s = sigmaOf(t), a2 = t2 <= 0 ? 1 : alphaOf(t2), s2 = t2 <= 0 ? 0 : sigmaOf(t2);
    let wt;
    const guided = (cond || prior) && (w !== 1 || record);
    if (guided) {
      rec.u = bank.denoise(x, a, s, null, xu);
      wt = bank.denoise(x, a, s, cond, xc, prior);
      for (let d = 0; d < D; d++) x0[d] = xu[d] + w * (xc[d] - xu[d]);
      if (record) { rec.xus.push(xu.slice()); rec.xcs.push(xc.slice()); }
    } else wt = bank.denoise(x, a, s, cond, x0, prior);
    for (let d = 0; d < D; d++) { const e = (x[d] - a * x0[d]) / s; x[d] = a2 * x0[d] + s2 * e; }
    if (record) { rec.xs.push(x.slice()); rec.x0s.push(x0.slice()); rec.ws.push(wt); rec.t.push(t2); }
    rec.last = wt;
  }
  rec.x = x;
  return rec;
}
// The training picture closest to x (root-mean-square difference per number).
Bank.prototype.nearest = function (x) {
  let best = Infinity, bi = 0; const D = this.D;
  for (let i = 0; i < this.N; i++) { let s = 0; const o = i * D; for (let d = 0; d < D; d++) { const u = x[d] - this.X[o + d]; s += u * u; } if (s < best) { best = s; bi = i; } }
  return { i: bi, rms: Math.sqrt(best / D) };
};
export function topWeights({ ids, w }, k = 5) {
  const o = []; for (let q = 0; q < ids.length; q++) o.push([w[q], ids[q]]);
  return o.sort((a, b) => b[0] - a[0]).slice(0, k);
}

// ---------------------------------------------------------------- the picture bank (chapters 2 and 4)
export const IMG = 24;
export const imageBank = (() => {
  let bank = null;
  return () => {
    if (bank) return bank;
    bank = new Bank(IMG * IMG * 3);
    const r = rng(11);
    for (const shape of SHAPES) for (const colour of CNAMES) for (let n = 0; n < 32; n++) {
      const rad = 3.6 + r() * 2.6, x = rad + 1.5 + r() * (IMG - 2 * rad - 3), y = rad + 1.5 + r() * (IMG - 2 * rad - 3);
      bank.add(renderFrame(IMG, [{ shape, colour, x, y, r: rad }]), { shape, colour, x, y, r: rad });
    }
    return bank.finish();
  };
})();

// ---------------------------------------------------------------- the clip bank (chapters 3 and 5)
// 8-frame clips of 16×16 pixels: one shape gliding in a straight line, 1 pixel per frame on each axis.
export const T = 8, CS = 16, FD = CS * CS * 3;
const STARTS = { 1: [3.5, 4.5, 5.5], [-1]: [12.5, 11.5, 10.5], 0: [5, 8, 11] };
export const clipBanks = (() => {
  let out = null;
  return () => {
    if (out) return out;
    const clips = new Bank(T * FD), frames = new Bank(FD), seen = new Map();
    for (const shape of SHAPES) for (const colour of CNAMES) for (let vx = -1; vx <= 1; vx++) for (let vy = -1; vy <= 1; vy++) {
      if (!vx && !vy) continue;
      for (let j = 0; j < 3; j++) {
        const x0 = STARTS[vx][j], y0 = STARTS[vy][(j + 1) % 3], flat = new Float32Array(T * FD), path = [];
        for (let f = 0; f < T; f++) {
          const o = { shape, colour, x: x0 + vx * f, y: y0 + vy * f, r: 2.7 }, fr = renderFrame(CS, [o]);
          flat.set(fr, f * FD); path.push([o.x, o.y]);
          const key = `${shape}|${colour}|${o.x}|${o.y}`;
          if (!seen.has(key)) { seen.set(key, 1); frames.add(fr, { shape, colour, x: o.x, y: o.y }); }
        }
        clips.add(flat, { shape, colour, vx, vy, path });
      }
    }
    out = { clips: clips.finish(), frames: frames.finish() };
    return out;
  };
})();
export const match = (colour, shape) => (m) => (colour === 'any' || m.colour === colour) && (shape === 'any' || m.shape === shape);

// ---------------------------------------------------------------- a linear autoencoder (PCA of pixel blocks)
function jacobi(A, n) {
  const a = A.map((r) => Float64Array.from(r)), V = [...Array(n)].map((_, i) => { const r = new Float64Array(n); r[i] = 1; return r; });
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-14) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-12) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  const order = [...Array(n).keys()].sort((i, j) => a[j][j] - a[i][i]);
  return { vals: order.map((i) => a[i][i]), vecs: order.map((i) => Float64Array.from(V.map((r) => r[i]))) };
}
export const B = 4, BD = B * B * 3;             // a block: 4×4 pixels × 3 colours = 48 numbers
export function blockOf(fr, bx, by) { const o = new Float32Array(BD); let q = 0; for (let j = 0; j < B; j++) for (let i = 0; i < B; i++) for (let c = 0; c < 3; c++) o[q++] = fr[((by * B + j) * CS + bx * B + i) * 3 + c]; return o; }
export function putBlock(fr, bx, by, v) { let q = 0; for (let j = 0; j < B; j++) for (let i = 0; i < B; i++) for (let c = 0; c < 3; c++) fr[((by * B + j) * CS + bx * B + i) * 3 + c] = v[q++]; }
export const autoencoder = (() => {
  let ae = null;
  return () => {
    if (ae) return ae;
    const { frames } = clipBanks(), mean = new Float64Array(BD), C = [...Array(BD)].map(() => new Float64Array(BD));
    const blocks = [];
    for (let i = 0; i < frames.N; i++) {
      const fr = frames.X.subarray(i * FD, (i + 1) * FD);
      for (let by = 0; by < CS / B; by++) for (let bx = 0; bx < CS / B; bx++) blocks.push(blockOf(fr, bx, by));
    }
    blocks.forEach((b) => { for (let d = 0; d < BD; d++) mean[d] += b[d] / blocks.length; });
    blocks.forEach((b) => { for (let p = 0; p < BD; p++) { const u = b[p] - mean[p]; for (let q = p; q < BD; q++) C[p][q] += u * (b[q] - mean[q]); } });
    for (let p = 0; p < BD; p++) for (let q = p; q < BD; q++) { C[p][q] /= blocks.length; C[q][p] = C[p][q]; }
    const { vals, vecs } = jacobi(C, BD);
    const total = vals.reduce((s, v) => s + Math.max(0, v), 0);
    ae = {
      mean, vecs, vals, total, blocks: blocks.length,
      encode(b, k) { const z = new Float32Array(k); for (let j = 0; j < k; j++) { let s = 0; for (let d = 0; d < BD; d++) s += (b[d] - mean[d]) * vecs[j][d]; z[j] = s; } return z; },
      decode(z) { const b = new Float32Array(BD); for (let d = 0; d < BD; d++) { let s = mean[d]; for (let j = 0; j < z.length; j++) s += z[j] * vecs[j][d]; b[d] = s; } return b; },
      kept(k) { let s = 0; for (let j = 0; j < k; j++) s += Math.max(0, vals[j]); return s / total; },
    };
    return ae;
  };
})();

// ---------------------------------------------------------------- measuring a generated clip
// Where the object is in a frame: centroid of pixels that differ clearly from the background.
export function objectAt(fr, S = CS) {
  let sx = 0, sy = 0, n = 0, col = [0, 0, 0];
  for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
    const p = (j * S + i) * 3, k = j / (S - 1);
    const bg = [0, 1, 2].map((q) => (BG0[q] + (BG1[q] - BG0[q]) * k) * 2 - 1);
    const d = Math.abs(fr[p] - bg[0]) + Math.abs(fr[p + 1] - bg[1]) + Math.abs(fr[p + 2] - bg[2]);
    if (d > 0.9) { sx += i + 0.5; sy += j + 0.5; n++; col[0] += fr[p]; col[1] += fr[p + 1]; col[2] += fr[p + 2]; }
  }
  if (!n) return null;
  return { x: sx / n, y: sy / n, n, rgb: col.map((v) => v / n) };
}

// ---------------------------------------------------------------- drawing helpers (as in LLMClear)
export const COL = { white: '#e8eef8', soft: 'rgba(255,255,255,.55)', hot: '#ffd166', teal: '#b8f2e6', cyan: '#8ef0ff', orange: '#ffb547', red: '#ff7a59', pink: '#ff5a8a', green: '#7be08c', violet: '#c49bff' };
export function panel(g, w, h) { g.clearRect(0, 0, w, h); g.fillStyle = 'rgba(9,11,16,.92)'; g.fillRect(0, 0, w, h); g.strokeStyle = 'rgba(184,242,230,.25)'; g.lineWidth = 3; g.strokeRect(1.5, 1.5, w - 3, h - 3); }
export function txt(g, s, x, y, font = '24px sans-serif', col = COL.white, align = 'left') { g.font = font; g.fillStyle = col; g.textAlign = align; g.fillText(s, x, y); g.textAlign = 'left'; }
export const F = (px, bold = false, mono = false) => `${bold ? 'bold ' : ''}${px}px ${mono ? '"Geist Mono", ui-monospace, Menlo, monospace' : 'Geist, sans-serif'}`;
export function wrap(g, s, x, y, maxW, lh, font, col) {
  g.font = font; g.fillStyle = col; const words = s.split(' '); let line = '', yy = y;
  for (const w of words) { const t = line ? line + ' ' + w : w; if (g.measureText(t).width > maxW && line) { g.fillText(line, x, yy); line = w; yy += lh; } else line = t; }
  if (line) g.fillText(line, x, yy);
  return yy + lh;
}
export function roundRect(g, x, y, w, h, r) { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); }
export function boardMesh(tex, w, h) { return new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: tex, transparent: true, toneMapped: false, side: THREE.DoubleSide })); }
export function board(pxW, pxH, w, h, draw) { const b = canvasTexture(pxW, pxH, draw); const m = boardMesh(b.tex, w, h); return { ...b, mesh: m }; }
// A small glowing tile with text on its face. redraw(text, colour, sub) to change it.
export function tile(text, w = 0.5, h = 0.5, color = COL.teal, px = 128) {
  const pw = Math.round(px * Math.max(1, w / h)), ph = px;
  const c = canvasTexture(pw, ph, (g, W, H, t = text, col = color, sub = '') => {
    g.clearRect(0, 0, W, H);
    g.fillStyle = 'rgba(12,16,24,.94)'; roundRect(g, 3, 3, W - 6, H - 6, 16); g.fill();
    g.strokeStyle = col; g.lineWidth = 5; roundRect(g, 3, 3, W - 6, H - 6, 16); g.stroke();
    let size = sub ? H * 0.46 : H * 0.58; g.font = F(size, true); while (g.measureText(t).width > W - 18 && size > 12) { size -= 2; g.font = F(size, true); }
    g.fillStyle = COL.white; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(t, W / 2, sub ? H * 0.4 : H / 2 + 2);
    if (sub) { g.font = F(H * 0.2, false, true); g.fillStyle = col; g.fillText(sub, W / 2, H * 0.8); }
    g.textAlign = 'left'; g.textBaseline = 'alphabetic';
  });
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: c.tex, transparent: true, toneMapped: false, side: THREE.DoubleSide }));
  m.redraw = c.redraw;
  return m;
}
// A framed picture: a pixel canvas shown crisp (nearest-neighbour) on a plane in a dark frame.
// put(x) takes values in [-1, 1] (RGB interleaved, W×H), like the models produce.
export function picture(W, H, w, h, { edge = 0.06, color = 0x2a3346 } = {}) {
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const g = c.getContext('2d'), img = g.createImageData(W, H), tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace; tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.LinearFilter; tex.generateMipmaps = false;
  const group = new THREE.Group();
  const frameMat = M.plastic(color, { emissive: color, emissiveIntensity: 0 });
  const back = box(w + edge * 2, h + edge * 2, 0.05, frameMat); back.position.z = -0.03; group.add(back);
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: tex, toneMapped: false, side: THREE.DoubleSide }));
  group.add(mesh);
  const put = (x, gain = 1, off = 0) => {
    const d = img.data;
    for (let p = 0; p < W * H; p++) {
      d[p * 4] = clamp((x[p * 3] * gain + off + 1) * 127.5, 0, 255);
      d[p * 4 + 1] = clamp((x[p * 3 + 1] * gain + off + 1) * 127.5, 0, 255);
      d[p * 4 + 2] = clamp((x[p * 3 + 2] * gain + off + 1) * 127.5, 0, 255);
      d[p * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0); tex.needsUpdate = true;
  };
  const glow = (col, k) => { frameMat.color.setHex(k > 0 ? col : color); frameMat.emissive.setHex(k > 0 ? col : color); frameMat.emissiveIntensity = k; };
  return { group, mesh, put, glow, back, canvas: c };
}
export const hex = (n) => '#' + n.toString(16).padStart(6, '0');
export const fmtBig = (x) => {
  if (x >= 1e12) return (x / 1e12).toFixed(x < 1e13 ? 1 : 0) + ' trillion';
  if (x >= 1e9) return (x / 1e9).toFixed(x < 1e10 ? 1 : 0) + ' billion';
  if (x >= 1e6) return (x / 1e6).toFixed(x < 1e7 ? 1 : 0) + ' million';
  return Math.round(x).toLocaleString('en');
};
export const sci = (x) => { const e = Math.floor(Math.log10(x)); return `${(x / 10 ** e).toFixed(1)} × 10^${e}`; };
export const inReel = () => document.body.classList.contains('gb-reel');
// Frame the model: on wide screens nudge the picture right so the readout in the top-left corner
// doesn't cover it; on phones hide minor labels and nudge it down instead.
export function fit(stage, minor = [], { x = 0, y = 0, ny = -0.12 } = {}) {
  const narrow = stage.host.clientWidth < 560, reel = inReel();
  minor.forEach((l) => { if (l) l.visible = !narrow; });
  const sx = narrow || reel ? 0 : x, sy = reel ? 0 : narrow ? ny : y;
  if (!stage.shift || stage.shift[0] !== sx || stage.shift[1] !== sy) stage.setShift(sx, sy);
  return narrow;
}
// Tall video frames (the 9:16 reel): re-aim the camera. dy raises the target; zoom scales the distance.
export const isTall = (stage) => inReel() && stage.host.clientWidth / Math.max(1, stage.host.clientHeight) < 0.8;
export function tallView(stage, dy, zoom = 1, dx = 0) {
  const tall = isTall(stage);
  if (tall && stage.home && stage.home !== stage._tallHome) {
    const { pos, target } = stage.home;
    const t = [target[0] + dx, target[1] + dy, target[2]];
    stage.setView(t.map((v, i) => v + (pos[i] - target[i]) * zoom), t, 1.0);
    stage._tallHome = stage.home;
  }
  return tall;
}
