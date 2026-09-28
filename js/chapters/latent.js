// Chapter 3: latent space and space-time patches.
//  - Autoencoder: a real linear autoencoder trained here. Every 4×4-pixel block (48 numbers) of the 1,044
//    distinct frames in the clip bank is collected, and PCA finds the k directions that keep the most
//    variation (vid.js, Jacobi eigen-solver). PCA is the optimal linear autoencoder (Baldi & Hornik 1989).
//    Real video models use a trained, non-linear 3D VAE. HunyuanVideo (Tencent, arXiv:2412.03603, 2024)
//    reports 4× compression in time, 8× in each spatial direction, into 16 channels: 48× fewer numbers.
//    Stable Diffusion's image VAE: 8× per side into 4 channels (Rombach et al. 2022), also 48×.
//  - Patches: the latent cube (8 frames × 4 × 4 cells) is cut into space-time patches, the tokens of a
//    diffusion transformer (Peebles & Xie 2022; OpenAI's Sora report, Feb 2024, calls them "spacetime
//    patches"). Real models use 1×2×2 or 2×2×2 latent patches.
//  - Attention: real scaled dot-product attention (Vaswani et al. 2017) over the patch tokens, with
//    hand-set toy weights: query = key = the token's latent content (minus the average), plus a small
//    distance penalty (a relative position bias). Masks show full space-time attention, space only
//    (within a frame) and time only (the same place across frames), the factorised form used by earlier
//    video models (Ho et al., "Video Diffusion Models", 2022).
// Real-scale token counts use 24 fps, the HunyuanVideo compression above and 2×2 patches; attention cost
// grows with tokens². Text for comparison: about 1.3 tokens per English word.
import { THREE, M, box, arrow } from '../kit.js';
import { clipBanks, autoencoder, blockOf, putBlock, picture, board, panel, txt, F, COL, fit, tallView, fmtBig, sci, T, CS, FD, B } from '../vid.js';

const G = CS / B, CELL = 0.3, GAP = 0.05, STEP = CELL + GAP, TZ = 0.4;
const CUBE = [1.35, 1.55, 0.5];
const PATCH = { '1': [1, 1], '2': [2, 1], '22': [2, 2] };            // [space, time]
const RES = { 480: [854, 480], 720: [1280, 720], 1080: [1920, 1080] };

function realTokens(secs, res) {
  const [w, h] = RES[res], frames = Math.round(secs * 24);
  const lt = Math.floor(frames / 4) + 1, lh = Math.ceil(h / 8), lw = Math.ceil(w / 8);
  return { pixels: frames * w * h * 3, latent: lt * lh * lw * 16, tokens: lt * Math.ceil(lh / 2) * Math.ceil(lw / 2) };
}

export default {
  id: 'latent',
  short: 'Latents and patches',
  title: 'Squeeze it, then cut it into patches',
  subtitle: 'Compress the video into a small latent cube, chop it into space-time tokens, and let a transformer attend.',
  view: { pos: [0.4, 4.6, 9.2], target: [0.2, 1.2, 0.4] },
  learn: `<p>Denoising 149 million numbers a second directly would be far too slow. So video models first <b>squeeze</b> the video. An <b>autoencoder</b> has two halves: an <b>encoder</b> that turns each patch of pixels into a few numbers, and a <b>decoder</b> that turns them back. The small version is called the <b>latent</b>. All the diffusion happens in latent space, and the decoder paints the final pixels at the end.</p>
    <p>Ours is real and was trained a moment ago on every frame in the toy clips. It turns each 4×4 block of pixels (48 numbers) into <b>k</b> numbers. Slide k down and watch the picture blur: the squeeze keeps the big shapes and throws away detail. Open video models reportedly shrink time by 4 and each side by 8, keeping 16 numbers per spot: about <b>48 times smaller</b>.</p>
    <p>Next, the latent cube is cut into <b>space-time patches</b>, a few cells wide and a few frames long. Each patch becomes one <b>token</b>, just like a word piece in a language model. OpenAI's Sora report (2024) described this idea as "spacetime patches".</p>
    <p>A <b>transformer</b> then lets every token look at every other one with <b>attention</b> (see LLMClear). Click a cell: its token looks for tokens with similar content. The red ball's token finds the ball in other frames. That is how a model keeps things consistent over time. The catch: attention compares every pair, so 100,000 tokens means 10 billion pairs.</p>
    <p class="tip"><b>Try it:</b> drop the latent channels to 1, then raise them to 12. Click the ball in the cube, then switch attention to "space only" and "time only". Finally set a 10-second 1080p clip and read the token count on the board.</p>`,
  terms: [
    { t: 'Autoencoder', d: 'An encoder that squeezes data into a few numbers, plus a decoder that rebuilds it.' },
    { t: 'Latent', d: 'The squeezed version of the video that the model actually works on.' },
    { t: 'Patch', d: 'A small block of the latent cube, a few cells across and a few frames long.' },
    { t: 'Token', d: 'One patch turned into a list of numbers, the unit a transformer reads.' },
    { t: 'Attention', d: 'Each token scores every other token and takes a weighted mix of their information.' },
    { t: 'Diffusion transformer', d: 'A transformer used as the denoiser in a diffusion model.' },
  ],
  defaults: { k: 4, f: 3, patch: '1', attn: 'full', qf: 3, qx: 1, qy: 1, secs: 5, res: 720 },
  controls: [
    { key: 'k', type: 'range', label: 'Latent numbers per 4×4 block (k)', min: 1, max: 12, step: 1, fmt: (v) => `${v} of 48 · ${(48 / v).toFixed(v > 5 ? 1 : 0)}× smaller` },
    { key: 'f', type: 'range', label: 'Frame shown in the autoencoder', min: 0, max: T - 1, step: 1, fmt: (v) => `frame ${v + 1} of ${T}` },
    { key: 'patch', type: 'seg', label: 'Patch size', options: [{ v: '1', label: '1×1×1' }, { v: '2', label: '2×2×1' }, { v: '22', label: '2×2×2' }], fmt: (v) => { const [a, b] = PATCH[v]; return `${(G / a) * (G / a) * (T / b)} tokens`; } },
    { key: 'attn', type: 'seg', label: 'Attention', options: [{ v: 'full', label: 'Space + time' }, { v: 'space', label: 'Space only' }, { v: 'time', label: 'Time only' }] },
    { key: 'q', type: 'buttons', label: 'Token that is looking (or click a cell)', items: [{ label: 'The ball, frame 4', act: (s) => Object.assign(s, { qf: 3, qx: 1, qy: 1 }) }, { label: 'Empty sky, frame 4', act: (s) => Object.assign(s, { qf: 3, qx: 3, qy: 0 }) }] },
    { key: 'secs', type: 'range', label: 'A real clip: seconds', min: 1, max: 20, step: 1, fmt: (v) => `${v} s at 24 fps` },
    { key: 'res', type: 'seg', label: 'A real clip: resolution', options: [480, 720, 1080].map((v) => ({ v, label: v + 'p' })) },
  ],
  quiz: [
    { q: 'Why do video models work in a latent space?', options: ['It looks nicer', 'It is many times smaller than the pixels, so denoising is affordable', 'It stores the text prompt', 'It adds colour'], answer: 1, why: 'Open models reportedly squeeze video about 48 times before any denoising happens.' },
    { q: 'What is a space-time patch?', options: ['A sticker on a video', 'A small block of the latent cube, a few cells wide and a few frames long, used as a token', 'One pixel', 'A sound clip'], answer: 1, why: 'Cutting the latent cube into patches turns video into tokens a transformer can read.' },
    { q: 'Why does attention get so expensive for long videos?', options: ['Each token compares itself with every other token, so the work grows with tokens squared', 'It needs the internet', 'Videos have no tokens', 'The decoder is slow'], answer: 0, why: 'Twice the tokens means about four times the pairs to compare.' },
  ],
  reel: [
    { ms: 5600, caption: 'Models first squeeze video into a small latent cube, then cut it into space-time patches.', set: { k: 4, patch: '2', attn: 'full', qf: 3, qx: 1, qy: 1, secs: 5, res: 720 }, anim: { f: [0, 7] }, spin: 0.3 },
  ],

  build({ stage }) {
    const { clips } = clipBanks(), ae = autoencoder();
    const ci = clips.where((m) => m.shape === 'circle' && m.colour === 'red' && m.vx === 1 && m.vy === 1)[0];
    const clip = [...Array(T)].map((_, f) => clips.X.slice(ci * clips.D + f * FD, ci * clips.D + (f + 1) * FD));
    const root = new THREE.Group(); stage.root.add(root);
    // the autoencoder row
    const aeG = new THREE.Group(); root.add(aeG);
    const orig = picture(CS, CS, 1.25, 1.25, { edge: 0.05 }); orig.group.position.set(-3.1, 1.3, 1.2); aeG.add(orig.group);
    const lat = picture(G, G, 0.8, 0.8, { edge: 0.05, color: 0x3a2f4a }); lat.group.position.set(-1.65, 1.3, 1.2); aeG.add(lat.group);
    const rec = picture(CS, CS, 1.25, 1.25, { edge: 0.05 }); rec.group.position.set(-0.2, 1.3, 1.2); aeG.add(rec.group);
    [[-2.35, 0x8ef0ff], [-0.95, 0xc49bff]].forEach(([x, c]) => { const a = arrow(c, 0.42, 0.16, 0.03); a.rotation.z = -Math.PI / 2; a.position.set(x - 0.2, 1.3, 1.25); aeG.add(a); });
    const shelf = box(4.4, 0.08, 1.0, M.matte(0x1a2030)); shelf.position.set(-1.65, 0.55, 1.2); aeG.add(shelf);
    // the latent space-time cube: 8 frames × 4 × 4 cells, frame 1 at the front
    const cellGeo = new THREE.BoxGeometry(CELL, CELL, CELL * 0.7);
    const cubeG = new THREE.Group(); cubeG.position.set(...CUBE); cubeG.rotation.y = -0.72; root.add(cubeG);
    const cells = [];
    for (let f = 0; f < T; f++) for (let by = 0; by < G; by++) for (let bx = 0; bx < G; bx++) {
      const mat = new THREE.MeshStandardMaterial({ color: 0x334, roughness: 0.6, transparent: true, opacity: 0.5, emissive: 0xffd166, emissiveIntensity: 0 });
      const m = new THREE.Mesh(cellGeo, mat);
      m.position.set((bx - (G - 1) / 2) * STEP, ((G - 1) / 2 - by) * STEP, -f * TZ);
      m.userData = { f, bx, by }; cubeG.add(m); cells.push(m);
    }
    stage.pickables = cells;
    const outlines = new THREE.Group(); cubeG.add(outlines);
    const qBox = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), new THREE.LineBasicMaterial({ color: 0xffffff })); cubeG.add(qBox);
    const tArrow = box(0.035, 0.035, T * TZ, M.glow(0x8ef0ff)); tArrow.position.set(0.95, -0.85, -(T - 1) * TZ / 2); cubeG.add(tArrow);
    const bd = board(1100, 470, 3.7, 1.58, (g, w, h, s) => s && drawBoard(g, w, h, s)); bd.mesh.position.set(1.9, 0.3, 2.75); bd.mesh.rotation.x = -1.0; root.add(bd.mesh);
    const L = {
      o: stage.label('pixels: 16×16×3', [-3.1, 2.2, 1.2], aeG),
      l: stage.label('latent', [-1.65, 2.2, 1.2], aeG, 'hot'),
      r: stage.label('decoded', [-0.2, 2.2, 1.2], aeG),
      e: stage.label('encoder', [-2.4, 0.75, 1.5], aeG),
      d: stage.label('decoder', [-0.95, 0.75, 1.5], aeG),
      c: stage.label('latent space-time cube', [0, 1.05, -1.6], cubeG),
      t: stage.label('time →', [1.0, -0.6, -T * TZ], cubeG),
    };
    let akey = '', pkey = '', Z = null, recon = null, err = 0, att = null, toks = null, cellTok = null, st = null;

    function encodeAll(k) {
      Z = clip.map((fr) => { const z = []; for (let by = 0; by < G; by++) for (let bx = 0; bx < G; bx++) z.push(ae.encode(blockOf(fr, bx, by), k)); return z; });
      recon = Z.map((z) => { const fr = new Float32Array(FD); z.forEach((v, i) => putBlock(fr, i % G, Math.floor(i / G), ae.decode(v))); return fr; });
      let e2 = 0, n = 0; clip.forEach((fr, f) => { for (let d = 0; d < FD; d++) { const u = fr[d] - recon[f][d]; e2 += u * u; n++; } }); err = Math.sqrt(e2 / n) / 2;
      cells.forEach((c) => {                    // cell colour = the decoded block's average colour
        const { f, bx, by } = c.userData, fr = recon[f]; let r = 0, g = 0, b = 0;
        for (let j = 0; j < B; j++) for (let i = 0; i < B; i++) { const p = ((by * B + j) * CS + bx * B + i) * 3; r += fr[p]; g += fr[p + 1]; b += fr[p + 2]; }
        const q = B * B, col = [r / q, g / q, b / q].map((v) => Math.min(1, Math.max(0, (v + 1) / 2)));
        c.material.color.setRGB(col[0], col[1], col[2], THREE.SRGBColorSpace);
        const bright = Math.max(...col) > 0.5; c.userData.obj = bright; c.material.opacity = bright ? 0.95 : 0.4;
      });
    }
    function tokens(patch) {
      const [ps, pt] = PATCH[patch], nx = G / ps, nt = T / pt, list = [];
      cellTok = new Int32Array(T * G * G);
      for (let tt = 0; tt < nt; tt++) for (let ty = 0; ty < nx; ty++) for (let tx = 0; tx < nx; tx++) {
        const idx = list.length, v = new Float64Array(Z[0][0].length); let n = 0;
        for (let f = tt * pt; f < (tt + 1) * pt; f++) for (let by = ty * ps; by < (ty + 1) * ps; by++) for (let bx = tx * ps; bx < (tx + 1) * ps; bx++) {
          const z = Z[f][by * G + bx]; z.forEach((u, q) => (v[q] += u)); n++; cellTok[(f * G + by) * G + bx] = idx;
        }
        list.push({ v: v.map((u) => u / n), x: (tx + 0.5) * ps, y: (ty + 0.5) * ps, t: (tt + 0.5) * pt, tt, tx, ty });
      }
      const mean = new Float64Array(list[0].v.length); list.forEach((o) => o.v.forEach((u, q) => (mean[q] += u / list.length)));
      list.forEach((o) => { o.c = o.v.map((u, q) => u - mean[q]); o.n2 = o.c.reduce((a, u) => a + u * u, 0); });
      const mx = Math.max(...list.map((o) => o.n2)) || 1;
      list.beta = 5 / mx;
      return list;
    }
    function attend(s) {
      const qi = cellTok[(s.qf * G + s.qy) * G + s.qx], q = toks[qi];
      const sc = toks.map((o) => {
        if (s.attn === 'space' && o.tt !== q.tt) return -Infinity;
        if (s.attn === 'time' && (o.tx !== q.tx || o.ty !== q.ty)) return -Infinity;
        const dot = q.c.reduce((a, u, i) => a + u * o.c[i], 0);
        return toks.beta * dot - 0.12 * ((o.x - q.x) ** 2 + (o.y - q.y) ** 2) - 0.03 * (o.t - q.t) ** 2;
      });
      const m = Math.max(...sc), ex = sc.map((v) => (v === -Infinity ? 0 : Math.exp(v - m))), z = ex.reduce((a, b) => a + b, 0);
      return { qi, w: ex.map((v) => v / z) };
    }
    function drawBoard(g, w, h, s) {
      panel(g, w, h);
      const r = realTokens(s.secs, s.res);
      txt(g, `A REAL CLIP: ${s.secs} s AT ${s.res}p, 24 fps`, 28, 50, F(28, true), COL.soft);
      const rows = [['Pixel numbers', fmtBig(r.pixels), COL.white], ['Latent numbers (48× smaller)', fmtBig(r.latent), COL.violet], ['Tokens (2×2 patches)', fmtBig(r.tokens), COL.hot], ['Attention pairs (tokens²)', sci(r.tokens * r.tokens), COL.red]];
      rows.forEach(([a, b, c], i) => { txt(g, a, 28, 112 + i * 70, F(32), COL.white); txt(g, b, w - 28, 112 + i * 70, F(34, true, true), c, 'right'); });
      txt(g, `A 1,000-word essay is about 1,300 tokens. Compression as reported for open models.`, 28, h - 26, F(22), COL.soft);
    }

    return {
      pick(o) { if (o.userData.f !== undefined && st) Object.assign(st, { qf: o.userData.f, qx: o.userData.bx, qy: o.userData.by }); },
      update(dt, s, time) {
        dt = Math.max(0, dt); st = s;
        fit(stage, [L.e, L.d, L.t], { x: 0.05, y: -0.1, ny: -0.14 });
        const tall = tallView(stage, 1.2, 0.6, -0.1);
        if (tall) { aeG.position.set(1.4, 2.6, -1.2); cubeG.position.set(0.1, CUBE[1], CUBE[2] + 0.4); bd.mesh.position.x = 0.1; }
        else { aeG.position.set(0, 0, 0); cubeG.position.set(...CUBE); bd.mesh.position.x = 1.9; }
        const fr = Math.round(s.f);
        if (s.k !== akey) { akey = s.k; encodeAll(s.k); pkey = ''; }
        const pk = `${s.k}|${s.patch}|${s.attn}|${s.qf}|${s.qx}|${s.qy}|${fr}|${s.secs}|${s.res}`;
        if (pk !== pkey) {
          const newTok = !pkey || pkey.split('|')[1] !== s.patch || pkey.split('|')[0] !== String(s.k);
          pkey = pk;
          if (newTok || !toks) {
            toks = tokens(s.patch);
            outlines.clear();
            const [ps, pt] = PATCH[s.patch];
            if (ps > 1) toks.forEach((o) => {
              const e = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(ps * STEP, ps * STEP, (pt - 1) * TZ + CELL * 0.8)), new THREE.LineBasicMaterial({ color: 0x8ef0ff, transparent: true, opacity: 0.3 }));
              e.position.set((o.x - G / 2) * STEP, (G / 2 - o.y) * STEP, -(o.t - 0.5) * TZ); outlines.add(e);
            });
          }
          att = attend(s);
          orig.put(clip[fr]); rec.put(recon[fr]);
          const lz = new Float32Array(G * G * 3);          // show the first 3 latent numbers as red, green, blue
          Z[fr].forEach((z, i) => { for (let q = 0; q < 3; q++) lz[i * 3 + q] = q < z.length ? Math.max(-1, Math.min(1, z[q] / (2.2 * Math.sqrt(ae.vals[q])))) : -1; });
          lat.put(lz);
          const q = toks[att.qi], [ps, pt] = PATCH[s.patch];
          qBox.scale.set(ps * STEP + 0.04, ps * STEP + 0.04, (pt - 1) * TZ + CELL * 0.9);
          qBox.position.set((q.x - G / 2) * STEP, (G / 2 - q.y) * STEP, -(q.t - 0.5) * TZ);
          bd.redraw(s);
        }
        const mx = Math.max(...att.w);
        cells.forEach((c) => {
          const { f } = c.userData, w = att.w[cellTok[(f * G + c.userData.by) * G + c.userData.bx]] / mx;
          c.material.emissiveIntensity = 0.9 * w;
          c.scale.setScalar(0.8 + 0.3 * w);
          c.material.opacity = Math.max(c.userData.obj ? 0.95 : 0.35, w > 0.05 ? 0.85 : 0);
          if (f === fr) c.material.emissiveIntensity = Math.max(c.material.emissiveIntensity, 0.12 + 0.06 * Math.sin(time * 5));
        });
      },
      readout(s) {
        if (!att) return '';
        const [ps, pt] = PATCH[s.patch], n = (G / ps) * (G / ps) * (T / pt);
        const top = att.w.map((w, i) => [w, i]).sort((a, b) => b[0] - a[0]).filter(([, i]) => i !== att.qi)[0];
        const tq = toks[att.qi], to = toks[top[1]];
        return `<div class="big">${(48 / s.k).toFixed(s.k > 5 ? 1 : 0)}× smaller, ${Math.round(ae.kept(s.k) * 100)}% kept</div>
          <div class="row"><span>Rebuild error</span><b>${(err * 100).toFixed(1)}% per number</b></div>
          <div class="row"><span>Tokens in the toy cube</span><b>${n} → ${(n * n).toLocaleString('en')} pairs</b></div>
          <div class="row"><span>Strongest link</span><b>frame ${Math.floor(tq.t) + 1} → frame ${Math.floor(to.t) + 1}, ${Math.round(top[0] * 100)}%</b></div>
          <small>"Kept" is the share of variation the ${s.k} numbers keep. Toy attention weights are hand-set.</small>`;
      },
    };
  },
};
