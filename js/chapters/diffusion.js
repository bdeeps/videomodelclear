// Chapter 2: diffusion, from noise to picture. A genuine diffusion sampler running in the browser:
//  - Training set: 288 procedural 24×24 pictures (3 shapes × 3 colours × 32 random places and sizes).
//  - Denoiser: the exact posterior mean E[x0 | x_t] for that set (vid.js), the "ideal denoiser" that a
//    trained network tries to approximate. It is a softmax-weighted average of the training pictures.
//  - Sampler: DDIM (Song, Meng & Ermon 2021), deterministic, cosine schedule a = cos(πt/2), s = sin(πt/2).
//    Each step: guess x0, work out the noise that guess implies, then re-mix at a lower noise level.
// Honest limits: because the denoiser is exact for 288 pictures, every finished sample is a copy of one of
// them (memorisation). Real networks approximate the guess smoothly and so can blend what they learned
// into new pictures (Kadkhodaie et al., "Generalization in diffusion models arises from
// geometry-adaptive harmonic representations", ICLR 2024). DDPM: Ho, Jain & Abbeel 2020.
import { THREE, M, box, arrow } from '../kit.js';
import { imageBank, ddim, gaussArr, topWeights, picture, board, panel, txt, F, COL, fit, tallView, alphaOf, sigmaOf, IMG } from '../vid.js';

const NT = 7;
function thumb(g, x, bank, i, px) {
  const c = document.createElement('canvas'); c.width = c.height = IMG;
  const cg = c.getContext('2d'), im = cg.createImageData(IMG, IMG), X = bank.X, o = i * bank.D;
  for (let p = 0; p < IMG * IMG; p++) { for (let q = 0; q < 3; q++) im.data[p * 4 + q] = (X[o + p * 3 + q] + 1) * 127.5; im.data[p * 4 + 3] = 255; }
  cg.putImageData(im, 0, 0);
  g.imageSmoothingEnabled = false; g.drawImage(c, x[0], x[1], px, px);
}

export default {
  id: 'diffusion',
  short: 'Noise to picture',
  title: 'Diffusion: from noise to picture',
  subtitle: 'Start with static. Guess the picture hiding in it. Remove a little noise. Repeat.',
  view: { pos: [0.4, 3.9, 9.3], target: [0.3, 1.6, 0.4] },
  learn: `<p>Most AI video and image models today are <b>diffusion models</b>. The trick is surprising: they learn to <b>remove noise</b>. In training, you take a real picture, bury it under random static, and ask a neural network to guess the clean picture. Do that millions of times at every amount of noise, and the network gets very good at it (see NeuralNetClear for how networks learn).</p>
    <p>To make something new, start from <b>pure noise</b>. Ask for a guess of the clean picture. The first guess is a blurry average, because it could be anything. Mix that guess back with a bit less noise, and ask again. Step by step, the picture <b>sharpens out of the static</b>. Real models take roughly 20 to 50 steps.</p>
    <p>This one is real and runs right here. Its "memory" is 288 small pictures of shapes. Its guess is the mathematically best one: a <b>weighted average</b> of those 288 pictures, weighted by how well each fits the noise. The board shows which pictures it leans on at each step.</p>
    <p><b>What is simplified:</b> our denoiser is perfect for its 288 pictures, so it can only ever end on a copy of one of them. A real network is trained to <i>approximate</i> this guess, for billions of pictures. Its small, smooth errors are what let it mix ideas and make pictures nobody has seen. The same flaw in reverse is <b>memorisation</b>: real models sometimes reproduce training images too closely.</p>
    <p class="tip"><b>Try it:</b> set the steps to 1 and see the blurry average. Then try 3, 10 and 50. Change the noise seed and watch a different picture appear from different static.</p>`,
  terms: [
    { t: 'Diffusion model', d: 'A generator that starts from random noise and removes it step by step.' },
    { t: 'Noise', d: 'Random static added to a picture. Pure noise has no picture in it at all.' },
    { t: 'Denoiser', d: 'The trained network that guesses the clean picture from a noisy one.' },
    { t: 'Sampling step', d: 'One round of guess, then re-noise a little less. Models use about 20 to 50.' },
    { t: 'Seed', d: 'The number that picks the starting noise. Same seed, same result.' },
    { t: 'Memorisation', d: 'When a model reproduces a training example instead of making something new.' },
  ],
  defaults: { steps: 20, seed: 7, prog: 1, play: true },
  controls: [
    { key: 'steps', type: 'range', label: 'Denoising steps', min: 1, max: 50, step: 1, fmt: (v) => `${v} step${v > 1 ? 's' : ''}` },
    { key: 'seed', type: 'range', label: 'Noise seed', min: 1, max: 40, step: 1, fmt: (v) => `#${v}` },
    { key: 'play', type: 'toggle', label: 'Play the denoising' },
    { key: 'prog', type: 'range', label: 'Progress', min: 0, max: 1, step: 0.001, fmt: (v, s) => `step ${Math.round(v * s.steps)} of ${s.steps}` },
  ],
  quiz: [
    { q: 'What is a diffusion model trained to do?', options: ['Copy videos from the internet', 'Guess the clean picture hidden under noise', 'Count pixels', 'Translate text'], answer: 1, why: 'Training buries real pictures in noise and teaches the network to undo it.' },
    { q: 'Why is the first guess from pure noise a blurry average?', options: ['The screen is too small', 'Pure noise fits every picture equally, so the best guess is their average', 'The model is broken', 'It uses too many steps'], answer: 1, why: 'With no picture left in the noise, the safest guess is the average of everything it knows.' },
    { q: 'Our toy only ever makes copies of its 288 pictures. Why can real models make new ones?', options: ['They are connected to cameras', 'Their networks approximate the guess smoothly, so they blend what they learned', 'They use no noise', 'They have fewer steps'], answer: 1, why: 'A network cannot store every picture exactly, so it learns general patterns, which lets it combine them.' },
  ],
  reel: [
    { ms: 6400, caption: 'Diffusion starts from pure static, guesses the picture, removes a little noise, and repeats.', set: { steps: 20, seed: 7, play: false }, anim: { prog: [0, 1] }, spin: 0.15 },
  ],

  build({ stage }) {
    const bank = imageBank();
    const root = new THREE.Group(); stage.root.add(root);
    const noisy = picture(IMG, IMG, 2.0, 2.0, { edge: 0.07 }); noisy.group.position.set(-1.35, 2.45, 0); noisy.group.rotation.y = 0.12; root.add(noisy.group);
    const guess = picture(IMG, IMG, 2.0, 2.0, { edge: 0.07, color: 0x3a3320 }); guess.group.position.set(1.35, 2.45, 0); guess.group.rotation.y = -0.12; root.add(guess.group);
    const arr = arrow(0x8ef0ff, 0.55, 0.2, 0.035); arr.rotation.z = -Math.PI / 2; arr.position.set(-0.28, 2.45, 0.1); root.add(arr);
    const thumbs = [...Array(NT)].map((_, j) => { const p = picture(IMG, IMG, 0.62, 0.62, { edge: 0.035 }); p.group.position.set(-2.55 + j * 0.85, 0.95, 0.35); p.group.rotation.x = -0.25; root.add(p.group); return p; });
    const rail = box(NT * 0.85, 0.05, 0.05, M.glow(0x3a4658)); rail.position.set(0, 0.52, 0.45); root.add(rail);
    const marker = box(0.08, 0.16, 0.08, M.glow(0xffd166)); marker.position.set(0, 0.52, 0.47); root.add(marker);
    const bd = board(1400, 440, 5.4, 1.7, (g, w, h, s) => s && res && drawBoard(g, w, h, s)); bd.mesh.position.set(0, 0.18, 2.15); bd.mesh.rotation.x = -1.2; root.add(bd.mesh);
    const L = {
      n: stage.label('noisy picture', [-1.35, 3.72, 0], root),
      g: stage.label("the model's guess", [1.35, 3.72, 0], root, 'hot'),
      a: stage.label('pure noise', [-2.55, 1.45, 0.3], root),
      b: stage.label('picture', [2.55, 1.45, 0.3], root),
    };
    let key = '', res = null, shownK = -1, hold = 0;

    function drawBoard(g, w, h, s) {
      panel(g, w, h);
      const k = kOf(s), wt = res.ws[Math.min(k, res.ws.length - 1)], top = topWeights(wt, 4);
      const rest = Math.max(0, 1 - top.reduce((a, [v]) => a + v, 0));
      txt(g, 'THE GUESS IS A WEIGHTED BLEND OF THE 288 PICTURES IT KNOWS', 28, 48, F(28, true), COL.soft);
      top.forEach(([v, i], n) => {
        const x = 30 + n * 250;
        thumb(g, [x, 78], bank, i, 150);
        g.strokeStyle = n === 0 ? COL.hot : 'rgba(255,255,255,.25)'; g.lineWidth = 3; g.strokeRect(x - 1.5, 76.5, 153, 153);
        txt(g, `${(v * 100).toFixed(v > 0.995 ? 0 : 1)}%`, x, 285, F(40, true), n === 0 ? COL.hot : COL.white);
        txt(g, `picture #${i + 1}`, x, 320, F(22, false, true), COL.soft);
      });
      g.fillStyle = 'rgba(255,255,255,.08)'; g.fillRect(1030, 78, 150, 150);
      txt(g, `+${bank.N - 4}`, 1105, 168, F(44, true), COL.soft, 'center');
      txt(g, `${(rest * 100).toFixed(1)}%`, 1030, 285, F(40, true), COL.white);
      txt(g, 'all the others', 1030, 320, F(22, false, true), COL.soft);
      txt(g, k >= s.steps ? 'Finished: an exact copy of one training picture. A real network would blend.' : 'Weight ∝ how well each picture fits the noisy one, at this noise level.', 28, h - 30, F(25), COL.teal);
    }
    const kOf = (s) => Math.min(s.steps, Math.round(s.prog * s.steps));

    return {
      update(dt, s) {
        dt = Math.max(0, dt);
        fit(stage, [L.a, L.b], { x: 0.14, y: -0.1, ny: -0.14 });
        tallView(stage, 0.4, 0.7);
        const k0 = `${s.steps}|${s.seed}`;
        if (k0 !== key) {
          key = k0; res = ddim(bank, gaussArr(bank.D, s.seed), s.steps, { record: true }); shownK = -1;
          const cols = [...Array(NT)].map((_, j) => Math.round((j * s.steps) / (NT - 1)));
          thumbs.forEach((p, j) => { p.put(res.xs[cols[j]]); p.k = cols[j]; });
        }
        if (s.play) {
          if (s.prog >= 1) { hold += dt; if (hold > 1.6) { hold = 0; s.prog = 0; } }
          else s.prog = Math.min(1, s.prog + dt / 4);
        }
        const k = kOf(s);
        if (k !== shownK) {
          shownK = k;
          noisy.put(res.xs[k]);
          guess.put(res.x0s[Math.min(k, s.steps - 1)]);
          bd.redraw(s);
        }
        thumbs.forEach((p) => { const on = p.k <= k; p.mesh.material.color.setScalar(on ? 1 : 0.25); p.glow(0xffd166, p.k === k ? 0.8 : 0); });
        marker.position.x = -2.55 + (k / s.steps) * (NT - 1) * 0.85;
      },
      readout(s) {
        if (!res) return '';
        const k = kOf(s), t = 1 - k / s.steps, top = topWeights(res.ws[Math.min(k, s.steps - 1)], 1)[0];
        return `<div class="big">Step ${k} of ${s.steps}</div>
          <div class="row"><span>Noise strength</span><b>${Math.round(sigmaOf(t) * 100)}%</b></div>
          <div class="row"><span>Picture strength</span><b>${Math.round(alphaOf(t) * 100)}%</b></div>
          <div class="row"><span>Guess leans on one picture</span><b>${Math.round(top[0] * 100)}%</b></div>
          <small>${bank.N} training pictures, ${IMG}×${IMG} pixels. Exact denoiser, DDIM steps.</small>`;
      },
    };
  },
};
