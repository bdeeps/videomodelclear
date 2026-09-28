// Chapter 4: text to video, with classifier-free guidance (Ho & Salimans 2021/2022).
//  - The prompt is two words (a colour and a shape). The toy "text encoder" is a lookup: 3 + 3 numbers.
//    Real models use a large pretrained text encoder; Imagen Video (Ho et al. 2022) reported using T5-XXL.
//  - The conditional denoiser is the exact posterior mean under a soft prompt: pictures that match the
//    prompt get a prior weight e^1.5 (about 4.5×), the rest weight 1. That models a network that has
//    only partly learned what words mean, which is the situation guidance was invented for.
//  - Classifier-free guidance: guess = unconditional + w × (conditional − unconditional). w = 0 ignores
//    the prompt, w = 1 is the plain conditional model, w > 1 pushes further in the prompt's direction.
//    (Guidance in "noise space", as in the paper, is the same thing: noise and x0 guesses are linearly
//    related at a fixed step.) Real systems often use w around 5 to 7 (Stable Diffusion's default is 7.5).
//  - Contact sheet: the same prompt from 16 different noise seeds, or the same seed with 9 prompts.
//    Each tile is a full 12-step DDIM run, computed one tile per frame.
import { THREE, M, box, arrow } from '../kit.js';
import { imageBank, ddim, gaussArr, match, picture, tile, board, panel, txt, F, COL, fit, tallView, IMG, CNAMES, SHAPES, COLOURS } from '../vid.js';

const BETA = 1.5, SHEET_STEPS = 12;
const promptText = (c, sh) => (c === 'any' && sh === 'any' ? '(no prompt)' : `a ${c === 'any' ? '' : c + ' '}${sh === 'any' ? 'shape' : sh}`);
const wq = (s) => Math.round(s.w * 2) / 2;
const PAIRS = CNAMES.flatMap((c) => SHAPES.map((sh) => [c, sh]));

export default {
  id: 'prompt',
  short: 'Text to video',
  title: 'Words steer the noise',
  subtitle: 'A prompt nudges every denoising step. Guidance decides how hard.',
  view: { pos: [0.3, 3.8, 9.4], target: [0.1, 1.75, 0.3] },
  learn: `<p>So far the model makes <i>something</i>. To make what you ask for, the prompt has to take part in every step. First a <b>text encoder</b>, a language model like the ones in LLMClear, turns your words into a list of numbers. The denoiser reads those numbers alongside the noisy picture, usually through attention. Training pairs millions of videos with their captions, so the network learns which pictures go with which words.</p>
    <p>That link is never perfect. So models use a trick called <b>classifier-free guidance</b>. At each step the network makes two guesses: one <b>with</b> the prompt and one <b>without</b>. The difference between them is "what the prompt adds". Guidance multiplies that difference by a number <b>w</b> and adds it back: guess = without + w × (with − without).</p>
    <p>Our toy's text encoder is just a lookup for two words, and its grasp of them is deliberately weak, like a partly trained model. At <b>w = 0</b> it ignores you. At <b>w = 1</b> it follows the prompt only some of the time. Push w higher and it obeys more often. Real models usually run at about 5 to 7. Too high and pictures turn harsh and samey, with less variety.</p>
    <p>The <b>seed</b> picks the starting noise. Same seed and same prompt give the same result every time. Same seed, different prompt: a different result, often with a similar layout, because the noise already leans towards certain places.</p>
    <p class="tip"><b>Try it:</b> ask for "a blue square" at w = 0, 1, 3 and 6, and count how many of the 16 seeds on the sheet obey. Then switch the sheet to "same seed, 9 prompts".</p>`,
  terms: [
    { t: 'Prompt', d: 'The text you give the model to describe what you want.' },
    { t: 'Text encoder', d: 'A language model that turns the prompt into numbers the denoiser can use.' },
    { t: 'Conditioning', d: 'Feeding extra information, like the prompt, into every denoising step.' },
    { t: 'Classifier-free guidance', d: 'Mixing guesses made with and without the prompt to follow it more strongly.' },
    { t: 'Guidance scale (w)', d: 'How strongly to push towards the prompt. 0 ignores it; higher obeys more.' },
    { t: 'Seed', d: 'The number that fixes the starting noise, so a result can be repeated.' },
  ],
  defaults: { colour: 'blue', shape: 'square', w: 3, seed: 16, prog: 1, play: true, sheet: 'seeds' },
  controls: [
    { key: 'colour', type: 'seg', label: 'Prompt: colour', options: [...CNAMES, 'any'].map((v) => ({ v, label: v })) },
    { key: 'shape', type: 'seg', label: 'Prompt: shape', options: [...SHAPES, 'any'].map((v) => ({ v, label: v })), fmt: (v, s) => `"${promptText(s.colour, v)}"` },
    { key: 'w', type: 'range', label: 'Guidance scale w', min: 0, max: 8, step: 0.5, fmt: (v) => (v === 0 ? '0: ignore the prompt' : v === 1 ? '1: plain conditional' : `${v}`) },
    { key: 'seed', type: 'range', label: 'Noise seed', min: 1, max: 40, step: 1, fmt: (v) => `#${v}` },
    { key: 'play', type: 'toggle', label: 'Play the denoising' },
    { key: 'prog', type: 'range', label: 'Progress', min: 0, max: 1, step: 0.001, fmt: (v) => `step ${Math.round(v * 20)} of 20` },
    { key: 'sheet', type: 'seg', label: 'Contact sheet', options: [{ v: 'seeds', label: '16 seeds, this prompt' }, { v: 'prompts', label: 'This seed, 9 prompts' }] },
  ],
  quiz: [
    { q: 'What does the text encoder do?', options: ['Draws the first frame', 'Turns the prompt into numbers the denoiser can read', 'Adds noise', 'Checks spelling'], answer: 1, why: 'The denoiser cannot read words, so a language model turns them into vectors first.' },
    { q: 'In classifier-free guidance, what happens at w = 0?', options: ['The prompt is followed perfectly', 'The prompt is ignored', 'No picture is made', 'The video plays backwards'], answer: 1, why: 'guess = without + 0 × (with − without) = the guess without the prompt.' },
    { q: 'Same seed, different prompt. What do you get?', options: ['Exactly the same picture', 'A different result from the same starting noise', 'An error', 'A blank frame'], answer: 1, why: 'The seed fixes the starting noise, but the prompt steers every step differently.' },
  ],
  reel: [
    { ms: 5400, caption: 'A prompt steers every denoising step. Here: "a blue square".', set: { colour: 'blue', shape: 'square', w: 3, seed: 16, sheet: 'seeds', play: false }, anim: { prog: [0, 1] }, spin: 0 },
    { ms: 5400, caption: 'Guidance turns the prompt up: at w = 0 it is ignored, higher and it obeys.', set: { colour: 'blue', shape: 'square', seed: 16, sheet: 'seeds', play: false, prog: 1 }, anim: { w: [0, 6] }, spin: 0.2 },
  ],

  build({ stage }) {
    const bank = imageBank();
    const root = new THREE.Group(); stage.root.add(root);
    // prompt → text encoder → numbers
    const pTile = tile('a blue square', 2.1, 0.46, COL.hot, 96); pTile.position.set(-3.35, 3.2, 0); root.add(pTile);
    const enc = box(1.1, 0.5, 0.5, M.plastic(0xc49bff, { emissive: 0x2a1850 })); enc.position.set(-3.35, 2.35, 0); root.add(enc);
    const vTile = tile('[0 1 0 · 0 1 0]', 2.1, 0.4, COL.violet, 80); vTile.position.set(-3.35, 1.55, 0); root.add(vTile);
    const downs = [[-3.35, 2.82], [-3.35, 1.95]].map(([x, y]) => { const a = arrow(0x8ef0ff, 0.26, 0.12, 0.025); a.rotation.z = Math.PI; a.position.set(x, y + 0.13, 0.1); root.add(a); return a; });
    const into = arrow(0x8ef0ff, 0.55, 0.18, 0.03); into.rotation.z = -Math.PI / 2; into.position.set(-2.25, 1.55, 0.1); root.add(into);
    // 9:16 video frames: stack the prompt chain above the picture and pull the sheet in
    let wasTall = false;
    const layout = (tall) => {
      const cx = tall ? -1.15 : -3.35, top = tall ? 5.55 : 3.2;
      pTile.position.set(cx, top, 0); enc.position.set(cx, top - 0.85, 0); vTile.position.set(cx, top - 1.65, 0);
      downs[0].position.set(cx, top - 0.25, 0.1); downs[1].position.set(cx, top - 1.12, 0.1);
      if (tall) { into.rotation.z = Math.PI; into.position.set(cx, top - 1.95, 0.1); cur.group.position.set(cx, 2.2, 0); }
      else { into.rotation.z = -Math.PI / 2; into.position.set(-2.25, 1.55, 0.1); cur.group.position.set(-0.55, 2.2, 0); }
      sheetX = tall ? 1.45 : 2.55; place(queue.length || 16);
      L.p.position.set(cx, top + 0.45, 0); L.e.position.set(cx, top - 0.85, 0.4); L.c.position.set(cur.group.position.x, 3.45, 0); L.s.position.set(sheetX, 3.62, 0);
    };
    // the current denoising run
    const cur = picture(IMG, IMG, 2.0, 2.0, { edge: 0.07 }); cur.group.position.set(-0.55, 2.2, 0); root.add(cur.group);
    // contact sheet
    const sheet = [...Array(16)].map((_, j) => { const p = picture(IMG, IMG, 0.52, 0.52, { edge: 0.035 }); root.add(p.group); return p; });
    let sheetX = 2.55;
    const place = (n) => { const c = n === 9 ? 3 : 4, sz = n === 9 ? 0.7 : 0.58; sheet.forEach((p, j) => { p.group.visible = j < n; p.group.scale.setScalar(sz / 0.58); p.group.position.set(sheetX + ((j % c) - (c - 1) / 2) * (sz + 0.08), 2.2 + ((c - 1) / 2 - Math.floor(j / c)) * (sz + 0.08), 0); }); };
    // the guidance board: without, with, guided
    const bd = board(1400, 470, 5.6, 1.88, (g, w, h, s) => s && res && drawBoard(g, w, h, s)); bd.mesh.position.set(0, 0.42, 2.2); bd.mesh.rotation.x = -0.98; root.add(bd.mesh);
    const L = {
      p: stage.label('your prompt', [-3.35, 3.65, 0], root, 'hot'),
      e: stage.label('text encoder', [-3.35, 2.35, 0.4], root),
      c: stage.label('this seed, step by step', [-0.55, 3.45, 0], root),
      s: stage.label('contact sheet', [2.55, 3.62, 0], root),
    };
    let key = '', res = null, shown = -1, hold = 0, skey = '', queue = [], results = [], mode = '';

    const prior = (c, sh) => { const f = match(c, sh), p = new Float32Array(bank.N); if (c === 'any' && sh === 'any') return null; for (let i = 0; i < bank.N; i++) p[i] = f(bank.meta[i]) ? BETA : 0; return p; };
    const obeys = (x, c, sh) => match(c, sh)(bank.meta[bank.nearest(x).i]);
    function thumbAt(g, x, px, X, Y) {
      const c = document.createElement('canvas'); c.width = c.height = IMG; const cg = c.getContext('2d'), im = cg.createImageData(IMG, IMG);
      for (let p = 0; p < IMG * IMG; p++) { for (let q = 0; q < 3; q++) im.data[p * 4 + q] = Math.max(0, Math.min(255, (x[p * 3 + q] + 1) * 127.5)); im.data[p * 4 + 3] = 255; }
      cg.putImageData(im, 0, 0); g.imageSmoothingEnabled = false; g.drawImage(c, X, Y, px, px);
      g.strokeStyle = 'rgba(255,255,255,.3)'; g.lineWidth = 3; g.strokeRect(X - 1.5, Y - 1.5, px + 3, px + 3);
    }
    function drawBoard(g, w, h, s) {
      panel(g, w, h);
      const k = Math.min(19, Math.round(s.prog * 20)), xu = res.xus[k], xc = res.xcs[k], xg = res.x0s[k];
      txt(g, `THE MODEL'S GUESSES AT STEP ${Math.min(20, Math.round(s.prog * 20))}`, 30, 50, F(28, true), COL.soft);
      const px = 220, y = 80;
      thumbAt(g, xu, px, 40, y); txt(g, 'without prompt', 40 + px / 2, y + px + 44, F(28, true), COL.white, 'center');
      txt(g, '+ w ×', 330, y + px / 2 + 12, F(40, true), COL.hot, 'center');
      txt(g, '(', 400, y + px / 2 + 20, F(90), COL.soft, 'center');
      thumbAt(g, xc, px, 430, y); txt(g, 'with prompt', 430 + px / 2, y + px + 44, F(28, true), COL.white, 'center');
      txt(g, '−', 690, y + px / 2 + 14, F(56, true), COL.soft, 'center');
      thumbAt(g, xu, px, 730, y); txt(g, 'without', 730 + px / 2, y + px + 44, F(28, true), COL.white, 'center');
      txt(g, ')', 975, y + px / 2 + 20, F(90), COL.soft, 'center');
      txt(g, '=', 1030, y + px / 2 + 14, F(56, true), COL.soft, 'center');
      thumbAt(g, xg, px, 1090, y); txt(g, `guided, w = ${wq(s)}`, 1090 + px / 2, y + px + 44, F(28, true), COL.hot, 'center');
      txt(g, 'Colours past the normal range are clipped: that harshness is what very high guidance does.', 30, h - 26, F(24), COL.soft);
    }

    return {
      update(dt, s) {
        dt = Math.max(0, dt);
        fit(stage, [L.e, L.p, L.c], { x: 0.07, y: -0.08, ny: -0.14 });
        const tall = tallView(stage, 1.3, 0.68, 0.15);
        if (tall !== wasTall) { wasTall = tall; layout(tall); }
        const pk = `${s.colour}|${s.shape}`, W = wq(s);
        const k0 = `${pk}|${W}|${s.seed}`;
        if (k0 !== key) {
          key = k0;
          res = ddim(bank, gaussArr(bank.D, s.seed), 20, { prior: prior(s.colour, s.shape) || new Float32Array(bank.N), w: W, record: true });
          shown = -1;
          const txtP = promptText(s.colour, s.shape);
          pTile.redraw(txtP, COL.hot);
          vTile.redraw(`[${CNAMES.map((c) => (c === s.colour ? 1 : 0)).join(' ')} · ${SHAPES.map((c) => (c === s.shape ? 1 : 0)).join(' ')}]`, COL.violet);
        }
        const sk = s.sheet === 'seeds' ? `seeds|${pk}|${W}` : `prompts|${s.seed}|${W}`;
        if (sk !== skey) {
          skey = sk; mode = s.sheet; results = [];
          queue = s.sheet === 'seeds' ? [...Array(16)].map((_, j) => ({ seed: j + 1, c: s.colour, sh: s.shape })) : PAIRS.map(([c, sh]) => ({ seed: s.seed, c, sh }));
          place(queue.length); sheet.forEach((p) => { p.put(new Float32Array(IMG * IMG * 3).fill(-0.8)); p.glow(0, 0); });
        }
        // one full sampling run per frame for the sheet (all of them at once in the video)
        const budget = document.body.classList.contains('gb-reel') ? 16 : 1;
        for (let n = 0; n < budget && results.length < queue.length; n++) {
          const q = queue[results.length], j = results.length;
          const r = ddim(bank, gaussArr(bank.D, q.seed), SHEET_STEPS, { prior: prior(q.c, q.sh) || new Float32Array(bank.N), w: W });
          const ok = obeys(r.x, q.c, q.sh); results.push(ok);
          sheet[j].put(r.x); sheet[j].glow(ok ? 0x7be08c : 0xff5a8a, 0.55);
        }
        if (s.play) {
          if (s.prog >= 1) { hold += dt; if (hold > 1.6) { hold = 0; s.prog = 0; } }
          else s.prog = Math.min(1, s.prog + dt / 3.5);
        }
        const k = Math.round(s.prog * 20);
        if (k !== shown) { shown = k; cur.put(res.xs[k]); bd.redraw(s); }
      },
      readout(s) {
        if (!res) return '';
        const done = Math.round(s.prog * 20) >= 20, ok = obeys(res.x, s.colour, s.shape), n = results.filter(Boolean).length;
        return `<div class="big">"${promptText(s.colour, s.shape)}", w = ${wq(s)}</div>
          <div class="row"><span>Seed #${s.seed} gives</span><b>${done ? (ok ? 'what you asked for ✓' : 'something else ✗') : 'denoising…'}</b></div>
          <div class="row"><span>${mode === 'seeds' ? '16 seeds that obey' : '9 prompts obeyed'}</span><b>${results.length < queue.length ? `${n} so far…` : `${n} of ${queue.length}`}</b></div>
          <small>Green frame: matches its prompt. Red: it does not. The toy's word knowledge is deliberately weak.</small>`;
      },
    };
  },
};
