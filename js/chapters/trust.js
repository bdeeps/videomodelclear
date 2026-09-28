// Chapter 6: cost, trust and use.
//  - Watermark lab: a real, classic spread-spectrum watermark (Cox, Kilian, Leighton & Shamoon 1997): a
//    secret ±1 pattern, one value per pixel, is added to the frame at a strength of a few 8-bit levels.
//    The detector high-pass filters the brightness (pixel minus its 3×3 average), correlates it with the
//    secret pattern and divides by the expected spread, giving a z-score: about 0 ± 1 for an unmarked
//    frame; above 4 counts as "found". It survives noise and re-saving with fewer shades, but a 3×3 blur
//    or a 1-pixel shift breaks it. Production watermarks (e.g. Google DeepMind's SynthID, launched for
//    images Aug 2023 and extended to Veo video in May 2024) are designed to survive many common edits.
//  - Content credentials: a toy manifest signed over a hash of the pixels (32-bit FNV-1a here; C2PA uses
//    SHA-256 hashes and X.509 digital signatures). Any pixel edit breaks the match, and stripping the
//    metadata removes it entirely. C2PA was announced on 22 Feb 2021 by Adobe, Arm, BBC, Intel, Microsoft
//    and Truepic (c2pa.org).
//  - Energy (MIT Technology Review, "We did the math on AI's energy footprint", 20 May 2025, measured on
//    open models, total energy incl. estimated overheads): text reply from Llama 3.1 8B ≈ 114 J, from
//    Llama 3.1 405B ≈ 6,706 J; one 1024×1024 image from Stable Diffusion 3 Medium ≈ 2,282 J; a
//    5-second video from CogVideoX ≈ 3.4 million J (0.94 kWh). Delavande, Pierrard & Luccioni, "Video
//    Killed the Energy Budget" (arXiv:2509.19222, Sept 2025): video generation energy grows roughly with
//    the square of both length and resolution. Closed commercial models do not publish such figures.
//  - India (hedged, as reported): MeitY advisory of 26 Dec 2023 on deepfakes; IT (Intermediary Guidelines
//    and Digital Media Ethics Code) Amendment Rules, 2026, notified 10 Feb 2026, in force 20 Feb 2026:
//    "synthetically generated information" must carry a prominent label; platforms must not let labels or
//    metadata be removed; takedown within 3 hours of a lawful order (2 hours for non-consensual intimate
//    imagery). Sources: MeitY gazette PDF, All India Radio (newsonair.gov.in), Khaitan & Co.
import { THREE, M, box } from '../kit.js';
import { renderFrame, rng, gaussArr, picture, tile, board, panel, txt, F, COL, fit, tallView } from '../vid.js';

const S = 48, N = S * S, THRESH = 4;
const FRAME = renderFrame(S, [{ shape: 'circle', colour: 'yellow', x: 34, y: 12, r: 6 }, { shape: 'square', colour: 'blue', x: 14, y: 30, r: 7 }, { shape: 'triangle', colour: 'red', x: 32, y: 34, r: 8 }]);
const KEY = (() => { const r = rng(2024); return new Float32Array(N).map(() => (r() < 0.5 ? -1 : 1)); })();
const lum = (y) => { const l = new Float32Array(N); for (let i = 0; i < N; i++) l[i] = (y[i * 3] + y[i * 3 + 1] + y[i * 3 + 2]) / 3; return l; };
function box3(l) {
  const o = new Float32Array(N);
  for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
    let s = 0; for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) s += l[Math.min(S - 1, Math.max(0, j + b)) * S + Math.min(S - 1, Math.max(0, i + a))];
    o[j * S + i] = s / 9;
  }
  return o;
}
function detect(y) { const l = lum(y), b = box3(l); let num = 0, den = 0; for (let i = 0; i < N; i++) { const h = l[i] - b[i]; num += h * KEY[i]; den += h * h; } return num / Math.sqrt(den || 1e-9); }
function embed(lv) { const y = FRAME.slice(), a = (lv * 2) / 255; for (let i = 0; i < N; i++) for (let c = 0; c < 3; c++) y[i * 3 + c] += a * KEY[i]; return y; }
const EDITS = {
  none: { label: 'No edit', f: (y) => y },
  noise: { label: 'Add grain', f: (y) => { const g = gaussArr(y.length, 9); return y.map((v, i) => v + g[i] * (16 / 255)); } },
  resave: { label: 'Re-save', f: (y) => y.map((v) => (Math.round(((v + 1) / 2) * 31) / 31) * 2 - 1) },
  blur: { label: 'Blur', f: (y) => { const o = y.slice(); for (let c = 0; c < 3; c++) { const l = new Float32Array(N); for (let i = 0; i < N; i++) l[i] = y[i * 3 + c]; const b = box3(l); for (let i = 0; i < N; i++) o[i * 3 + c] = b[i]; } return o; } },
  shift: { label: 'Shift 1 px', f: (y) => { const o = y.slice(); for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) for (let c = 0; c < 3; c++) o[(j * S + i) * 3 + c] = y[(j * S + Math.max(0, i - 1)) * 3 + c]; return o; } },
};
function fnv(y) { let h = 0x811c9dc5; for (let i = 0; i < y.length; i++) { const b = Math.max(0, Math.min(255, Math.round((y[i] + 1) * 127.5))); h ^= b; h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(16).padStart(8, '0'); }
const ENERGY = [
  ['Text reply, small model (Llama 3.1 8B)', 114, COL.teal],
  ['Picture, 1024 × 1024 (SD3 Medium)', 2282, COL.violet],
  ['Text reply, big model (Llama 3.1 405B)', 6706, COL.teal],
  ['5-second video (CogVideoX)', 3.4e6, COL.hot],
];

export default {
  id: 'trust',
  short: 'Cost, trust and use',
  title: 'Cost, trust and good use',
  subtitle: 'Video is the most expensive thing AI makes, and the easiest to mistake for real.',
  view: { pos: [0.3, 3.7, 9.2], target: [0.2, 1.75, 0.3] },
  learn: `<p><b>Cost.</b> Video is the heaviest thing these models make. In 2025, MIT Technology Review measured open models: one 5-second video took about <b>3.4 million joules</b> (roughly 1 kWh), against about 2,300 to 4,400 J for one picture and roughly 100 to 7,000 J for a text reply. A Hugging Face study found the energy grows roughly with the <b>square</b> of the length: twice as long, about four times the energy. Big commercial models do not publish their figures, so treat all of these as estimates. Running models is its own topic (see InferenceClear, being built).</p>
    <p><b>Trust.</b> A realistic fake video of a real person is a <b>deepfake</b>. Making one without the person's <b>consent</b> can hurt them badly, and it is illegal in many places. In India, a viral deepfake of an actor in November 2023 led MeitY to issue an advisory to platforms that December. Amended IT Rules, reported as notified in February 2026, require AI-made audio and video to carry a prominent <b>label</b>, and platforms to remove unlawful content within 3 hours of an official order. The EU and China have labelling rules too.</p>
    <p><b>Two tools help.</b> A <b>watermark</b> hides a faint signal in the pixels that a detector can find, like Google DeepMind's SynthID. <b>Content credentials</b> (the C2PA standard, started in 2021 by Adobe, Microsoft, the BBC and others) attach a signed record of how a file was made. Neither is perfect: watermarks can be weakened, and credentials can be stripped. A video with no label is not proof it is real.</p>
    <p><b>Use.</b> In film, these tools help with <b>storyboards and previs</b>, rough moving sketches before a shoot (see StoryboardClear), concept ideas and some effects work (see VFXClear), with shots still edited and graded by people (see GradeClear). They differ from <b>CGI rendering</b>, which computes light in a 3D scene exactly as built (see CGIClear): a video model predicts pixels it thinks are likely. That makes it fast and flexible, but harder to control. People are still debating training on artists' work without permission, and the effect on jobs.</p>
    <p class="tip"><b>Try it:</b> add a watermark, then try each edit and watch the detector. Turn off the credentials to see what a stripped file looks like. A blur or a 1-pixel shift defeats this simple watermark.</p>`,
  terms: [
    { t: 'Deepfake', d: 'A realistic fake video or voice of a real person, made with AI.' },
    { t: 'Consent', d: 'Clear permission from a person before their face or voice is used.' },
    { t: 'Watermark', d: 'A hidden signal in the pixels that a detector can later find.' },
    { t: 'Content credentials', d: 'A signed record attached to a file saying how it was made and edited (C2PA).' },
    { t: 'Provenance', d: 'Where a piece of media came from and what happened to it since.' },
    { t: 'Previs', d: 'Rough moving versions of scenes, made before filming to plan them.' },
  ],
  defaults: { lv: 3, edit: 'none', creds: true },
  controls: [
    { key: 'lv', type: 'range', label: 'Watermark strength', min: 0, max: 6, step: 1, fmt: (v) => (v ? `±${v} of 255 levels` : 'off') },
    { key: 'edit', type: 'seg', label: 'Then someone edits the frame', options: Object.entries(EDITS).map(([v, e]) => ({ v, label: e.label })) },
    { key: 'creds', type: 'toggle', label: 'Keep the content credentials' },
  ],
  quiz: [
    { q: 'About how much more energy did a 5-second AI video take than one AI picture, in MIT Technology Review\'s 2025 tests?', options: ['About the same', 'About 10 times', 'More than 700 times', 'Less'], answer: 2, why: 'About 3.4 million joules against a few thousand for a picture. The reporters put it at more than 700 times.' },
    { q: 'A video has no AI label and no content credentials. What does that prove?', options: ['It is real', 'It is fake', 'Nothing either way', 'It was made in India'], answer: 2, why: 'Labels and credentials can be missing or stripped. Their absence proves nothing.' },
    { q: 'What makes a deepfake harmful even if it is labelled?', options: ['Its file size', 'Using a real person\'s face or voice without their consent', 'Its frame rate', 'Its colours'], answer: 1, why: 'Consent matters. A label does not undo the harm of putting words or actions on someone who never agreed.' },
  ],
  reel: [
    { ms: 5600, caption: 'Watermarks and content credentials help show a video was AI-made. Neither is perfect.', set: { lv: 3, creds: true }, anim: { edit: ['none', 'blur'] }, spin: 0.2 },
    { ms: 5000, caption: 'A 5-second AI video can take about a kilowatt-hour. Use it well, label it, ask consent.', set: { lv: 3, edit: 'none', creds: true }, spin: 0.15, view: { pos: [0.2, 4.6, 7.4], target: [0.2, 0.9, 1.2] } },
  ],

  build({ stage }) {
    const root = new THREE.Group(); stage.root.add(root);
    const frame = picture(S, S, 2.3, 2.3, { edge: 0.08 }); frame.group.position.set(-1.1, 2.25, 0); root.add(frame.group);
    const diff = picture(S, S, 1.2, 1.2, { edge: 0.05, color: 0x2a2440 }); diff.group.position.set(1.05, 2.8, 0); root.add(diff.group);
    const tag = tile('Content Credentials: valid', 2.5, 0.36, COL.green, 80); tag.position.set(-1.1, 0.8, 0.1); root.add(tag);
    // detector gauge: a tube that fills to the z-score, with a line at the threshold
    const gx = 2.75, gy0 = 1.0, gh = 2.4, zMax = 20;
    const tubeGlass = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, gh, 32, 1, true), M.clear(0xcfe8ff, 0.18)); tubeGlass.position.set(gx, gy0 + gh / 2, 0); root.add(tubeGlass);
    const fill = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 1, 32), M.glow(0x7be08c)); fill.position.set(gx, gy0, 0); root.add(fill);
    const line = box(0.6, 0.03, 0.03, M.glow(0xff7a59)); line.position.set(gx, gy0 + (THRESH / zMax) * gh, 0); root.add(line);
    const base = box(0.7, 0.14, 0.7, M.metal(0x5a6272)); base.position.set(gx, gy0 - 0.07, 0); root.add(base);
    const bd = board(1400, 470, 5.6, 1.88, (g, w, h) => drawEnergy(g, w, h)); bd.mesh.position.set(0.2, 0.3, 2.2); bd.mesh.rotation.x = -1.05; root.add(bd.mesh);
    const L = {
      f: stage.label('an AI-made frame', [-1.1, 3.62, 0], root),
      d: stage.label('hidden pattern × 40', [1.05, 3.6, 0], root, 'hot'),
      g: stage.label('detector', [gx, gy0 + gh + 0.25, 0], root),
      t: stage.label('found above here', [gx + 0.75, gy0 + (THRESH / zMax) * gh, 0], root),
    };
    const SIGNED = fnv(embed(3));
    let key = '', z = 0, zShow = 0, hashOk = true;

    function drawEnergy(g, w, h) {
      panel(g, w, h);
      txt(g, 'ENERGY FOR ONE ITEM (joules, log scale; open models, MIT Technology Review 2025)', 28, 48, F(25, true), COL.soft);
      const x0 = 560, span = w - x0 - 230, L10 = (v) => (Math.log10(v) / 6.6) * span;
      ENERGY.forEach(([name, v, col], i) => {
        const y = 110 + i * 80;
        txt(g, name, 28, y + 10, F(28, true), COL.white);
        g.fillStyle = col; g.fillRect(x0, y - 20, L10(v), 40);
        txt(g, v >= 1e6 ? '3.4 million J' : v.toLocaleString('en') + ' J', x0 + L10(v) + 14, y + 10, F(27, true, true), col);
      });
      txt(g, '3.4 million J ≈ 0.94 kWh. Estimates: closed models do not publish figures.', 28, h - 24, F(23), COL.soft);
    }

    return {
      update(dt, s) {
        dt = Math.max(0, dt);
        fit(stage, [L.t], { x: 0.08, y: -0.12, ny: -0.14 });
        tallView(stage, 0.5, 0.6, 0.2);
        const edit = EDITS[s.edit] ? s.edit : 'none';
        const k = `${s.lv}|${edit}|${s.creds}`;
        if (k !== key) {
          key = k;
          const y = EDITS[edit].f(embed(s.lv));
          frame.put(y);
          const d = new Float32Array(N * 3); for (let i = 0; i < N * 3; i++) d[i] = (y[i] - FRAME[i]) * 40;
          diff.put(d);
          z = detect(y);
          hashOk = fnv(y) === SIGNED && s.lv === 3;
          tag.redraw(!s.creds ? 'No credentials found' : hashOk ? 'Content Credentials: valid ✓' : 'Credentials broken: edited', !s.creds ? COL.soft : hashOk ? COL.green : COL.red);
        }
        zShow = zShow + (z - zShow) * (1 - Math.exp(-8 * dt));
        const hh = Math.max(0.02, (Math.min(zMax, Math.max(0, zShow)) / zMax) * gh);
        fill.scale.y = hh; fill.position.y = gy0 + hh / 2;
        fill.material.color.setHex(zShow > THRESH ? 0x7be08c : 0x8ea0b8);
      },
      readout(s) {
        const found = z > THRESH;
        return `<div class="big">${found ? 'Watermark found' : 'No watermark found'}</div>
          <div class="row"><span>Detector score</span><b>${z.toFixed(1)} (found above ${THRESH})</b></div>
          <div class="row"><span>Edit</span><b>${EDITS[s.edit]?.label || 'No edit'}</b></div>
          <div class="row"><span>Credentials</span><b>${!s.creds ? 'stripped' : hashOk ? 'match the pixels' : 'do not match'}</b></div>
          <small>An unmarked frame scores about 0 ± 1. Credentials were signed for strength 3 with no edit.</small>`;
      },
    };
  },
};
