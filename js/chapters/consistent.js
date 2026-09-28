// Chapter 5: keeping it consistent. Three real ways to make an 8-frame, 16×16 clip from the same prompt,
// all with the exact denoiser and 16 DDIM steps (vid.js):
//  1. Frame by frame: an image model (trained on the 1,044 distinct frames in the clip bank) makes each
//     frame on its own, from its own noise. Every frame is a fine picture, but nothing ties them together:
//     the object jumps about and can change colour or shape between frames.
//  2. Same noise every frame: the same image model and the same noise for all 8 frames. Now every frame
//     is identical: steady, but frozen. Nothing moves.
//  3. Whole clip together: a video model (trained on the 216 clips) denoises all 8 frames as one
//     space-time block, so every frame's guess depends on every other frame. Motion comes out smooth.
// Early text-to-video work extended image models with layers that look across time (Make-A-Video, Meta
// 2022; Imagen Video, Google 2022); diffusion transformers attend over the whole space-time patch cube
// (Sora report, OpenAI 2024). Real models still slip: OpenAI's Sora page (Feb 2024) listed weaknesses such
// as physics that is not simulated accurately and objects that appear out of nowhere, e.g. a cookie
// with no bite mark after being bitten.
// Measurements use the nearest training frame to each generated frame (exact for this denoiser).
import { THREE, M, box } from '../kit.js';
import { clipBanks, ddim, gaussArr, match, picture, tile, board, panel, txt, F, COL, fit, tallView, T, CS, FD, CNAMES, SHAPES, COLOURS } from '../vid.js';

const STEPS = 16, SZ = 0.62, SP = 0.7, X0 = -3.0, PX = 3.35;
const MODES = {
  frames: { name: 'Frame by frame', sub: 'image model, new noise each frame', col: COL.red, y: 3.05 },
  same: { name: 'Same noise every frame', sub: 'image model, one noise', col: COL.violet, y: 2.05 },
  joint: { name: 'Whole clip together', sub: 'video model, one space-time block', col: COL.green, y: 1.05 },
};
const KEYS = Object.keys(MODES);

export default {
  id: 'consistent',
  short: 'Keeping it consistent',
  title: 'Keeping it consistent',
  subtitle: 'Every frame can look fine and the video can still be wrong.',
  view: { pos: [0.2, 3.3, 9.6], target: [0.1, 1.75, 0.3] },
  learn: `<p>A video model has a harder job than an image model. Every frame must look right, and every frame must also <b>agree</b> with the ones around it: the same ball, the same colour, moving smoothly. When that fails you see <b>flicker</b>, <b>morphing</b> (a thing slowly turns into something else) and things that pop in or out of existence.</p>
    <p>Here are three real ways to make an 8-frame clip of the same prompt. <b>Frame by frame</b>: each frame is made alone, from its own noise. Each frame is fine, but the ball teleports and can even change colour. <b>Same noise every frame</b>: steady, but frozen, because the same noise always gives the same picture. <b>Whole clip together</b>: a video model denoises all 8 frames as one space-time block, so each frame's guess depends on the others. The motion comes out smooth.</p>
    <p>That third way is what modern video models do, with attention across the whole space-time cube (chapter 3). Early text-to-video systems (2022) started from image models and added layers that look across time.</p>
    <p><b>Why it is still hard.</b> The model has no physics engine inside. It learned what videos usually look like, not the rules behind them. So it can get <b>hands</b> wrong, let objects pass through each other, or forget a detail after something walks in front of it. OpenAI itself listed failures like a cookie showing no bite mark after being bitten. Longer videos make it worse: more tokens, more to keep straight.</p>
    <p class="tip"><b>Try it:</b> set the prompt to "any colour, circle" and watch the top row change colour between frames. Then pick different seeds, and press "Denoise again" to watch all three rows form at once.</p>`,
  terms: [
    { t: 'Temporal consistency', d: 'Things staying the same from frame to frame unless they should change.' },
    { t: 'Flicker', d: 'Small, fast changes between frames that should not be there.' },
    { t: 'Morphing', d: 'An object gradually turning into a different object or shape by mistake.' },
    { t: 'Joint denoising', d: 'Removing noise from all frames at once, so each frame can depend on the others.' },
    { t: 'Physics error', d: 'Motion that breaks real-world rules, like objects passing through each other.' },
  ],
  defaults: { colour: 'any', shape: 'circle', seed: 3, prog: 1, run: false, mode: 'frames', play: true },
  controls: [
    { key: 'colour', type: 'seg', label: 'Prompt: colour', options: ['any', ...CNAMES].map((v) => ({ v, label: v })) },
    { key: 'shape', type: 'seg', label: 'Prompt: shape', options: ['any', ...SHAPES].map((v) => ({ v, label: v })) },
    { key: 'seed', type: 'range', label: 'Noise seed', min: 1, max: 30, step: 1, fmt: (v) => `#${v}` },
    { key: 'go', type: 'buttons', label: 'Watch it form', items: [{ label: 'Denoise again', act: (s) => { s.prog = 0; s.run = true; } }] },
    { key: 'prog', type: 'range', label: 'Denoising progress', min: 0, max: 1, step: 0.001, fmt: (v) => `step ${Math.round(v * STEPS)} of ${STEPS}` },
    { key: 'mode', type: 'seg', label: 'Measure this row', options: KEYS.map((v) => ({ v, label: MODES[v].name })) },
    { key: 'play', type: 'toggle', label: 'Play the clips' },
  ],
  quiz: [
    { q: 'Why does making each frame separately cause flicker?', options: ['The frames are too small', 'Nothing ties one frame to the next, so each frame makes its own choices', 'The prompt is ignored', 'It uses too few colours'], answer: 1, why: 'Each frame starts from its own noise and never sees its neighbours.' },
    { q: 'What keeps the "whole clip together" row smooth?', options: ['A physics engine', 'All frames are denoised as one block, so each frame depends on the others', 'It has more colours', 'It is slower'], answer: 1, why: 'Joint denoising lets every frame see every other frame at every step.' },
    { q: 'Why do video models still make physics mistakes?', options: ['They learned what videos look like, not the rules behind them', 'They are always broken', 'Physics does not apply to video', 'Their frame rate is too high'], answer: 0, why: 'They learn patterns in pixels. Nothing inside checks gravity, solidity or counting fingers.' },
  ],
  reel: [
    { ms: 5600, caption: 'Make each frame alone and the ball teleports and changes colour: flicker.', set: { colour: 'any', shape: 'circle', seed: 3, mode: 'frames', play: true, run: false }, anim: { prog: [0, 1] }, spin: 0 },
    { ms: 5200, caption: 'Denoise the whole clip as one space-time block, and the motion comes out smooth.', set: { colour: 'any', shape: 'circle', seed: 3, mode: 'joint', play: true, prog: 1, run: false }, spin: 0.15, view: { pos: [1.2, 2.6, 8.4], target: [0.3, 1.35, 0.3] } },
  ],

  build({ stage }) {
    const { clips, frames } = clipBanks();
    const root = new THREE.Group(); stage.root.add(root);
    const rows = {};
    KEYS.forEach((k) => {
      const m = MODES[k];
      const strip = box(8 * SP + 0.2, SZ + 0.24, 0.06, M.matte(0x151a24)); strip.position.set(X0 + 3.5 * SP, m.y, -0.06); root.add(strip);
      const pics = [...Array(T)].map((_, f) => { const p = picture(CS, CS, SZ, SZ, { edge: 0.03 }); p.group.position.set(X0 + f * SP, m.y, 0); root.add(p.group); return p; });
      const player = picture(CS, CS, 0.86, 0.86, { edge: 0.05, color: 0x1a1f2b }); player.group.position.set(PX, m.y, 0.1); player.group.rotation.y = -0.2; root.add(player.group);
      const name = tile(m.name, 2.0, 0.26, m.col, 64); name.position.set(X0 + 0.7, m.y + 0.5, 0); root.add(name);
      rows[k] = { pics, player, name, strip };
    });
    const bd = board(1400, 440, 5.8, 1.82, (g, w, h, s) => s && out && drawBoard(g, w, h, s)); bd.mesh.position.set(0.2, 0.3, 2.1); bd.mesh.rotation.x = -1.02; root.add(bd.mesh);
    const L = { p: stage.label('playing', [PX, 3.75, 0.1], root), t: stage.label('frame 1 → 8', [X0 + 3.5 * SP, 3.72, 0], root) };
    let key = '', out = null, shown = -1, shownP = -1, st = null;

    function generate(s) {
      const fcond = frames.where(match(s.colour, s.shape)), ccond = clips.where(match(s.colour, s.shape));
      const res = {};
      const perFrame = (noiseSeed) => [...Array(T)].map((_, f) => ddim(frames, gaussArr(FD, noiseSeed(f)), STEPS, { cond: fcond, record: true }));
      const fr = perFrame((f) => s.seed * 100 + f), sm = perFrame(() => s.seed * 100);
      const jt = ddim(clips, gaussArr(clips.D, s.seed), STEPS, { cond: ccond, record: true });
      const frameAt = (k, step, f) => (k === 'joint' ? jt.xs[step].subarray(f * FD, (f + 1) * FD) : (k === 'frames' ? fr : sm)[f].xs[step]);
      KEYS.forEach((k) => {
        const metas = [...Array(T)].map((_, f) => frames.meta[frames.nearest(frameAt(k, STEPS, f)).i]);
        let dist = 0, acc = 0, changes = 0;
        for (let f = 1; f < T; f++) {
          dist += Math.hypot(metas[f].x - metas[f - 1].x, metas[f].y - metas[f - 1].y);
          if (metas[f].colour !== metas[f - 1].colour || metas[f].shape !== metas[f - 1].shape) changes++;
          if (f > 1) acc += Math.hypot(metas[f].x - 2 * metas[f - 1].x + metas[f - 2].x, metas[f].y - 2 * metas[f - 1].y + metas[f - 2].y);
        }
        res[k] = { metas, step: dist / (T - 1), jerk: acc / (T - 2), changes };
      });
      return { res, frameAt };
    }
    function drawBoard(g, w, h, s) {
      panel(g, w, h);
      txt(g, 'WHERE THE OBJECT IS IN EACH FRAME (dots joined in order)', 28, 46, F(26, true), COL.soft);
      KEYS.forEach((k, i) => {
        const x0 = 28 + i * 460, y0 = 70, S = 230, sc = S / CS, r = out.res[k], m = MODES[k];
        g.fillStyle = 'rgba(255,255,255,.05)'; g.fillRect(x0, y0, S, S);
        g.strokeStyle = s.mode === k ? m.col : 'rgba(255,255,255,.2)'; g.lineWidth = s.mode === k ? 4 : 2; g.strokeRect(x0, y0, S, S);
        g.strokeStyle = m.col; g.lineWidth = 3; g.beginPath();
        r.metas.forEach((o, f) => { const X = x0 + o.x * sc, Y = y0 + o.y * sc; if (f) g.lineTo(X, Y); else g.moveTo(X, Y); }); g.stroke();
        r.metas.forEach((o, f) => { g.fillStyle = COLOURS[o.colour].css; g.beginPath(); g.arc(x0 + o.x * sc, y0 + o.y * sc, f === 0 ? 11 : 8, 0, 7); g.fill(); });
        txt(g, m.name.split(' ').slice(0, 2).join(' '), x0 + S + 14, y0 + 30, F(25, true), m.col);
        txt(g, `moves ${r.step.toFixed(1)} px/frame`, x0 + S + 14, y0 + 80, F(22), COL.white);
        txt(g, `wobble ${r.jerk.toFixed(1)}`, x0 + S + 14, y0 + 115, F(22), COL.white);
        txt(g, `${r.changes} identity change${r.changes === 1 ? '' : 's'}`, x0 + S + 14, y0 + 150, F(22), r.changes ? COL.red : COL.white);
      });
      txt(g, 'Big dot: frame 1. Real clips here move 1 to 1.4 px per frame in a straight line: wobble 0.', 28, h - 26, F(23), COL.soft);
    }

    return {
      update(dt, s, time) {
        dt = Math.max(0, dt); st = s;
        fit(stage, [L.t], { x: 0.07, y: -0.09, ny: -0.14 });
        const tall = tallView(stage, 0.6, 0.74, -0.15);
        KEYS.forEach((k) => { rows[k].player.group.visible = !tall; }); L.p.visible = !tall;
        const k0 = `${s.colour}|${s.shape}|${s.seed}`;
        if (k0 !== key) { key = k0; out = generate(s); shown = -1; bd.redraw(s); }
        if (s.run) { s.prog = Math.min(1, s.prog + dt / 3); if (s.prog >= 1) s.run = false; }
        const step = Math.round(s.prog * STEPS), p = Math.floor(time * 5) % T;
        if (step !== shown) { shown = step; shownP = -1; KEYS.forEach((k) => rows[k].pics.forEach((pc, f) => pc.put(out.frameAt(k, step, f)))); }
        if (p !== shownP && s.play) { shownP = p; KEYS.forEach((k) => rows[k].player.put(out.frameAt(k, step, p))); }
        if (!s.play && shownP < 0) { shownP = 0; KEYS.forEach((k) => rows[k].player.put(out.frameAt(k, step, 0))); }
        KEYS.forEach((k) => {
          rows[k].pics.forEach((pc, f) => pc.glow(0xffd166, s.play && f === p ? 0.8 : 0));
          rows[k].strip.material.color.setHex(s.mode === k ? 0x26324a : 0x151a24);
        });
        if (bd.mode !== s.mode) { bd.mode = s.mode; bd.redraw(s); }
      },
      readout(s) {
        if (!out) return '';
        const r = out.res[s.mode], j = out.res.joint;
        return `<div class="big">${MODES[s.mode].name}</div>
          <div class="row"><span>Moves per frame</span><b>${r.step.toFixed(1)} px</b></div>
          <div class="row"><span>Wobble (0 = smooth line)</span><b>${r.jerk.toFixed(1)}</b></div>
          <div class="row"><span>Colour or shape changes</span><b>${r.changes}</b></div>
          <small>${s.mode === 'joint' ? 'Smooth: every frame was denoised with all the others.' : s.mode === 'same' ? 'Steady but frozen: it never moves at all.' : `Compare the joint video model: wobble ${j.jerk.toFixed(1)}, ${j.changes} changes.`}</small>`;
      },
    };
  },
};
