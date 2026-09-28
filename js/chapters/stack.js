// Chapter 1: a video is a stack of images. A real 1-second clip (48×27 pixels, a bouncing ball) is
// rendered here at the chosen frame rate and stacked into a space-time cube you can slice: an x-y slice
// is one frame; an x-t slice (one pixel row through every frame) shows motion as a path, the
// "slit-scan" view used in video research.
// Numbers per second, uncompressed, 8 bits per colour: width × height × 3 × fps.
//  - 1080p at 24 fps: 1920 × 1080 × 3 × 24 = 149,299,200 numbers per second (≈ 149 MB/s, ≈ 9 GB per minute).
//  - Sound, CD quality: 44,100 samples per second × 2 channels = 88,200 numbers per second (Red Book CD audio).
//  - Reading: about 238 words per minute for silent reading of non-fiction in English (Brysbaert 2019,
//    meta-analysis), and about 1.3 tokens per English word (OpenAI's rule of thumb: 1 token ≈ 0.75 words),
//    so about 5 tokens per second.
//  - Streaming video is compressed: Netflix suggests about 5 Mbit/s for 1080p HD (help.netflix.com, "Internet
//    connection speed recommendations").
// 24 fps became the standard for sound film in the late 1920s (see FilmClear).
import { THREE, M, box } from '../kit.js';
import { renderFrame, picture, board, panel, txt, F, COL, fit, tallView, fmtBig, inReel } from '../vid.js';

const W = 48, H = 27, FW = 3.2, FH = 1.8, DEPTH = 4.2, Y0 = 1.45;
const RES = { 480: [854, 480, '480p (SD)'], 720: [1280, 720, '720p (HD)'], 1080: [1920, 1080, '1080p (Full HD)'], 2160: [3840, 2160, '4K (Ultra HD)'] };
const FLOOR = 22.5;
function frameAt(t) {
  // A ball thrown left to right, bouncing on the floor: a real parabola, g scaled to the picture.
  const x = 5 + 38 * t, r = 3.2, ph = (t * 2.2) % 1, hgt = 15 * 4 * ph * (1 - ph);
  const img = renderFrame(W, [{ shape: 'circle', colour: 'yellow', x, y: FLOOR - r - hgt, r }], H);
  for (let j = Math.ceil(FLOOR); j < H; j++) for (let i = 0; i < W; i++) { const p = (j * W + i) * 3; img[p] = -0.52; img[p + 1] = -0.5; img[p + 2] = -0.38; }
  const sx = Math.round(x), sw = Math.max(1, 3 - hgt / 6);           // soft shadow under the ball
  for (let i = Math.round(sx - sw); i <= sx + sw; i++) if (i >= 0 && i < W) { const p = (Math.ceil(FLOOR) * W + i) * 3; img[p] = img[p + 1] = img[p + 2] = -0.72; }
  return img;
}
const perSec = (w, h, fps) => w * h * 3 * fps;

export default {
  id: 'stack',
  short: 'A stack of images',
  title: 'A video is a stack of images',
  subtitle: 'Pictures over time make a block of numbers. And that block is huge.',
  view: { pos: [0.5, 4.9, 8.4], target: [0.5, 1.2, -0.9] },
  learn: `<p>A video is not one moving thing. It is a quick run of still pictures, called <b>frames</b>. Show them fast enough and your brain sees smooth motion. Cinema settled on <b>24 frames per second</b> (fps) when sound arrived in the late 1920s, and it still uses it (see FilmClear). TV and phones often use 25, 30 or 60.</p>
    <p>Stack the frames one behind another and you get a block: two directions of space and one of time. This is the <b>space-time cube</b>, and it is exactly what an AI video model has to make. Cut the cube across and you get one frame. Cut it along time, one pixel row through every frame, and the ball's motion becomes a curved <b>path</b>.</p>
    <p>Each frame is a grid of <b>pixels</b>, and each pixel is 3 numbers: red, green and blue. So one second of Full HD video at 24 fps is 1920 × 1080 × 3 × 24 ≈ <b>149 million numbers</b>. Reading this page, you take in about 5 tokens of text a second. That gap, millions of times, is why making video is so much harder than making text (see LLMClear for tokens).</p>
    <p>Real files are compressed, so a 1080p stream is closer to 5 megabits a second. But a model has to decide every pixel of every frame, and keep them all agreeing with each other.</p>
    <p class="tip"><b>Try it:</b> drop the frame rate to 12 fps and watch the player stutter. Then switch the slice to "row over time" and drag the row up and down through the ball's path.</p>`,
  terms: [
    { t: 'Frame', d: 'One still picture in a video.' },
    { t: 'Frame rate (fps)', d: 'How many frames are shown each second. Cinema uses 24.' },
    { t: 'Pixel', d: 'One dot of a picture, stored as three numbers: red, green and blue.' },
    { t: 'Resolution', d: 'How many pixels wide and tall each frame is, such as 1920 × 1080.' },
    { t: 'Space-time cube', d: 'All the frames stacked in order: width, height and time.' },
    { t: 'Compression', d: 'Storing a video in fewer bits by skipping what repeats, like a still background.' },
  ],
  defaults: { fps: 24, res: 1080, t: 0.4, play: true, slice: 'both', row: 14 },
  controls: [
    { key: 'fps', type: 'seg', label: 'Frame rate', options: [12, 24, 30, 60].map((v) => ({ v, label: v + ' fps' })), fmt: (v) => (v === 24 ? 'cinema' : v === 12 ? 'choppy' : v === 60 ? 'games, sport' : 'TV, phones') },
    { key: 'res', type: 'seg', label: 'Resolution (for the numbers)', options: Object.entries(RES).map(([v, r]) => ({ v: +v, label: v === '2160' ? '4K' : v + 'p' })), fmt: (v) => `${RES[v][0]} × ${RES[v][1]}` },
    { key: 'play', type: 'toggle', label: 'Play' },
    { key: 't', type: 'range', label: 'Time in the clip', min: 0, max: 0.999, step: 0.001, fmt: (v, s) => `frame ${Math.floor(v * s.fps) + 1} of ${s.fps}` },
    { key: 'slice', type: 'seg', label: 'Slice the cube', options: [{ v: 'frame', label: 'One frame' }, { v: 'row', label: 'Row over time' }, { v: 'both', label: 'Both' }] },
    { key: 'row', type: 'range', label: 'Which row', min: 0, max: H - 1, step: 1, fmt: (v) => `row ${v + 1} of ${H}` },
  ],
  quiz: [
    { q: 'What is a video, to a computer?', options: ['One moving picture', 'A run of still frames, each a grid of numbers', 'A list of words', 'A sound wave'], answer: 1, why: 'Each frame is a grid of pixels, and each pixel is three numbers for red, green and blue.' },
    { q: 'About how many numbers make one second of uncompressed 1080p video at 24 fps?', options: ['About 5', 'About 88,000', 'About 149 million', 'About 1 trillion'], answer: 2, why: '1920 × 1080 pixels × 3 colours × 24 frames ≈ 149 million.' },
    { q: 'In the space-time cube, what does a slice along time (one row through every frame) show?', options: ['A single frame', 'How things in that row move over time', 'The sound', 'Nothing useful'], answer: 1, why: 'Motion turns into a path. A still object would make a straight stripe.' },
  ],
  reel: [
    { ms: 5200, caption: 'A video is a stack of still pictures: 24 of them every second in cinema.', set: { fps: 24, res: 1080, slice: 'frame', play: true, row: 14 }, spin: 0.35 },
    { ms: 5200, caption: 'One second of Full HD is about 149 million numbers. Text is a few words.', set: { fps: 24, res: 1080, slice: 'both', play: true }, anim: { row: [8, 20] }, spin: 0.25, view: { pos: [5.0, 5.4, 6.4], target: [0.3, 1.2, -1.6] } },
  ],

  build({ stage }) {
    const root = new THREE.Group(); stage.root.add(root);
    const cube = new THREE.Group(); root.add(cube);
    // the outline of the space-time cube
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(FW, FH, DEPTH)), new THREE.LineBasicMaterial({ color: 0x8ef0ff, transparent: true, opacity: 0.45 }));
    edges.position.set(0, Y0, -DEPTH / 2); cube.add(edges);
    // up to 60 frames; the chosen rate shows `fps` of them across 1 second of depth
    const frames = [...Array(60)].map(() => {
      const p = picture(W, H, FW, FH, { edge: 0.02 });
      p.back.visible = false;
      p.mesh.material.transparent = true; p.mesh.material.depthWrite = false;
      cube.add(p.group); return p;
    });
    const cur = picture(W, H, FW, FH, { edge: 0.05, color: 0xffd166 }); cube.add(cur.group); cur.glow(0xffd166, 0.6);
    // the x-t slice: one pixel row from every frame, laid flat through the cube
    const sliceC = document.createElement('canvas'); sliceC.width = W; sliceC.height = 60;
    const sliceTex = new THREE.CanvasTexture(sliceC); sliceTex.colorSpace = THREE.SRGBColorSpace; sliceTex.magFilter = THREE.NearestFilter; sliceTex.generateMipmaps = false; sliceTex.minFilter = THREE.LinearFilter;
    const slice = new THREE.Mesh(new THREE.PlaneGeometry(FW, DEPTH), new THREE.MeshBasicMaterial({ map: sliceTex, toneMapped: false, side: THREE.DoubleSide }));
    slice.rotation.x = -Math.PI / 2; slice.position.z = -DEPTH / 2; cube.add(slice);
    const rim = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(FW, DEPTH)), new THREE.LineBasicMaterial({ color: 0xff7a59 }));
    rim.rotation.x = -Math.PI / 2; rim.position.z = -DEPTH / 2; cube.add(rim);
    // time arrow along the floor
    const tArrow = box(0.04, 0.04, DEPTH + 0.6, M.glow(0x8ef0ff)); tArrow.position.set(FW / 2 + 0.25, 0.3, -DEPTH / 2 - 0.3); cube.add(tArrow);
    const tip = new THREE.Mesh(new THREE.ConeGeometry(0.1, 0.3, 16), M.glow(0x8ef0ff)); tip.rotation.x = -Math.PI / 2; tip.position.set(FW / 2 + 0.25, 0.3, -DEPTH - 0.7); cube.add(tip);
    // the player: what you see on a screen
    const player = picture(W, H, 2.3, 2.3 * H / W, { edge: 0.08, color: 0x1a1f2b }); player.group.position.set(3.2, 1.55, -1.2); player.group.rotation.y = -0.35; root.add(player.group);
    const stand = box(0.12, 0.8, 0.12, M.metal(0x5a6272)); stand.position.set(3.2, 0.4, -1.3); root.add(stand);
    // the numbers-per-second board
    const bd = board(1200, 520, 4.4, 1.9, (g, w, h, s) => s && drawBoard(g, w, h, s)); bd.mesh.position.set(0.9, 0.62, 2.0); bd.mesh.rotation.x = -1.05; root.add(bd.mesh);
    const L = {
      t: stage.label('time →', [FW / 2 + 0.35, 0.55, -DEPTH - 0.4], root),
      f: stage.label('one frame', [0, Y0 + FH / 2 + 0.3, 0], root),
      r: stage.label('one row over time', [-FW / 2 - 0.2, Y0, -DEPTH / 2], root, 'hot'),
      p: stage.label('what you see', [3.2, 2.45, -1.2], root),
    };
    let key = '', imgs = [], shown = -1, rowKey = '';

    function drawBoard(g, w, h, s) {
      panel(g, w, h);
      const [rw, rh, rn] = RES[s.res], you = perSec(rw, rh, s.fps);
      txt(g, 'NUMBERS PER SECOND (log scale: each step is ×10)', 28, 50, F(26, true), COL.soft);
      const rows = [
        ['Reading text', 5, '≈ 5 tokens', COL.teal],
        ['CD-quality sound', 88200, '88,200', COL.violet],
        [`This tiny clip, ${W}×${H} at ${s.fps} fps`, perSec(W, H, s.fps), perSec(W, H, s.fps).toLocaleString('en'), COL.orange],
        [`${rn} at ${s.fps} fps`, you, fmtBig(you), COL.hot],
      ];
      const x0 = 470, span = w - x0 - 210, L10 = (v) => (Math.log10(v) / 9) * span;
      rows.forEach(([name, v, lab, col], i) => {
        const y = 110 + i * 92;
        txt(g, name, 28, y + 8, F(28, true), COL.white);
        g.fillStyle = col; g.fillRect(x0, y - 22, Math.max(4, L10(v)), 40);
        txt(g, lab, x0 + Math.max(4, L10(v)) + 12, y + 8, F(26, true, true), col);
      });
      txt(g, 'Uncompressed, 3 numbers (red, green, blue) per pixel.', 28, h - 22, F(22), COL.soft);
    }

    return {
      update(dt, s, time) {
        dt = Math.max(0, dt);
        fit(stage, [L.r, L.t], { x: 0.06, y: -0.07, ny: -0.14 });
        const tall = tallView(stage, 1.0, 0.5, -0.4);
        if (tall) { player.group.position.set(0, 4.0, -1.4); player.group.rotation.y = 0; stand.visible = false; L.p.position.set(0, 4.95, -1.4); }
        else { player.group.position.set(3.2, 1.55, -1.2); player.group.rotation.y = -0.35; stand.visible = true; L.p.position.set(3.2, 2.45, -1.2); }
        if (s.play) { s.t = (s.t + dt * 0.5) % 1; }       // half speed, so each frame is easy to see
        const k = `${s.fps}|${s.res}`;
        if (k !== key) {
          const reframe = !key || +key.split('|')[0] !== s.fps; key = k;
          if (reframe) {
            imgs = [...Array(s.fps)].map((_, f) => frameAt(f / s.fps));
            frames.forEach((p, f) => { p.group.visible = f < s.fps; if (f < s.fps) { p.put(imgs[f]); p.group.position.set(0, Y0, -(f / (s.fps - 1)) * DEPTH); } });
            shown = -1; rowKey = '';
          }
          bd.redraw(s);
        }
        const f = Math.min(s.fps - 1, Math.floor(s.t * s.fps));
        const sl = s.slice;
        frames.forEach((p, i) => { if (i < s.fps) p.mesh.material.opacity = sl === 'row' ? 0.07 : i === f ? 0 : 0.14; });
        if (f !== shown) { shown = f; cur.put(imgs[f]); player.put(imgs[f]); }
        cur.group.position.set(0, Y0, -(f / (s.fps - 1)) * DEPTH); cur.group.visible = sl !== 'row';
        L.f.position.set(0, Y0 + FH / 2 + 0.25, cur.group.position.z);
        L.f.visible = sl !== 'row';
        const rk = `${s.row}|${s.fps}`;
        if (rk !== rowKey) {
          rowKey = rk;
          sliceC.height = s.fps;
          const g = sliceC.getContext('2d'), im = g.createImageData(W, s.fps);
          for (let fr = 0; fr < s.fps; fr++) for (let i = 0; i < W; i++) {
            const src = (s.row * W + i) * 3, dst = ((s.fps - 1 - fr) * W + i) * 4, a = imgs[fr];
            im.data[dst] = (a[src] + 1) * 127.5; im.data[dst + 1] = (a[src + 1] + 1) * 127.5; im.data[dst + 2] = (a[src + 2] + 1) * 127.5; im.data[dst + 3] = 255;
          }
          g.putImageData(im, 0, 0); sliceTex.dispose(); sliceTex.needsUpdate = true;
        }
        const ry = Y0 + FH / 2 - ((s.row + 0.5) / H) * FH;
        slice.position.y = ry; rim.position.y = ry + 0.004; slice.visible = rim.visible = sl !== 'frame';
        L.r.position.set(-FW / 2 - 0.25, ry, -DEPTH / 2); L.r.visible = sl !== 'frame' && stage.host.clientWidth >= 560;
        if (inReel()) L.p.visible = true;
      },
      readout(s) {
        const [rw, rh, rn] = RES[s.res], n = perSec(rw, rh, s.fps);
        return `<div class="big">${fmtBig(n)} numbers a second</div>
          <div class="row"><span>${rn}, one frame</span><b>${(rw * rh).toLocaleString('en')} pixels</b></div>
          <div class="row"><span>One minute, uncompressed</span><b>${((n * 60) / 1e9).toFixed(1)} GB</b></div>
          <div class="row"><span>Compared with reading text</span><b>${fmtBig(n / 5)} ×</b></div>
          <small>1 byte per number. Streaming compresses this hundreds of times.</small>`;
      },
    };
  },
};
