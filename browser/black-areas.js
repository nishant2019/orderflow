// ==UserScript==
// @name         Black Areas Detector (footprint POC boxes)
// @namespace    orderflow
// @version      0.1.1
// @description  Finds solid-black and black-outlined rectangles (e.g. the POC box of a footprint chart) on the page's
//               charts, many times a second, and outlines them. Everything runs in your browser; nothing is sent anywhere.
// @match        https://gocharting.com/terminal/*
// @match        https://*.gocharting.com/terminal/*
// @run-at       document-idle
// @grant        none
// ==/UserScript==
//
// Use it as a userscript (Tampermonkey / Violentmonkey: add the site you trade on to @match), as a bookmarklet, or paste
// the whole file into the browser console. API:  window.blackAreas.{start,stop,scan,last,onChange,options}
//
(function (root) {
  "use strict";

  const DEFAULTS = {
    maxChannel: 64,      // a pixel is "black" when R, G and B are all below this (0-255)
    minWidth: 24,        // px (in the scanned image) of the smallest area reported
    minHeight: 8,
    minFill: 0.55,       // black pixels / box area at or above this: a solid area
    minSideLength: 6,    // px a vertical side must stay black below a frame's top edge (the frame is taller than this)
    maxEdgeThickness: 8, // px: thicker horizontal black bars are solid areas, not frame edges
    edgeGap: 16,         // px: a gap (candle, price line) that may cut a frame's edge in two
    maxOutlineHeight: 90,// px: tallest frame (gap between its top and bottom edge)
    rowStep: 1,          // 1 = every row. (2 is ~25% faster but misses a frame edge that is only 1 px thick where a price line covers it)
    minRun: 2,           // ignore black runs shorter than this inside a row (anti-aliasing specks)
    roi: null,           // {x, y, w, h} limit the scan to a region (e.g. skip a page header)
    scale: 1,            // 0.5 scans a half-size copy: ~4x faster, boxes are scaled back
    intervalMs: 250,     // how often the page mode rescans
    minCanvasWidth: 400, minCanvasHeight: 200,
  };

  // ---------------------------------------------------------------------------------------------
  // The detector: pure function of pixels. Black runs per row -> union-find across rows -> boxes.
  // ---------------------------------------------------------------------------------------------
  function detectBlackAreas(data, width, height, opts) {
    const o = Object.assign({}, DEFAULTS, opts || {});
    const t = o.maxChannel;
    // "all channels < t" for t a power of two is one AND on the packed pixel: no channel of the mask bits may be set
    const pow2 = (t & (t - 1)) === 0;
    const m8 = (~(t - 1)) & 0xff;
    const mask = (m8 | (m8 << 8) | (m8 << 16)) >>> 0;
    const px = new Uint32Array(data.buffer, data.byteOffset, width * height);
    const step = Math.max(1, o.rowStep | 0);
    const x0 = o.roi ? Math.max(0, o.roi.x | 0) : 0;
    const y0 = o.roi ? Math.max(0, o.roi.y | 0) : 0;
    const x1 = o.roi ? Math.min(width, (o.roi.x + o.roi.w) | 0) : width;
    const y1 = o.roi ? Math.min(height, (o.roi.y + o.roi.h) | 0) : height;

    // run storage (grows as needed)
    let cap = 4096;
    let rs = new Int32Array(cap), re = new Int32Array(cap), ry = new Int32Array(cap), parent = new Int32Array(cap);
    let n = 0;
    const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
    const union = (a, b) => { a = find(a); b = find(b); if (a !== b) parent[b] = a; };

    let prevStart = 0, prevEnd = 0; // runs of the previous row are rs[prevStart..prevEnd)
    for (let y = y0; y < y1; y += step) {
      const rowStart = n;
      let i = (y * width + x0) * 4, runFrom = -1, pi = y * width + x0;
      for (let x = x0; x <= x1; x++, i += 4, pi++) {
        // transparent pixels (0,0,0,0) of a layered canvas are not black: require a mostly opaque pixel
        const black = x < x1 && (pow2 ? ((px[pi] & mask) === 0 && (px[pi] & 0x80000000) !== 0) : (data[i] < t && data[i + 1] < t && data[i + 2] < t && data[i + 3] > 127));
        if (black) { if (runFrom < 0) runFrom = x; }
        else if (runFrom >= 0) {
          if (x - runFrom >= o.minRun) {
            if (n === cap) {
              cap *= 2;
              const g = (a) => { const b = new Int32Array(cap); b.set(a); return b; };
              rs = g(rs); re = g(re); ry = g(ry); parent = g(parent);
            }
            rs[n] = runFrom; re[n] = x; ry[n] = y; parent[n] = n; n++;
          }
          runFrom = -1;
        }
      }
      // join this row's runs to the previous row's runs they overlap (two-pointer sweep)
      let p = prevStart;
      for (let c = rowStart; c < n; c++) {
        while (p < prevEnd && re[p] <= rs[c]) p++;
        for (let q = p; q < prevEnd && rs[q] < re[c]; q++) union(c, q);
      }
      prevStart = rowStart; prevEnd = n;
    }

    // collect components
    const comp = new Map();
    for (let r = 0; r < n; r++) {
      const root = find(r);
      let c = comp.get(root);
      if (!c) { c = { x0: rs[r], x1: re[r], y0: ry[r], y1: ry[r] + step, area: 0 }; comp.set(root, c); }
      if (rs[r] < c.x0) c.x0 = rs[r];
      if (re[r] > c.x1) c.x1 = re[r];
      if (ry[r] < c.y0) c.y0 = ry[r];
      if (ry[r] + step > c.y1) c.y1 = ry[r] + step;
      c.area += (re[r] - rs[r]) * step;
    }

    const isBlack = (x, y) => { const i = (y * width + x) * 4; return data[i] < t && data[i + 1] < t && data[i + 2] < t && data[i + 3] > 127; };

    const out = [];
    for (const c of comp.values()) {
      const w = c.x1 - c.x0, h = c.y1 - c.y0;
      if (w < o.minWidth || h < o.minHeight) continue;
      const fill = c.area / (w * h);
      if (fill >= o.minFill) out.push({ x: c.x0, y: c.y0, w, h, kind: "solid", fill: +fill.toFixed(3) });
    }

    // ---- outlines (hollow rectangles, e.g. the POC box drawn as a black frame) -----------------------------------
    // Neighbouring frames touch and merge into one blob, and a price line may cross a side, so a whole-blob test
    // fails. Instead: find the long horizontal edges, pair a top edge with the bottom edge below it that has the same
    // extent, check the sides, and split the pair wherever a vertical line crosses its inside.
    const edges = [];   // {x0, x1, y0, y1}: a thin horizontal black bar at least minWidth long
    let open = [];      // edges still growing downwards
    const longRuns = [];
    for (let r = 0; r < n; r++) if (re[r] - rs[r] >= o.minWidth) longRuns.push(r);
    for (const r of longRuns) {
      let e = open.find((q) => ry[r] - q.lastY <= step && Math.abs(q.x0 - rs[r]) <= 3 && Math.abs(q.x1 - re[r]) <= 3);
      if (e) { e.y1 = ry[r] + step; e.lastY = ry[r]; }
      else { e = { x0: rs[r], x1: re[r], y0: ry[r], y1: ry[r] + step, lastY: ry[r] }; open.push(e); edges.push(e); }
      if (open.length > 64) open = open.filter((q) => ry[r] - q.lastY <= step);
    }
    // a candle or price line can cut an edge into pieces on the same rows: join pieces with a small gap
    edges.sort((p, q) => p.y0 - q.y0 || p.x0 - q.x0);
    const joined = [];
    for (const e of edges) {
      const m = joined.find((q) => Math.abs(q.y0 - e.y0) <= 2 && Math.abs(q.y1 - e.y1) <= 2 && e.x0 - q.x1 <= o.edgeGap && e.x0 >= q.x0);
      if (m) { m.x1 = Math.max(m.x1, e.x1); m.y0 = Math.min(m.y0, e.y0); m.y1 = Math.max(m.y1, e.y1); } else joined.push(Object.assign({}, e));
    }
    const thin = joined.filter((e) => e.y1 - e.y0 <= o.maxEdgeThickness);
    thin.sort((p, q) => p.y0 - q.y0);
    if (o.debug) out.debugEdges = thin;
    // From each horizontal edge look straight down: columns that stay black for at least minHeight rows are the sides
    // of a frame hanging from that edge. Neighbouring sides pair up into frames; a bottom edge, when visible, gives the
    // height (it can be hidden under a price line, so it is optional).
    const seen = new Set();
    for (const top of thin) {
      const lines = [];
      for (let x = top.x0; x < top.x1; x++) {
        let len = 0;
        while (len < o.maxOutlineHeight && top.y1 + len < y1 && isBlack(x, top.y1 + len)) len++;
        if (len >= o.minSideLength) {
          const l = lines[lines.length - 1];
          if (l && x - l.x1 <= 1) { l.x1 = x + 1; l.len = Math.max(l.len, len); } else lines.push({ x0: x, x1: x + 1, len });
        }
      }
      const sides = lines.filter((l) => l.x1 - l.x0 >= 2);
      for (let k = 0; k + 1 < sides.length; k++) {
        const L = sides[k], R = sides[k + 1];
        const w = R.x1 - L.x0;
        if (w < o.minWidth) continue;
        // bottom edge below that spans this frame, if any
        let bottom = null;
        for (const q of thin) {
          if (q.y0 > top.y1 + 2 && q.y0 <= top.y1 + o.maxOutlineHeight && q.x0 <= L.x0 + 4 && q.x1 >= R.x1 - 4) { if (!bottom || q.y0 < bottom.y0) bottom = q; }
        }
        const hgt = bottom ? bottom.y1 - top.y0 : Math.min(L.len, R.len) + (top.y1 - top.y0);
        if (hgt < o.minHeight) continue;
        const key = (L.x0 >> 2) + "," + (top.y0 >> 2) + "," + (w >> 2);
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({ x: L.x0, y: top.y0, w, h: hgt, kind: "outline", fill: 0 });
      }
    }
    out.sort((a, b) => a.y - b.y || a.x - b.x);
    return out;
  }

  // scan a canvas / video / image element (draws it to a scratch canvas first: works for 2D and WebGL canvases)
  const scratch = { c: null, g: null };
  function grab(source, scale) {
    const sw = source.videoWidth || source.naturalWidth || source.width;
    const sh = source.videoHeight || source.naturalHeight || source.height;
    const w = Math.max(1, Math.round(sw * scale)), h = Math.max(1, Math.round(sh * scale));
    if (!scratch.c) { scratch.c = document.createElement("canvas"); scratch.g = scratch.c.getContext("2d", { willReadFrequently: true }); }
    if (scratch.c.width !== w || scratch.c.height !== h) { scratch.c.width = w; scratch.c.height = h; }
    scratch.g.clearRect(0, 0, w, h); // the scratch canvas is shared: without this, the previous canvas shows through transparent areas
    scratch.g.drawImage(source, 0, 0, w, h);
    return { image: scratch.g.getImageData(0, 0, w, h), w, h, sw, sh };
  }

  function scanSource(source, opts) {
    const o = Object.assign({}, DEFAULTS, opts || {});
    const t0 = performance.now();
    const { image, w, h, sw, sh } = grab(source, o.scale);
    const k = o.scale;
    const sc = Object.assign({}, o, {
      minWidth: Math.max(2, Math.round(o.minWidth * k)), minHeight: Math.max(2, Math.round(o.minHeight * k)),
      roi: o.roi ? { x: o.roi.x * k, y: o.roi.y * k, w: o.roi.w * k, h: o.roi.h * k } : null,
    });
    const boxes = detectBlackAreas(image.data, w, h, sc).map((b) => ({
      x: Math.round(b.x / k), y: Math.round(b.y / k), w: Math.round(b.w / k), h: Math.round(b.h / k), kind: b.kind, fill: b.fill,
    }));
    return { boxes, ms: performance.now() - t0, width: sw, height: sh };
  }

  const api = { detectBlackAreas, scanSource, DEFAULTS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.BlackAreasCore = api;
})(typeof window !== "undefined" ? window : globalThis);

// -----------------------------------------------------------------------------------------------------------------
// Page mode: find the chart canvases, scan them on a timer, outline what was found, show a small HUD.
// -----------------------------------------------------------------------------------------------------------------
(function (root) {
  "use strict";
  if (typeof document === "undefined") return;
  const core = root.BlackAreasCore;
  if (root.blackAreas && root.blackAreas.running) return; // already installed on this page

  const state = {
    running: false, timer: null, options: Object.assign({}, core.DEFAULTS), last: [], listeners: [],
    overlays: new Map(), hud: null, mode: "page", stream: null, video: null, lastMs: 0, busy: false, error: null,
  };

  const visibleCanvases = () => Array.from(document.querySelectorAll("canvas")).filter((c) => {
    if (c.dataset.blackAreasOverlay || c.dataset.blackAreasPreview) return false; // never scan our own drawings
    const r = c.getBoundingClientRect();
    return r.width >= state.options.minCanvasWidth && r.height >= state.options.minCanvasHeight &&
      r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth && getComputedStyle(c).visibility !== "hidden";
  });

  function overlayFor(source, rect) {
    let ov = state.overlays.get(source);
    if (!ov) {
      ov = document.createElement("canvas");
      ov.dataset.blackAreasOverlay = "1";
      ov.style.cssText = "position:fixed;pointer-events:none;z-index:2147483646;";
      document.body.appendChild(ov);
      state.overlays.set(source, ov);
    }
    ov.style.left = rect.left + "px"; ov.style.top = rect.top + "px";
    ov.style.width = rect.width + "px"; ov.style.height = rect.height + "px";
    if (ov.width !== Math.round(rect.width)) ov.width = Math.round(rect.width);
    if (ov.height !== Math.round(rect.height)) ov.height = Math.round(rect.height);
    return ov;
  }

  function draw(ov, boxes, sx, sy) {
    const g = ov.getContext("2d");
    g.clearRect(0, 0, ov.width, ov.height);
    g.lineWidth = 2; g.font = "11px system-ui, sans-serif"; g.textBaseline = "top";
    for (const b of boxes) {
      const x = b.x * sx, y = b.y * sy, w = b.w * sx, h = b.h * sy;
      g.strokeStyle = b.kind === "solid" ? "#ff2bd6" : "#00e5ff";
      g.strokeRect(x - 2, y - 2, w + 4, h + 4);
      const label = b.kind + " " + b.w + "×" + b.h;
      const tw = g.measureText(label).width + 6;
      g.fillStyle = g.strokeStyle; g.fillRect(x - 2, y - 15, tw, 13);
      g.fillStyle = "#000"; g.fillText(label, x + 1, y - 14);
    }
  }

  function emit(all) {
    state.last = all;
    for (const f of state.listeners) { try { f(all); } catch (e) { console.error(e); } }
    document.dispatchEvent(new CustomEvent("blackareas", { detail: all }));
  }

  function tick() {
    if (!state.running || state.busy || document.hidden) return;
    state.busy = true;
    const t0 = performance.now();
    const all = [];
    try {
      if (state.mode === "screen") {
        const v = state.video;
        if (v && v.videoWidth) {
          const r = core.scanSource(v, state.options);
          r.boxes.forEach((b) => all.push(Object.assign({ source: "screen" }, b)));
          drawPreview(v, r.boxes);
        }
      } else {
        const seen = new Set();
        for (const c of visibleCanvases()) {
          const rect = c.getBoundingClientRect();
          const r = core.scanSource(c, state.options);
          seen.add(c);
          const sx = rect.width / r.width, sy = rect.height / r.height;
          draw(overlayFor(c, rect), r.boxes, sx, sy);
          r.boxes.forEach((b) => all.push(Object.assign({
            source: "canvas", canvas: c,
            pageX: Math.round(rect.left + b.x * sx + scrollX), pageY: Math.round(rect.top + b.y * sy + scrollY),
            pageW: Math.round(b.w * sx), pageH: Math.round(b.h * sy),
          }, b)));
        }
        for (const [c, ov] of state.overlays) if (!seen.has(c)) { ov.remove(); state.overlays.delete(c); }
      }
      state.error = null;
    } catch (e) {
      // a tainted (cross-origin) or blank canvas cannot be read: say so and offer the screen-capture route
      state.error = e && e.name === "SecurityError" ? "this page's chart canvas is protected: use “Capture tab”" : String(e);
    }
    state.lastMs = performance.now() - t0;
    state.busy = false;
    emit(all);
    renderHud();
  }

  // ---- screen capture fallback (the browser asks which tab to share) -----------------------------------------------
  let preview = null;
  function drawPreview(video, boxes) {
    if (!preview) {
      preview = document.createElement("canvas");
      preview.dataset.blackAreasPreview = "1";
      preview.style.cssText = "position:fixed;right:12px;bottom:12px;width:420px;border:2px solid #ff2bd6;z-index:2147483646;background:#000;pointer-events:none;";
      document.body.appendChild(preview);
    }
    preview.width = video.videoWidth; preview.height = video.videoHeight;
    const g = preview.getContext("2d");
    g.drawImage(video, 0, 0);
    g.lineWidth = Math.max(2, video.videoWidth / 500); g.strokeStyle = "#ff2bd6";
    for (const b of boxes) g.strokeRect(b.x - 2, b.y - 2, b.w + 4, b.h + 4);
  }
  async function captureTab() {
    const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false, preferCurrentTab: true });
    const v = document.createElement("video");
    v.srcObject = stream; v.muted = true; await v.play();
    state.stream = stream; state.video = v; state.mode = "screen";
    stream.getVideoTracks()[0].addEventListener("ended", () => { state.mode = "page"; state.stream = state.video = null; if (preview) { preview.remove(); preview = null; } });
    for (const ov of state.overlays.values()) ov.remove();
    state.overlays.clear();
  }

  // ---- HUD -----------------------------------------------------------------------------------------------------
  function renderHud() {
    if (!state.hud) {
      const host = document.createElement("div");
      host.style.cssText = "position:fixed;top:10px;right:10px;z-index:2147483647;";
      const sh = host.attachShadow({ mode: "open" });
      sh.innerHTML = `<style>
        div{font:12px/1.4 system-ui,sans-serif;background:#101418e6;color:#e8edf3;border:1px solid #2c3540;border-radius:8px;padding:8px 10px;min-width:190px}
        b{color:#ff2bd6} button{font:inherit;margin:4px 4px 0 0;background:#1d2530;color:inherit;border:1px solid #394656;border-radius:6px;padding:2px 8px;cursor:pointer}
        .e{color:#ffb454}</style><div><div id="t"></div><div id="e" class="e"></div>
        <button id="p">Pause</button><button id="c">Capture tab</button><button id="j">Copy JSON</button></div>`;
      sh.getElementById("p").onclick = (e) => { state.running ? api.stop(true) : api.start(); e.target.textContent = state.running ? "Pause" : "Resume"; };
      sh.getElementById("c").onclick = () => captureTab().catch((e) => { state.error = String(e); renderHud(); });
      sh.getElementById("j").onclick = () => navigator.clipboard.writeText(JSON.stringify(state.last.map(({ canvas, ...r }) => r), null, 1));
      document.body.appendChild(host);
      state.hud = sh;
    }
    const sh = state.hud;
    sh.getElementById("t").innerHTML = `<b>${state.last.length}</b> black area${state.last.length === 1 ? "" : "s"} · ${state.lastMs.toFixed(1)} ms · ${state.mode}`;
    sh.getElementById("e").textContent = state.error || "";
  }

  // ---- public API ----------------------------------------------------------------------------------------------
  const api = {
    get running() { return state.running; },
    get last() { return state.last; },
    get options() { return state.options; },
    set options(o) { Object.assign(state.options, o); },
    start() { if (state.running) return; state.running = true; state.timer = setInterval(tick, state.options.intervalMs); tick(); },
    stop(keepHud) {
      state.running = false; clearInterval(state.timer);
      if (!keepHud) { for (const ov of state.overlays.values()) ov.remove(); state.overlays.clear(); if (state.hud) state.hud.host.remove(); state.hud = null; }
    },
    scan() { const was = state.running; state.running = true; tick(); state.running = was; return state.last; },
    onChange(f) { state.listeners.push(f); return () => state.listeners.splice(state.listeners.indexOf(f), 1); },
    captureTab,
  };
  root.blackAreas = api;
  api.start();
})(typeof window !== "undefined" ? window : globalThis);
