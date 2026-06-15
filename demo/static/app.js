"use strict";

// ----------------------------------------------------------------------- //
// Pipeline wiring (source step -> target step). Used both to draw the SVG
// connectors and to light them up as stages complete.
// ----------------------------------------------------------------------- //
const EDGES = [
  ["input", "vessel_seg"],
  ["vessel_seg", "graph"],
  ["vessel_seg", "masked"],
  ["graph", "graph_branch"],
  ["masked", "pixel_branch"],
  ["graph_branch", "fusion"],
  ["pixel_branch", "fusion"],
  ["fusion", "result"],
];

const $ = (id) => document.getElementById(id);
const cardOf = (step) => $("card-" + step);

let imageDataURI = null;
let caseName = "upload";
let running = false;

// ----------------------------------------------------------------------- //
// Input: drag & drop + file picker.
// ----------------------------------------------------------------------- //
const drop = $("drop");
const fileInput = $("file");
const preview = $("preview");

drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("drag");
  if (e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) loadFile(fileInput.files[0]);
});

function loadFile(file) {
  if (!file.type.startsWith("image/")) { log("Not an image file.", "err"); return; }
  caseName = file.name.replace(/\.[^.]+$/, "") || "upload";
  const reader = new FileReader();
  reader.onload = () => {
    imageDataURI = reader.result;
    preview.src = imageDataURI;
    drop.classList.add("has-img");
    $("start").disabled = false;
    log("Loaded " + file.name);
  };
  reader.readAsDataURL(file);
}

// ----------------------------------------------------------------------- //
// Parameter sliders.
// ----------------------------------------------------------------------- //
const lam = $("lam"), thr = $("thr");
lam.addEventListener("input", () => $("lamVal").textContent = (+lam.value).toFixed(2));
thr.addEventListener("input", () => $("thrVal").textContent = (+thr.value).toFixed(2));

// ----------------------------------------------------------------------- //
// Start → stream the pipeline.
// ----------------------------------------------------------------------- //
$("start").addEventListener("click", start);

async function start() {
  if (running || !imageDataURI) return;
  running = true;
  resetCards();
  const btn = $("start");
  btn.classList.add("running");
  btn.textContent = "● Running…";
  btn.disabled = true;
  $("summary").hidden = true;

  try {
    const resp = await fetch("/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: imageDataURI,
        caseid: caseName,
        lam: +lam.value,
        threshold: +thr.value,
      }),
    });
    if (!resp.ok || !resp.body) {
      const t = await resp.text().catch(() => "");
      throw new Error("server " + resp.status + " " + t);
    }
    await consume(resp.body);
  } catch (err) {
    log("Request failed: " + err.message, "err");
  } finally {
    running = false;
    btn.classList.remove("running");
    btn.textContent = "▶ Start inference";
    btn.disabled = false;
  }
}

// Parse the SSE-framed stream ("data: {json}\n\n") chunk by chunk.
async function consume(stream) {
  const reader = stream.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (line) handleEvent(JSON.parse(line.slice(5).trim()));
    }
  }
}

// ----------------------------------------------------------------------- //
// Event handling.
// ----------------------------------------------------------------------- //
function handleEvent(ev) {
  switch (ev.stage) {
    case "meta":
      if (ev.colorbar) $("cbar").src = ev.colorbar;
      log(`λ=${ev.lam}  τ=${ev.threshold}  ${ev.num_folds}-fold ensemble`);
      break;
    case "status":
      setRunning(ev.step);
      log("→ " + ev.message);
      break;
    case "result":
      setResult(ev.step, ev.image, ev.caption);
      break;
    case "done":
      $("sNodes").textContent = ev.num_nodes;
      $("sPos").textContent = ev.positive;
      $("sParams").textContent = `${ev.lam} / ${ev.threshold}`;
      $("summary").hidden = false;
      log(`Done — ${ev.positive}/${ev.num_nodes} collateral nodes`, "ok");
      break;
    case "error":
      log("ERROR: " + ev.message, "err");
      break;
  }
}

function setRunning(step) {
  const card = cardOf(step);
  if (!card) return;
  card.classList.add("running");
  card.classList.remove("done");
  edgesTo(step).forEach((e) => e.classList.add("active"));
}

function setResult(step, image, caption) {
  const card = cardOf(step);
  if (!card) return;
  card.classList.remove("running");
  card.classList.add("done");
  if (image) {
    const thumb = card.querySelector(".thumb");
    let img = thumb.querySelector("img");
    if (!img) { img = document.createElement("img"); thumb.appendChild(img); }
    img.src = image;
    thumb.classList.add("has-img");
  }
  if (caption !== undefined) card.querySelector(".cap").textContent = caption || "";
  edgesTo(step).forEach((e) => { e.classList.remove("active"); e.classList.add("done"); });
}

function resetCards() {
  document.querySelectorAll(".card").forEach((c) => {
    c.classList.remove("running", "done");
    const thumb = c.querySelector(".thumb");
    thumb.classList.remove("has-img");
    const img = thumb.querySelector("img");
    if (img) img.remove();
    c.querySelector(".cap").textContent = "";
  });
  document.querySelectorAll(".wires path").forEach((p) => p.classList.remove("active", "done"));
}

// ----------------------------------------------------------------------- //
// SVG connectors between cards (recomputed on layout changes).
// ----------------------------------------------------------------------- //
const svg = $("wires");
const SVGNS = "http://www.w3.org/2000/svg";
const wirePaths = {}; // "from->to" -> <path>

function edgeKey(a, b) { return a + "->" + b; }
function edgesTo(step) {
  return EDGES.filter(([, t]) => t === step).map(([s, t]) => wirePaths[edgeKey(s, t)]).filter(Boolean);
}

function buildWires() {
  svg.innerHTML = "";
  for (const [a, b] of EDGES) {
    const path = document.createElementNS(SVGNS, "path");
    path.dataset.edge = edgeKey(a, b);
    svg.appendChild(path);
    wirePaths[edgeKey(a, b)] = path;
  }
  layoutWires();
}

function layoutWires() {
  const canvas = $("canvas");
  const base = canvas.getBoundingClientRect();
  for (const [a, b] of EDGES) {
    const ca = cardOf(a), cb = cardOf(b);
    if (!ca || !cb) continue;
    const ra = ca.getBoundingClientRect(), rb = cb.getBoundingClientRect();
    const x1 = ra.right - base.left + canvas.scrollLeft;
    const y1 = ra.top + ra.height / 2 - base.top + canvas.scrollTop;
    const x2 = rb.left - base.left + canvas.scrollLeft;
    const y2 = rb.top + rb.height / 2 - base.top + canvas.scrollTop;
    const mx = (x1 + x2) / 2;
    // Smooth cubic elbow.
    const d = `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
    wirePaths[edgeKey(a, b)].setAttribute("d", d);
  }
}

window.addEventListener("resize", layoutWires);
$("canvas").addEventListener("scroll", layoutWires);

// ----------------------------------------------------------------------- //
// Log helper.
// ----------------------------------------------------------------------- //
function log(msg, cls) {
  const el = $("log");
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

// init
buildWires();
window.addEventListener("load", layoutWires);
