# LMC live-inference web demo — deployment guide

An interactive, browser-based visualization of the full pipeline. Drag a DSA
frame into the page, press **Start**, and each stage renders as soon as the
backend finishes computing it:

<p align="center">
  <img src="demo.gif" alt="LMC live inference demo" width="850">
</p>

```
DSA ─▶ Vessel Seg ─▶ Graph Build ─▶ Graph Branch (GNN) ─┐
                  └▶ Masked DSA  ─▶ Pixel Branch ────────┴▶ Fusion ─▶ Result
```

| Card | What it shows |
|---|---|
| **DSA** | the uploaded frame (echoed back) |
| **Vessel Seg** | predicted vessel mask (green) over the frame |
| **Graph Build** | the vessel-segment line-graph (nodes + edges) |
| **Masked DSA** | `image × (mask > 0)` — exactly what the pixel branch ingests |
| **Graph Branch · GNN** | per-node `p_gnn` (DINOv3 ViT-L/16 + GAT), colour-coded |
| **Pixel Branch** | dense `p_nn` heatmap (vessel-masked nnU-Net) |
| **Fusion** | `p_fuse = λ·p_gnn + (1−λ)·p_nn`, colour-coded |
| **Result** | thresholded decision overlay (identical to the CLI's `<caseid>_overlay.png`) |

The backend reuses the **same runners** as `scripts/infer_one.py`
(`EndToEndInference`) — it just calls each stage in order and streams its
visualization to the page. This folder is **purely additive**: it does not
change anything under `src/lmc/` or `scripts/`. Delete `demo/` and the
inference CLI is unaffected.

---

## 1. Prerequisites

The demo has the **same requirements as CLI inference**, plus Flask:

1. **The inference environment** (conda env from the repo root):
   ```bash
   conda env create -f environment.yml
   conda activate lmc
   pip install -r requirements.txt
   ```
2. **A local clone of the DINOv3 source repo** (for the ViT-L/16 backbone):
   ```bash
   git clone https://github.com/facebookresearch/dinov3 /path/to/dinov3
   export LMC_DINOV3_REPO=/path/to/dinov3
   ```
3. **The pretrained checkpoints** under `ckpt/` (see [`../ckpt/README.md`](../ckpt/README.md)).
4. **A CUDA GPU.** The 5-fold ensemble loads ~10.5 GB of weights; ~12–16 GB
   of VRAM is comfortable. See [§7 GPU notes](#7-gpu--performance-notes).
5. **Flask** (the only extra dependency):
   ```bash
   pip install -r demo/requirements.txt
   ```

---

## 2. Launch

From the **repo root**:

```bash
export LMC_DINOV3_REPO=/path/to/dinov3
python demo/app.py                 # serves http://127.0.0.1:5000
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | bind address (`0.0.0.0` to expose on the LAN — see security note below) |
| `--port` | `5000` | TCP port |
| `--ckpt_root` | `ckpt` (from config) | checkpoint root directory |
| `--dino_repo_dir` | `$LMC_DINOV3_REPO` | DINOv3 source-repo path (overrides the env var) |
| `--device` | `cuda` | torch device |

DINOv3 path resolution order: `--dino_repo_dir` → `LMC_DINOV3_REPO` →
`configs/inference_default.yaml`. λ and τ are **not** CLI flags here — adjust
them live in the left panel before each run.

> **Note on `--host 0.0.0.0`:** this is Flask's *development* server with no
> authentication. Only bind to `0.0.0.0` on a trusted network; for remote
> access prefer SSH / tunnel forwarding (§4) over exposing the port directly.

### Run it in the background

```bash
# fully detached; logs to demo/server.log
LMC_DINOV3_REPO=/path/to/dinov3 \
  setsid python demo/app.py --port 5000 > demo/server.log 2>&1 < /dev/null &
disown
```

Stop it with `pkill -f demo/app.py`.

---

## 3. Open it (local machine)

If the browser runs on the **same machine** as the server, just open:

```
http://127.0.0.1:5000
```

Drop in `examples/sample_frame/s50_f12.png`, optionally tweak λ / τ, and press
**▶ Start inference**.

---

## 4. Open it remotely (server has the GPU, you browse from a laptop)

The server binds to `127.0.0.1` on the GPU host. Forward that port to your
local machine — **do not** expose it publicly.

### SSH port forwarding

```bash
ssh -L 5000:127.0.0.1:5000 user@gpu-host
# then open http://localhost:5000 in your local browser
```

### VSCode Remote — PORTS panel

1. Open the **PORTS** tab (next to TERMINAL).
2. **Add Port** → `5000`.
3. Click the **`Forwarded Address`** link VSCode generates (or the 🌐 globe /
   "Open in Browser" icon). **Do not type `localhost:5000` by hand** — VSCode
   may map the remote port to a *different* local port, and you must use the
   one it shows.

### ⚠️ The `localhost:5000` gotcha (macOS especially)

If forwarding "succeeds" but the page won't load and the **server log shows no
incoming request**, your **local** port 5000 is almost certainly taken by
another process — on **macOS, port 5000 is used by the AirPlay Receiver**
(Control Center). The tunnel is then shadowed by AirPlay locally and never
reaches the server.

Fix — use a non-conflicting port (e.g. `8090`):

```bash
python demo/app.py --port 8090
ssh -L 8090:127.0.0.1:8090 user@gpu-host        # or forward 8090 in VSCode
```

Ports to avoid on macOS: **5000** and **7000** (AirPlay / Control Center).
Good choices: `8080`, `8090`, `8000`, `7860`.

To confirm whether a request is even reaching the server, watch the log:

```bash
tail -f demo/server.log
# a working browser load prints: "GET / HTTP/1.1" 200, "GET /static/app.js" 200, ...
```

---

## 5. Using the demo

1. **Drag & drop** a DSA frame (PNG/JPG) into the left panel, or click to browse.
2. (Optional) adjust **λ** (fusion weight) and **τ** (decision threshold).
3. Press **▶ Start inference**.
4. Watch the pipeline diagram: the active module pulses while it computes, then
   fills with its visualization; connectors light up as data flows. The left
   panel shows a live log and a final summary (node count, collateral count,
   λ / τ).

Only **one inference runs at a time** per server process (serialized by a lock);
a second Start while one is running returns a friendly "busy" message.

---

## 6. How it works (internals)

- **Backend** ([`app.py`](app.py)) — a Flask app that holds one
  `EndToEndInference` instance. `POST /infer` (JSON: base64 image, `lam`,
  `threshold`) returns a streamed `text/event-stream` response. The handler
  runs the stages in the same order as `EndToEndInference.run_single_frame`
  and emits one **SSE event per stage** instead of returning a single dict.
- **Stage events**: `meta` → `result:input` → `status/result:vessel_seg` →
  `status/result:graph` → `result:masked` → `status/result:pixel_branch` →
  `status/result:graph_branch` → `status/result:fusion` → `result:result` →
  `done`. Each `result` carries an inline base64 PNG.
- **Visualization** ([`viz.py`](viz.py)) — pure Pillow + numpy (no matplotlib):
  vessel overlay, masked input, graph overlay, pixel heatmap, per-node score
  colouring, and the final decision overlay (byte-for-byte the CLI's overlay).
- **Frontend** ([`static/`](static/)) — vanilla HTML/CSS/JS. `app.js` reads the
  stream with `fetch()` + `ReadableStream`, parses SSE frames, fills each card,
  and redraws the SVG connectors. No build step, no framework.

> **Dev note on streaming:** browsers (`fetch` + `ReadableStream`) deliver the
> SSE bytes progressively, so each stage pops as it completes. Some CLI HTTP
> clients (`urllib`, `curl | python` pipelines) buffer and may *look* like the
> events arrive in one batch — that is a client-side artifact, not the server.
> Verify streaming with a raw socket or a real browser.

---

## 7. GPU & performance notes

- **First run is slow.** Model weights (~10.5 GB: 2× nnU-Net 5-fold + DINOv3
  ViT-L/16 + GAT 5-fold) load **lazily on the first inference**, so the first
  Start takes a few minutes. Subsequent runs reuse the in-memory models
  (~20 s end-to-end on a V100).
- **VRAM.** The 5-fold ensemble needs roughly 8–10 GB of VRAM. If you hit a
  CUDA out-of-memory error, the pipeline also supports single-fold inference
  via the CLI (`scripts/infer_one.py --mode fold:0`); the demo currently runs
  the 5-fold ensemble.
- **Throughput.** One run at a time (GPU-bound); the lock prevents concurrent
  runs from clobbering each other.

---

## 8. Files

```
demo/
├── app.py                 Flask backend: staged inference + SSE streaming
├── viz.py                 rendering helpers (Pillow + numpy)
├── requirements.txt       flask (on top of the core requirements.txt)
├── README.md              this file
└── static/
    ├── index.html         page layout (drop zone + pipeline diagram)
    ├── style.css          dark theme, cards, animated SVG connectors
    └── app.js             drag-drop, SSE consumer, progressive reveal
```

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Page won't load after forwarding, **no request in `server.log`** | Local port conflict (macOS AirPlay on 5000/7000). Relaunch on `8090` and forward that. See [§4](#-the-localhost5000-gotcha-macos-especially). |
| `ERROR: DINOv3 source-repo path is not set` | `export LMC_DINOV3_REPO=/path/to/dinov3` or pass `--dino_repo_dir`. |
| `FileNotFoundError: ... checkpoint ...` | Checkpoints not under `ckpt/`. See [`../ckpt/README.md`](../ckpt/README.md). |
| `CUDA out of memory` | Free other GPU processes, or run the CLI with `--mode fold:0`. |
| "Another inference is already running" | One run at a time per process; wait for the current run to finish. |
| Stages all appear at once instead of progressively | You're testing with a buffering HTTP client, not a browser. See the dev note in [§6](#6-how-it-works-internals). |
| Server won't start / port already in use | `pkill -f demo/app.py`, or pick another `--port`. |
