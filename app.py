#!/usr/bin/env python3
"""
PersonaPlex Flask Demo Bootstrapper (single Python script)

Adds what you asked for:
- HF token ingress on the FRONTEND (password field stored in browser localStorage)
- Dark monospace UI, 8px padding + 8px radius, #ffae00 accents, nested dark-grey panels/buttons
- Model availability + download stats (bytes downloaded / total bytes, %)
- Auto-downloads the model on first run (when token is present)

What it does overall:
- Clones https://github.com/NVIDIA/personaplex
- Creates a local venv
- Installs moshi/. + Flask + huggingface_hub
- Writes a Flask demo web UI:
    - Model panel: token entry, status, download button, progress, cache path
    - Audio panel: record mic -> resample to 24kHz WAV -> runs `python -m moshi.offline` -> plays output WAV
- Runs Flask

Security note (demo):
- The HF token is stored in the browser's localStorage and sent to the Flask server via an HTTP header.
- Use on localhost or on a trusted LAN only. Don’t expose this publicly.

Run:
  python3 bootstrap_personaplex_flask_demo.py
  python3 bootstrap_personaplex_flask_demo.py --dir ./ppdemo --host 0.0.0.0 --port 5000

Optional:
  --no-run   only bootstrap files, don’t run Flask
"""

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import venv
from pathlib import Path

REPO_URL = "https://github.com/NVIDIA/personaplex.git"
DEFAULT_MODEL_REPO_ID = "nvidia/personaplex-7b-v1"

APP_PY = r"""import os
import sys
import json
import uuid
import glob
import threading
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template

# ---- Config ----
MODEL_REPO_ID = os.environ.get("PP_MODEL_REPO_ID", "nvidia/personaplex-7b-v1")

APP_DIR = Path(__file__).resolve().parent
WORKDIR = APP_DIR.parent
REPO_DIR = WORKDIR / "personaplex"
RESULTS_DIR = APP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Put HF cache inside the project directory so downloads are visible + portable
HF_HOME = Path(os.environ.get("HF_HOME", str(WORKDIR / ".hf"))).resolve()
HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE", str(HF_HOME / "hub"))).resolve()

app = Flask(__name__, static_folder=str(APP_DIR / "static"), template_folder=str(APP_DIR / "templates"))

# ---- Model download state ----
_dl_lock = threading.Lock()
_dl_state = {
    "state": "idle",          # idle | downloading | ready | error
    "last_error": None,
    "snapshot_dir": None,
    "started_at": None,
    "finished_at": None,
    "total_bytes": None,      # best-effort; requires token to query
}

def _hf_model_cache_dir(repo_id: str) -> Path:
    # HF cache layout: <HF_HUB_CACHE>/models--org--name
    # repo_id like "nvidia/personaplex-7b-v1"
    org, name = repo_id.split("/", 1)
    return HF_HUB_CACHE / f"models--{org}--{name}"

def _sum_file_sizes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for p in root.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except Exception:
            pass
    return total

def _downloaded_bytes(repo_id: str) -> int:
    # "blobs" is where the actual content sits in HF cache
    base = _hf_model_cache_dir(repo_id)
    blobs = base / "blobs"
    return _sum_file_sizes(blobs)

def _snapshots(repo_id: str):
    base = _hf_model_cache_dir(repo_id)
    snaps = base / "snapshots"
    if not snaps.exists():
        return []
    out = []
    for p in snaps.iterdir():
        if p.is_dir():
            out.append(p.name)
    out.sort()
    return out

def _voice_map():
    patterns = [
        str(REPO_DIR / "**" / "NAT*.pt"),
        str(REPO_DIR / "**" / "VAR*.pt"),
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat, recursive=True))
    found = sorted(set(map(str, map(Path, found))))
    out = {}
    for p in found:
        name = Path(p).name
        out[name.replace(".pt", "")] = p
    return out

def _get_hf_token_from_request():
    # Prefer header, then JSON/form fallback
    tok = (request.headers.get("X-HF-Token") or "").strip()
    if tok:
        return tok
    # JSON body
    if request.is_json:
        try:
            j = request.get_json(silent=True) or {}
            tok = (j.get("hf_token") or "").strip()
            if tok:
                return tok
        except Exception:
            pass
    # Form field
    try:
        tok = (request.form.get("hf_token") or "").strip()
        if tok:
            return tok
    except Exception:
        pass
    return ""

def _ensure_hf_libs():
    try:
        import huggingface_hub  # noqa: F401
        return True, None
    except Exception as e:
        return False, f"huggingface_hub import failed: {e}"

def _compute_total_bytes(repo_id: str, token: str):
    # Best effort; requires gated access
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        info = api.model_info(repo_id, files_metadata=True)
        total = 0
        siblings = getattr(info, "siblings", []) or []
        for s in siblings:
            sz = getattr(s, "size", None)
            if isinstance(sz, int):
                total += sz
        return total if total > 0 else None, None
    except Exception as e:
        return None, str(e)

def _is_ready(repo_id: str, token: str = "") -> bool:
    # If we know total_bytes, check near-complete; else just check any snapshot exists.
    snaps = _snapshots(repo_id)
    if not snaps:
        return False
    with _dl_lock:
        tb = _dl_state.get("total_bytes")
    if tb and tb > 0:
        db = _downloaded_bytes(repo_id)
        return db >= int(tb * 0.98)
    return True

def _start_download_background(repo_id: str, token: str):
    ok, err = _ensure_hf_libs()
    if not ok:
        with _dl_lock:
            _dl_state["state"] = "error"
            _dl_state["last_error"] = err
        return

    with _dl_lock:
        if _dl_state["state"] == "downloading":
            return
        _dl_state["state"] = "downloading"
        _dl_state["last_error"] = None
        _dl_state["started_at"] = time.time()
        _dl_state["finished_at"] = None
        _dl_state["snapshot_dir"] = None

    def _worker():
        try:
            from huggingface_hub import snapshot_download

            # cache_dir should be the HF hub cache root
            cache_dir = str(HF_HUB_CACHE)

            snap = snapshot_download(
                repo_id=repo_id,
                token=token,
                cache_dir=cache_dir,
                resume_download=True,
            )

            with _dl_lock:
                _dl_state["snapshot_dir"] = snap
                _dl_state["state"] = "ready"
                _dl_state["finished_at"] = time.time()
        except Exception as e:
            with _dl_lock:
                _dl_state["state"] = "error"
                _dl_state["last_error"] = str(e)
                _dl_state["finished_at"] = time.time()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

def _fmt_mb(n):
    if n is None:
        return None
    return round(n / (1024 * 1024), 2)

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/voices")
def voices():
    vmap = _voice_map()
    return jsonify({"voices": sorted(vmap.keys())})

@app.get("/api/model/status")
def model_status():
    token = _get_hf_token_from_request()
    # If we have a token, compute total size best-effort and remember it
    if token:
        total, terr = _compute_total_bytes(MODEL_REPO_ID, token)
        if total:
            with _dl_lock:
                _dl_state["total_bytes"] = total

    with _dl_lock:
        state = dict(_dl_state)

    db = _downloaded_bytes(MODEL_REPO_ID)
    tb = state.get("total_bytes")
    pct = None
    if tb and tb > 0:
        pct = max(0.0, min(100.0, (db / tb) * 100.0))

    ready = _is_ready(MODEL_REPO_ID, token)

    return jsonify({
        "ok": True,
        "repo_id": MODEL_REPO_ID,
        "hf_home": str(HF_HOME),
        "hf_hub_cache": str(HF_HUB_CACHE),
        "cache_model_dir": str(_hf_model_cache_dir(MODEL_REPO_ID)),
        "state": state.get("state"),
        "last_error": state.get("last_error"),
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "snapshot_dir": state.get("snapshot_dir"),
        "snapshots": _snapshots(MODEL_REPO_ID),
        "downloaded_bytes": db,
        "total_bytes": tb,
        "downloaded_mb": _fmt_mb(db),
        "total_mb": _fmt_mb(tb) if tb else None,
        "percent": round(pct, 2) if pct is not None else None,
        "ready": bool(ready),
    })

@app.post("/api/model/download")
def model_download():
    token = _get_hf_token_from_request()
    if not token:
        return jsonify({"ok": False, "error": "Missing HF token. Provide X-HF-Token header or hf_token in JSON."}), 400

    # Update total_bytes if possible
    total, terr = _compute_total_bytes(MODEL_REPO_ID, token)
    if total:
        with _dl_lock:
            _dl_state["total_bytes"] = total

    _start_download_background(MODEL_REPO_ID, token)
    return jsonify({"ok": True, "message": "Download started (or already running)."}), 202

@app.post("/api/offline")
def offline():
    # Require model ready (or at least snapshots exist) to keep UX clean.
    # If you want to allow moshi to download on demand, remove this gate.
    token = _get_hf_token_from_request()

    if not _is_ready(MODEL_REPO_ID, token):
        # If we have a token, auto-start download then instruct client to wait
        if token:
            _start_download_background(MODEL_REPO_ID, token)
            return jsonify({
                "ok": False,
                "error": "Model not ready yet. Download started; poll /api/model/status.",
                "code": "MODEL_NOT_READY"
            }), 409
        return jsonify({
            "ok": False,
            "error": "Model not ready and no HF token provided. Enter token and download first.",
            "code": "MODEL_NOT_READY_NO_TOKEN"
        }), 409

    if "audio" not in request.files:
        return jsonify({"ok": False, "error": "Missing form file field 'audio'"}), 400

    audio_file = request.files["audio"]
    voice = (request.form.get("voice") or "NATF2").strip()
    text_prompt = (request.form.get("text_prompt") or "").strip()
    seed = (request.form.get("seed") or "42424242").strip()

    vmap = _voice_map()
    if voice not in vmap:
        return jsonify({"ok": False, "error": f"Unknown voice '{voice}'. Try /api/voices"}), 400

    job_id = uuid.uuid4().hex
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    in_wav = job_dir / "input.wav"
    out_wav = job_dir / "output.wav"
    out_json = job_dir / "output.json"

    audio_file.save(in_wav)

    cmd = [
        sys.executable, "-m", "moshi.offline",
        "--voice-prompt", vmap[voice],
        "--input-wav", str(in_wav),
        "--seed", str(seed),
        "--output-wav", str(out_wav),
        "--output-text", str(out_json),
    ]
    if text_prompt:
        cmd.extend(["--text-prompt", text_prompt])

    env = os.environ.copy()
    # Force moshi + HF libs to use our project-local HF cache
    env["HF_HOME"] = str(HF_HOME)
    env["HF_HUB_CACHE"] = str(HF_HUB_CACHE)

    # If provided via UI, forward token for any gated access
    if token:
        env["HF_TOKEN"] = token

    try:
        proc = __import__("subprocess").run(
            cmd,
            cwd=str(REPO_DIR),
            env=env,
            stdout=__import__("subprocess").PIPE,
            stderr=__import__("subprocess").PIPE,
            text=True,
            check=True,
        )
    except __import__("subprocess").CalledProcessError as e:
        err = (e.stderr or e.stdout or str(e))[-4000:]
        return jsonify({"ok": False, "error": "moshi.offline failed", "details": err}), 500

    txt = None
    if out_json.exists():
        try:
            txt = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            txt = {"raw": out_json.read_text(encoding="utf-8", errors="replace")}

    return jsonify({
        "ok": True,
        "job_id": job_id,
        "voice": voice,
        "output_wav_url": f"/results/{job_id}/output.wav",
        "output_json_url": f"/results/{job_id}/output.json",
        "output_text": txt,
        "stdout_tail": (proc.stdout or "")[-2000:],
    })

@app.get("/results/<job_id>/<path:filename>")
def results(job_id, filename):
    safe_dir = RESULTS_DIR / job_id
    if not safe_dir.exists():
        return "Not found", 404
    return send_from_directory(str(safe_dir), filename, as_attachment=False)

if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    print(f"Flask demo: http://{host}:{port}")
    print(f"Model: {MODEL_REPO_ID}")
    print(f"HF_HOME: {HF_HOME}")
    print(f"HF_HUB_CACHE: {HF_HUB_CACHE}")
    app.run(host=host, port=port, debug=debug, threaded=True)
"""

INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PersonaPlex Flask Demo</title>
  <style>
    :root{
      --bg:#0b0b0b;
      --panel:#141414;
      --panel2:#1b1b1b;
      --btn:#232323;
      --btn2:#2a2a2a;
      --text:#e8e8e8;
      --muted:#a9a9a9;
      --accent:#ffae00;
      --danger:#ff4d4d;
      --ok:#66ff99;
      --radius:8px;
      --pad:8px;
    }
    *{ box-sizing:border-box; }
    body{
      margin:0;
      background:var(--bg);
      color:var(--text);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }
    .wrap{
      max-width:1200px;
      margin:0 auto;
      padding:16px;
      display:grid;
      grid-template-columns: 420px 1fr;
      gap:16px;
    }
    .title{
      grid-column:1 / -1;
      display:flex;
      align-items:center;
      justify-content:space-between;
      padding:var(--pad);
      border:1px solid #202020;
      border-radius:var(--radius);
      background:linear-gradient(180deg, #111 0%, #0c0c0c 100%);
    }
    .title h1{
      margin:0;
      font-size:18px;
      letter-spacing:0.4px;
    }
    .badge{
      padding:6px 10px;
      border-radius:999px;
      border:1px solid rgba(255,174,0,0.35);
      color:var(--accent);
      background:rgba(255,174,0,0.08);
      font-size:12px;
    }
    .card{
      padding:var(--pad);
      border-radius:var(--radius);
      border:1px solid #232323;
      background:var(--panel);
    }
    .card .card{
      background:var(--panel2);
      border:1px solid #2a2a2a;
      margin-top:10px;
    }
    .h{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      margin-bottom:8px;
    }
    .h h2{
      margin:0;
      font-size:14px;
      color:var(--accent);
    }
    label{
      display:block;
      font-size:12px;
      color:var(--muted);
      margin:10px 0 6px;
    }
    input, textarea, select{
      width:100%;
      padding:10px 10px;
      border-radius:var(--radius);
      border:1px solid #2c2c2c;
      background:#0f0f0f;
      color:var(--text);
      outline:none;
    }
    input:focus, textarea:focus, select:focus{
      border-color:rgba(255,174,0,0.75);
      box-shadow:0 0 0 2px rgba(255,174,0,0.12);
    }
    textarea{ min-height:140px; resize:vertical; }
    .row{ display:flex; gap:10px; flex-wrap:wrap; }
    button{
      padding:10px 12px;
      border-radius:var(--radius);
      border:1px solid #333;
      background:var(--btn);
      color:var(--text);
      cursor:pointer;
      transition: transform .03s ease-in-out, background .12s ease-in-out, border-color .12s ease-in-out;
    }
    button:hover{
      background:var(--btn2);
      border-color:rgba(255,174,0,0.55);
    }
    button:active{ transform: translateY(1px); }
    button.primary{
      border-color:rgba(255,174,0,0.75);
      background:rgba(255,174,0,0.12);
      color:var(--accent);
    }
    button.danger{
      border-color:rgba(255,77,77,0.6);
      background:rgba(255,77,77,0.08);
      color:#ffb3b3;
    }
    button:disabled{
      opacity:0.55;
      cursor:not-allowed;
    }
    .kv{
      font-size:12px;
      line-height:1.35;
      color:var(--muted);
      white-space:pre-wrap;
      word-break:break-word;
    }
    .kv b{ color:var(--text); }
    .bar{
      height:10px;
      width:100%;
      background:#0f0f0f;
      border:1px solid #2a2a2a;
      border-radius:999px;
      overflow:hidden;
    }
    .bar > div{
      height:100%;
      width:0%;
      background:linear-gradient(90deg, rgba(255,174,0,0.25), rgba(255,174,0,0.95));
    }
    .statusline{
      font-size:12px;
      color:var(--muted);
      margin-top:8px;
      white-space:pre-wrap;
    }
    audio{ width:100%; margin-top:8px; }
    .tiny{ font-size:11px; color:#8a8a8a; }
    code{
      padding:2px 6px;
      border-radius:6px;
      background:#101010;
      border:1px solid #222;
      color:var(--accent);
    }
    @media (max-width: 960px){
      .wrap{ grid-template-columns:1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">
      <h1>PersonaPlex Eval</h1>
      <div class="badge">VERSION 1.1</div>
    </div>

    <!-- LEFT: MODEL + TOKEN -->
    <div class="card">
      <div class="h">
        <h2>MODEL / DOWNLOAD</h2>
        <div class="tiny">auto-download on first run</div>
      </div>

      <div class="card">
        <label>Hugging Face Token (stored in browser localStorage; sent as <code>X-HF-Token</code>)</label>
        <input id="hfToken" type="password" placeholder="hf_..." autocomplete="off" />
        <div class="row" style="margin-top:10px;">
          <button class="primary" id="saveToken">Save token</button>
          <button class="danger" id="clearToken">Clear</button>
        </div>
        <div class="statusline" id="tokenNote"></div>
      </div>

      <div class="card">
        <div class="row" style="justify-content:space-between; align-items:center;">
          <div class="kv" id="modelId"></div>
          <div class="row">
            <button id="refresh">Refresh</button>
            <button class="primary" id="download">Download model</button>
          </div>
        </div>

        <label>Download progress</label>
        <div class="bar"><div id="prog"></div></div>
        <div class="statusline" id="dlLine"></div>

        <div class="kv" id="modelStats" style="margin-top:10px;"></div>
      </div>
    </div>

    <!-- RIGHT: AUDIO + INFERENCE -->
    <div class="card">
      <div class="h">
        <h2>OFFLINE INFERENCE</h2>
        <div class="tiny">mic → 24kHz wav → moshi.offline</div>
      </div>

      <div class="card">
        <div class="row">
          <div style="flex:1; min-width:260px;">
            <label>Voice</label>
            <select id="voice"></select>
          </div>
          <div style="flex:1; min-width:260px;">
            <label>Seed</label>
            <input id="seed" value="42424242" />
          </div>
        </div>

        <label>Role / text prompt (optional)</label>
        <textarea id="prompt" placeholder="You are a wise and friendly teacher..."></textarea>
      </div>

      <div class="card">
        <div class="row">
          <button id="start">Start recording</button>
          <button id="stop" disabled>Stop</button>
          <button class="primary" id="send" disabled>Send to PersonaPlex</button>
        </div>

        <label>Input (recorded)</label>
        <audio id="inAudio" controls></audio>

        <label>Output (model)</label>
        <audio id="outAudio" controls></audio>

        <div class="statusline" id="runLine">Idle.</div>
      </div>
    </div>
  </div>

<script>
const LS_TOKEN_KEY = "pp_hf_token_v1";

const hfTokenEl = document.getElementById("hfToken");
const saveTokenBtn = document.getElementById("saveToken");
const clearTokenBtn = document.getElementById("clearToken");
const tokenNote = document.getElementById("tokenNote");

const modelIdEl = document.getElementById("modelId");
const refreshBtn = document.getElementById("refresh");
const downloadBtn = document.getElementById("download");
const progEl = document.getElementById("prog");
const dlLine = document.getElementById("dlLine");
const modelStats = document.getElementById("modelStats");

const voiceSel = document.getElementById("voice");
const seedEl = document.getElementById("seed");
const promptEl = document.getElementById("prompt");

const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const sendBtn = document.getElementById("send");
const inAudio = document.getElementById("inAudio");
const outAudio = document.getElementById("outAudio");
const runLine = document.getElementById("runLine");

let lastStatus = null;
let statusTimer = null;

function fmtMB(x){
  if (x === null || x === undefined) return "—";
  return (Math.round(x * 100) / 100).toFixed(2) + " MB";
}
function fmtBytes(n){
  if (n === null || n === undefined) return "—";
  const units = ["B","KB","MB","GB","TB"];
  let i = 0, v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return v.toFixed(i === 0 ? 0 : 2) + " " + units[i];
}

function getToken(){
  return (localStorage.getItem(LS_TOKEN_KEY) || "").trim();
}
function setToken(tok){
  localStorage.setItem(LS_TOKEN_KEY, tok.trim());
}
function clearToken(){
  localStorage.removeItem(LS_TOKEN_KEY);
}

function tokenBanner(){
  const tok = getToken();
  if (tok){
    tokenNote.textContent = "Token saved locally. It will be sent to this server only when you press Download/Run.";
  } else {
    tokenNote.textContent = "No token saved. Enter token to enable download + stats.";
  }
}

async function apiStatus(){
  const tok = getToken();
  const headers = {};
  if (tok) headers["X-HF-Token"] = tok;
  const r = await fetch("/api/model/status", { headers });
  return await r.json();
}

async function apiDownload(){
  const tok = getToken();
  if (!tok) throw new Error("No HF token saved.");
  const r = await fetch("/api/model/download", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-HF-Token": tok },
    body: JSON.stringify({ hf_token: tok })
  });
  return await r.json();
}

async function refreshStatus(){
  const j = await apiStatus();
  lastStatus = j;

  modelIdEl.textContent = "repo_id: " + (j.repo_id || "—");
  const ready = !!j.ready;
  const state = j.state || "—";

  const pct = (j.percent === null || j.percent === undefined) ? null : j.percent;
  progEl.style.width = (pct === null ? (ready ? "100%" : "0%") : Math.max(0, Math.min(100, pct)).toFixed(2) + "%");

  const dlMb = j.downloaded_mb;
  const totalMb = j.total_mb;

  const lineParts = [];
  lineParts.push("state=" + state);
  if (pct !== null) lineParts.push("progress=" + pct.toFixed(2) + "%");
  lineParts.push("downloaded=" + fmtMB(dlMb));
  if (totalMb !== null && totalMb !== undefined) lineParts.push("total=" + fmtMB(totalMb));
  if (ready) lineParts.push("READY");

  dlLine.textContent = lineParts.join(" • ");

  modelStats.textContent =
    "cache_model_dir: " + (j.cache_model_dir || "—") + "\n" +
    "hf_home: " + (j.hf_home || "—") + "\n" +
    "hf_hub_cache: " + (j.hf_hub_cache || "—") + "\n" +
    "snapshots: " + ((j.snapshots && j.snapshots.length) ? j.snapshots.join(", ") : "—") + "\n" +
    "downloaded_bytes: " + fmtBytes(j.downloaded_bytes) + "\n" +
    "total_bytes: " + fmtBytes(j.total_bytes) + "\n" +
    (j.last_error ? ("last_error: " + j.last_error) : "");

  // Gate inference buttons on readiness
  startBtn.disabled = !ready;
  if (!ready){
    stopBtn.disabled = true;
    sendBtn.disabled = true;
  }

  // Download button enabled when token exists and not ready
  const tok = getToken();
  downloadBtn.disabled = !tok || ready;

  return j;
}

async function ensureAutoDownload(){
  // Auto download on first run IF token exists and model not ready
  const tok = getToken();
  if (!tok) return;
  const st = await refreshStatus();
  if (!st.ready && st.state !== "downloading"){
    await apiDownload().catch(()=>{});
  }
}

async function loadVoices(){
  const r = await fetch("/api/voices");
  const j = await r.json();
  voiceSel.innerHTML = "";
  (j.voices || []).forEach(v=>{
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v;
    voiceSel.appendChild(o);
  });
  if (j.voices && j.voices.includes("NATF2")) voiceSel.value = "NATF2";
}

saveTokenBtn.onclick = () => {
  setToken(hfTokenEl.value || "");
  tokenBanner();
  refreshStatus().catch(()=>{});
};
clearTokenBtn.onclick = () => {
  clearToken();
  hfTokenEl.value = "";
  tokenBanner();
  refreshStatus().catch(()=>{});
};
refreshBtn.onclick = () => refreshStatus().catch(e => dlLine.textContent = "status error: " + e);
downloadBtn.onclick = async () => {
  try{
    dlLine.textContent = "Starting download...";
    await apiDownload();
  }catch(e){
    dlLine.textContent = "download error: " + e;
  }
};

function startPolling(){
  if (statusTimer) clearInterval(statusTimer);
  statusTimer = setInterval(()=>refreshStatus().catch(()=>{}), 1000);
}

/**
 * Mic capture -> Float32 PCM -> resample to 24000 Hz -> encode 16-bit PCM WAV
 */
function resampleLinear(input, inRate, outRate) {
  if (inRate === outRate) return input;
  const ratio = inRate / outRate;
  const outLength = Math.floor(input.length / ratio);
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const t = i * ratio;
    const i0 = Math.floor(t);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = t - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}
function floatTo16BitPCM(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    let s = Math.max(-1, Math.min(1, f32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return out;
}
function writeWavMono16(pcm16, sampleRate) {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample * 1;
  const byteRate = sampleRate * blockAlign;
  const dataSize = pcm16.length * bytesPerSample;
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);

  function writeStr(off, s) { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); }

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, 'WAVE');

  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);

  writeStr(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < pcm16.length; i++, offset += 2) view.setInt16(offset, pcm16[i], true);

  return new Blob([buf], { type: 'audio/wav' });
}

let audioCtx = null;
let mediaStream = null;
let processor = null;
let chunks = [];
let recordedBlob = null;
let inputSampleRate = 48000;

async function startRecording() {
  recordedBlob = null;
  outAudio.src = "";
  inAudio.src = "";
  runLine.textContent = "Recording...";
  chunks = [];

  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  inputSampleRate = audioCtx.sampleRate;

  const source = audioCtx.createMediaStreamSource(mediaStream);
  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(input));
  };

  source.connect(processor);
  processor.connect(audioCtx.destination);

  startBtn.disabled = true;
  stopBtn.disabled = false;
  sendBtn.disabled = true;
}

async function stopRecording() {
  stopBtn.disabled = true;
  startBtn.disabled = false;

  if (processor) processor.disconnect();
  if (audioCtx) await audioCtx.close();
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());

  let total = 0;
  for (const c of chunks) total += c.length;
  const mono = new Float32Array(total);
  let o = 0;
  for (const c of chunks) { mono.set(c, o); o += c.length; }

  const targetRate = 24000;
  const resampled = resampleLinear(mono, inputSampleRate, targetRate);
  const pcm16 = floatTo16BitPCM(resampled);
  recordedBlob = writeWavMono16(pcm16, targetRate);

  inAudio.src = URL.createObjectURL(recordedBlob);
  sendBtn.disabled = false;
  runLine.textContent = "Recorded @ 24kHz. Ready to send.";
}

async function sendToServer() {
  if (!recordedBlob) return;

  // Ensure model is ready
  const st = await refreshStatus();
  if (!st.ready){
    runLine.textContent = "Model not ready. Download first (token required).";
    return;
  }

  runLine.textContent = "Running moshi.offline...";
  sendBtn.disabled = true;

  const fd = new FormData();
  fd.append("audio", recordedBlob, "input.wav");
  fd.append("voice", voiceSel.value);
  fd.append("seed", seedEl.value || "42424242");
  fd.append("text_prompt", promptEl.value || "");

  const tok = getToken();
  const headers = {};
  if (tok) headers["X-HF-Token"] = tok;

  const r = await fetch("/api/offline", { method:"POST", body: fd, headers });
  const j = await r.json();

  if (!j.ok){
    runLine.textContent = "Error: " + (j.error || "unknown") + (j.details ? ("\n" + j.details) : "");
    sendBtn.disabled = false;
    return;
  }

  outAudio.src = j.output_wav_url;
  runLine.textContent = "Done. Output: " + j.output_wav_url;
  sendBtn.disabled = false;
}

startBtn.onclick = () => startRecording().catch(e => runLine.textContent = "Mic error: " + e);
stopBtn.onclick  = () => stopRecording().catch(e => runLine.textContent = "Stop error: " + e);
sendBtn.onclick  = () => sendToServer().catch(e => runLine.textContent = "Send error: " + e);

(async function init(){
  // populate token input from localStorage
  hfTokenEl.value = getToken();
  tokenBanner();

  await loadVoices().catch(()=>{});
  await refreshStatus().catch(()=>{});
  startPolling();
  await ensureAutoDownload().catch(()=>{});
})();
</script>
</body>
</html>
"""

def _run(cmd, cwd=None, env=None):
    print(f"\n$ {' '.join(map(str, cmd))}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)

def _check_dep(name):
    if shutil.which(name) is None:
        raise SystemExit(f"Missing dependency '{name}'. Please install it and re-run.")

def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Bootstrap + run a Flask demo for NVIDIA PersonaPlex (moshi.offline) with UI token ingress + model download stats.",
    )
    ap.add_argument("--dir", default="./personaplex_flask_demo", help="Working directory")
    ap.add_argument("--host", default="127.0.0.1", help="Flask bind host")
    ap.add_argument("--port", default="5000", help="Flask bind port")
    ap.add_argument("--model", default=DEFAULT_MODEL_REPO_ID, help="HF model repo id (default: nvidia/personaplex-7b-v1)")
    ap.add_argument("--no-run", action="store_true", help="Only bootstrap; do not run Flask")
    args = ap.parse_args()

    if sys.version_info < (3, 9):
        raise SystemExit("Python 3.9+ recommended for this demo.")

    _check_dep("git")

    workdir = Path(args.dir).expanduser().resolve()
    repo_dir = workdir / "personaplex"
    venv_dir = workdir / ".venv"
    demo_dir = workdir / "flask_demo"
    templates_dir = demo_dir / "templates"
    results_dir = demo_dir / "results"

    workdir.mkdir(parents=True, exist_ok=True)

    # 1) Clone or update repo
    if not (repo_dir / ".git").exists():
        print("[1/6] Cloning NVIDIA/personaplex...")
        _run(["git", "clone", REPO_URL, str(repo_dir)], cwd=str(workdir))
    else:
        print("[1/6] Updating existing repo...")
        try:
            _run(["git", "pull", "--ff-only"], cwd=str(repo_dir))
        except subprocess.CalledProcessError:
            print("Warning: git pull failed; continuing with existing checkout.")

    # 2) Create venv
    print("[2/6] Creating venv...")
    if not venv_dir.exists():
        venv.EnvBuilder(with_pip=True, clear=False).create(str(venv_dir))

    py = _venv_python(venv_dir)
    if not py.exists():
        raise SystemExit(f"Could not find venv python at: {py}")

    # 3) Upgrade pip tooling
    print("[3/6] Upgrading pip tooling...")
    _run([str(py), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])

    # 4) Install moshi/. + Flask + HF hub client
    print("[4/6] Installing moshi/. (from repo) + Flask + huggingface_hub...")
    moshi_dir = repo_dir / "moshi"
    if not moshi_dir.exists():
        raise SystemExit(f"Expected '{moshi_dir}' to exist, but it doesn't. Repo layout may have changed.")
    _run([str(py), "-m", "pip", "install", "-U", "moshi/."], cwd=str(repo_dir))
    _run([str(py), "-m", "pip", "install", "-U", "flask", "werkzeug", "huggingface_hub"])

    # 5) Write demo files
    print("[5/6] Writing Flask demo app + UI...")
    templates_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    (demo_dir / "app.py").write_text(APP_PY, encoding="utf-8")
    (templates_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")

    # 6) Run
    print("[6/6] Ready.")
    print(
        textwrap.dedent(
            f"""
            ============================================================
            Flask demo will be available at:
              http://{args.host}:{args.port}

            Model repo id:
              {args.model}

            Notes:
              - Enter your Hugging Face token in the web UI to enable:
                  * model size stats
                  * one-click download
                  * auto-download on first run
              - The HF cache is stored under:
                  {workdir}/.hf

            ============================================================
            """
        ).strip()
    )

    if args.no_run:
        print("\nBootstrap complete (--no-run set).")
        return

    env = os.environ.copy()
    env["FLASK_HOST"] = str(args.host)
    env["FLASK_PORT"] = str(args.port)
    env["PP_MODEL_REPO_ID"] = str(args.model)

    # Keep HF cache project-local (also used by moshi.offline subprocess)
    env["HF_HOME"] = str((workdir / ".hf").resolve())
    env["HF_HUB_CACHE"] = str(((workdir / ".hf") / "hub").resolve())

    _run([str(py), str(demo_dir / "app.py")], cwd=str(workdir), env=env)

if __name__ == "__main__":
    main()
