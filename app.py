#!/usr/bin/env python3
# PersonaPlex Eval (all-in-one): venv bootstrap + clone NVIDIA/personaplex + install + Flask(WebSocket) UI
#
# What you get:
# - Setup UI (dark monospace, #ffae00 accents) with HF token ingress (server-side only)
# - Model download panel with clear stats + auto-download after token set
# - "Start Talking" pill button enabled only after model is ready
# - Live mic streaming to PersonaPlex (24kHz) and live playback of generated audio
# - Fullscreen SVG ring visualizer driven by FFT (magenta=input, teal=output) + fading text cascade
# - "X" button to return to setup view
#
# Notes:
# - PersonaPlex is gated on Hugging Face: you must accept the license and provide HF token.
# - This demo uses the repo's moshi/ package (installed editable) and Hugging Face cache.
# - Requires a capable NVIDIA GPU for real-time. CPU will be slow or unusable.
#
# Run:
#   python3 personaplex_eval_demo.py
#
# Optional env:
#   PERSONAPLEX_HF_REPO="nvidia/personaplex-7b-v1"
#   PERSONAPLEX_HOST="127.0.0.1"
#   PERSONAPLEX_PORT="8787"
#   TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"   (if you want to force CUDA wheels)
#
import base64
import contextlib
import dataclasses
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple, List

ROOT = Path.cwd() / "personaplex_eval_demo"
VENV_DIR = ROOT / ".venv"
VENDOR_DIR = ROOT / "vendor"
PERSONAPLEX_DIR = VENDOR_DIR / "personaplex"

DEFAULT_HF_REPO = os.environ.get("PERSONAPLEX_HF_REPO", "nvidia/personaplex-7b-v1")
HOST = os.environ.get("PERSONAPLEX_HOST", "127.0.0.1")
PORT = int(os.environ.get("PERSONAPLEX_PORT", "8787"))

# --------- bootstrap helpers ---------
def run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)

def in_venv() -> bool:
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix

def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def venv_pip() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"

def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

def ensure_venv_and_deps() -> None:
    ensure_dirs()

    if not VENV_DIR.exists():
        print(f"Creating venv at: {VENV_DIR}")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])

    if not in_venv():
        print("Re-executing inside venv...")
        os.execv(str(venv_python()), [str(venv_python()), str(Path(__file__).resolve())] + sys.argv[1:])

    # Inside venv now
    run([str(venv_pip()), "install", "--upgrade", "pip", "setuptools", "wheel"])

    # Install core deps (torch/torchaudio can be heavy; we attempt a sensible default)
    # If you already have torch installed (recommended), pip will skip.
    core = [
        "flask>=3.0.0",
        "flask-sock>=0.7.0",
        "gevent>=24.2.1",
        "gevent-websocket>=0.10.1",
        "huggingface_hub>=0.23.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
    ]
    run([str(venv_pip()), "install"] + core)

    # Torch install (optional override)
    try:
        import torch  # noqa
    except Exception:
        idx = os.environ.get("TORCH_INDEX_URL", "").strip()
        if idx:
            run([str(venv_pip()), "install", "torch", "torchaudio", "--index-url", idx])
        else:
            # default: let pip decide (often CPU wheel)
            run([str(venv_pip()), "install", "torch", "torchaudio"])

def ensure_personaplex_repo() -> None:
    ensure_dirs()
    if not PERSONAPLEX_DIR.exists():
        if shutil.which("git") is None:
            raise RuntimeError("git is required but not found in PATH.")
        print(f"Cloning NVIDIA/personaplex into: {PERSONAPLEX_DIR}")
        run(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/personaplex.git", str(PERSONAPLEX_DIR)])
    else:
        # Keep it simple: don't auto-pull to avoid surprising changes
        print(f"Using existing repo at: {PERSONAPLEX_DIR}")

def ensure_personaplex_moshi_installed() -> None:
    moshi_path = PERSONAPLEX_DIR / "moshi"
    if not moshi_path.exists():
        raise RuntimeError(f"Expected moshi/ folder at: {moshi_path}")
    # Install editable; if already installed, this is quick
    run([str(venv_pip()), "install", "-e", str(moshi_path)])

# --------- download + model state ---------
@dataclasses.dataclass
class DownloadState:
    status: str = "idle"  # idle|need_token|downloading|ready|error
    message: str = ""
    repo_id: str = DEFAULT_HF_REPO
    total_bytes: int = 0
    done_bytes: int = 0
    files_total: int = 0
    files_done: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    last_error: str = ""

    def as_dict(self) -> dict:
        pct = 0.0
        if self.total_bytes > 0:
            pct = max(0.0, min(1.0, self.done_bytes / float(self.total_bytes)))
        return {
            "status": self.status,
            "message": self.message,
            "repo_id": self.repo_id,
            "total_bytes": self.total_bytes,
            "done_bytes": self.done_bytes,
            "files_total": self.files_total,
            "files_done": self.files_done,
            "pct": pct,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_error": self.last_error,
        }

DOWNLOAD = DownloadState(status="need_token", message="HF token required to access gated model.")

# HF token is kept server-side in memory keyed by session id.
TOKENS: Dict[str, str] = {}
TOKENS_LOCK = threading.Lock()

# Model load is expensive; we keep a single shared weight set loaded.
MODEL_LOCK = threading.Lock()
MODEL_LOADED = False
MODEL_ERR: Optional[str] = None
# These will be set after load
MIMI = None
MOSHI_LM = None
LOADERS = None
LMGen = None
TORCH = None

VOICE_PROMPTS: Dict[str, str] = {}  # name -> path


def find_voice_prompts(repo_root: Path) -> Dict[str, str]:
    # Try to discover NAT*/VAR*.pt files anywhere in the repo.
    out: Dict[str, str] = {}
    patterns = ["NATF*.pt", "NATM*.pt", "VARF*.pt", "VARM*.pt"]
    for pat in patterns:
        for p in repo_root.rglob(pat):
            name = p.stem
            out[name] = str(p)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _hf_cache_bytes_for_repo(repo_id: str) -> int:
    # Best-effort: compute size from huggingface cache snapshots for this repo.
    # Works with default cache layout (~/.cache/huggingface/hub).
    home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    hub = home / "hub"
    if not hub.exists():
        return 0
    # Hugging Face cache uses "models--org--name"
    safe = repo_id.replace("/", "--")
    model_dir = hub / f"models--{safe}"
    if not model_dir.exists():
        return 0
    total = 0
    for fp in model_dir.rglob("*"):
        if fp.is_file():
            try:
                total += fp.stat().st_size
            except Exception:
                pass
    return total


def start_download_if_needed(session_id: str, repo_id: str) -> Tuple[bool, str]:
    """
    Starts download thread if:
      - not already ready/downloading
      - token present
    Returns (started, message)
    """
    global DOWNLOAD
    if DOWNLOAD.status in ("downloading", "ready"):
        return (False, "Download already in progress or complete.")
    with TOKENS_LOCK:
        token = TOKENS.get(session_id, "").strip()
    if not token:
        DOWNLOAD.status = "need_token"
        DOWNLOAD.message = "HF token required."
        return (False, "HF token required.")

    def worker():
        global DOWNLOAD
        try:
            from huggingface_hub import HfApi, hf_hub_download

            DOWNLOAD.status = "downloading"
            DOWNLOAD.message = "Fetching model metadata..."
            DOWNLOAD.repo_id = repo_id
            DOWNLOAD.started_at = time.time()
            DOWNLOAD.finished_at = 0.0
            DOWNLOAD.last_error = ""
            DOWNLOAD.done_bytes = 0
            DOWNLOAD.total_bytes = 0
            DOWNLOAD.files_done = 0
            DOWNLOAD.files_total = 0

            api = HfApi()
            info = api.model_info(repo_id, token=token)
            # Keep to the practical set of files needed for inference; include common config/tokenizer files.
            allow_suffixes = (
                ".safetensors",
                ".pt",
                ".bin",
                ".json",
                ".txt",
                ".model",
                ".tiktoken",
            )
            siblings = []
            total = 0
            for s in getattr(info, "siblings", []) or []:
                rfilename = getattr(s, "rfilename", "") or ""
                size = int(getattr(s, "size", 0) or 0)
                if rfilename.endswith(allow_suffixes):
                    siblings.append((rfilename, size))
                    total += size

            # If we found nothing (HF page JS sometimes blocks sizes), fall back to downloading everything.
            if not siblings:
                DOWNLOAD.message = "Could not enumerate files; downloading full snapshot..."
                from huggingface_hub import snapshot_download

                snapshot_download(repo_id, token=token)
                DOWNLOAD.status = "ready"
                DOWNLOAD.message = "Download complete."
                DOWNLOAD.finished_at = time.time()
                return

            DOWNLOAD.total_bytes = total
            DOWNLOAD.files_total = len(siblings)
            DOWNLOAD.message = f"Downloading {len(siblings)} files..."

            done = 0
            files_done = 0

            for fname, sz in siblings:
                # each file downloads to cache if not present
                path = hf_hub_download(repo_id, fname, token=token)
                # count actual size on disk if available
                try:
                    done += Path(path).stat().st_size
                except Exception:
                    done += sz
                files_done += 1
                DOWNLOAD.done_bytes = done
                DOWNLOAD.files_done = files_done
                DOWNLOAD.message = f"Downloading... ({files_done}/{DOWNLOAD.files_total})"

            DOWNLOAD.status = "ready"
            DOWNLOAD.message = "Model cached and ready."
            DOWNLOAD.finished_at = time.time()

        except Exception as e:
            DOWNLOAD.status = "error"
            DOWNLOAD.message = "Download failed."
            DOWNLOAD.last_error = "".join(traceback.format_exception_only(type(e), e)).strip()
            DOWNLOAD.finished_at = time.time()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return (True, "Started download.")


def load_models_if_ready(session_id: str, repo_id: str) -> Tuple[bool, str]:
    """
    Lazy-load Mimi + Moshi LM weights into memory (GPU).
    """
    global MODEL_LOADED, MODEL_ERR, MIMI, MOSHI_LM, LOADERS, LMGen, TORCH
    if not (DOWNLOAD.status == "ready"):
        return (False, "Model not ready (download not complete).")

    with TOKENS_LOCK:
        token = TOKENS.get(session_id, "").strip()
    if not token:
        return (False, "HF token missing for loading (needed to fetch gated files even from cache).")

    with MODEL_LOCK:
        if MODEL_LOADED:
            return (True, "Model already loaded.")
        if MODEL_ERR:
            return (False, f"Previous load error: {MODEL_ERR}")

        try:
            import torch
            from huggingface_hub import hf_hub_download

            TORCH = torch

            # Import the moshi package installed from NVIDIA/personaplex repo.
            from moshi.models import loaders, LMGen as _LMGen

            LOADERS = loaders
            LMGen = _LMGen

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Grab weight filenames expected by loaders (MIMI_NAME/MOSHI_NAME).
            mimi_name = getattr(loaders, "MIMI_NAME", None)
            moshi_name = getattr(loaders, "MOSHI_NAME", None)
            if not mimi_name or not moshi_name:
                raise RuntimeError("moshi.models.loaders missing MIMI_NAME/MOSHI_NAME; incompatible moshi package?")

            mimi_weight = hf_hub_download(repo_id, mimi_name, token=token)
            moshi_weight = hf_hub_download(repo_id, moshi_name, token=token)

            mimi = loaders.get_mimi(mimi_weight, device=device)
            # Moshi expects up to 8 codebooks in many configs; keep as default but try set if available.
            with contextlib.suppress(Exception):
                mimi.set_num_codebooks(8)

            moshi_lm = loaders.get_moshi_lm(moshi_weight, device=device)

            # Warm up (optional)
            MODEL_LOADED = True
            MODEL_ERR = None
            MIMI = mimi
            MOSHI_LM = moshi_lm
            return (True, f"Loaded on {device}.")

        except Exception as e:
            MODEL_ERR = "".join(traceback.format_exception_only(type(e), e)).strip()
            return (False, MODEL_ERR)


# --------- live session inference ---------
class PersonaPlexLiveSession:
    """
    One live, stateful session.
    We create an LMGen instance per session, but share weights (MOSHI_LM, MIMI).
    """
    def __init__(self, text_prompt: str, voice_prompt_path: Optional[str], temp: float = 0.8, temp_text: float = 0.7):
        if not MODEL_LOADED:
            raise RuntimeError("Model not loaded.")
        self.torch = TORCH
        self.device = "cuda" if TORCH.cuda.is_available() else "cpu"

        # Share weights but create fresh streaming state:
        self.mimi = MIMI
        self.lm_gen = LMGen(MOSHI_LM, temp=temp, temp_text=temp_text)

        self.frame_size = int(getattr(self.mimi, "frame_size", 1920))  # 80ms @ 24kHz
        self.text_prompt = (text_prompt or "").strip()
        self.voice_prompt_path = voice_prompt_path

        self._text_decoder = self._discover_text_decoder()

    def _discover_text_decoder(self):
        # Best-effort: PersonaPlex/Moshi may expose tokenizer or decode utilities.
        # We'll try common attribute names.
        candidates = [
            ("tokenizer", "decode"),
            ("text_tokenizer", "decode"),
            ("tok", "decode"),
        ]
        for obj_name, meth in candidates:
            obj = getattr(self.lm_gen, obj_name, None)
            if obj and hasattr(obj, meth):
                return lambda t: obj.decode([t])
        # Some builds attach tokenizer on the underlying model
        base = getattr(self.lm_gen, "lm", None) or getattr(self.lm_gen, "model", None) or MOSHI_LM
        for obj_name, meth in candidates:
            obj = getattr(base, obj_name, None)
            if obj and hasattr(obj, meth):
                return lambda t: obj.decode([t])
        return None

    def _maybe_apply_text_prompt(self):
        if not self.text_prompt:
            return
        # Try known prompt setters (varies by fork).
        for name in ("set_text_prompt", "set_prompt", "set_system_prompt", "apply_text_prompt", "condition_text_prompt"):
            fn = getattr(self.lm_gen, name, None)
            if callable(fn):
                fn(self.text_prompt)
                return
        # Try on underlying model
        base = getattr(self.lm_gen, "lm", None) or getattr(self.lm_gen, "model", None) or MOSHI_LM
        for name in ("set_text_prompt", "set_prompt", "set_system_prompt", "apply_text_prompt", "condition_text_prompt"):
            fn = getattr(base, name, None)
            if callable(fn):
                fn(self.text_prompt)
                return
        # If no method found, silently continue (still works, just without persona conditioning).

    def _prime_voice_prompt(self):
        if not self.voice_prompt_path:
            return
        p = Path(self.voice_prompt_path)
        if not p.exists():
            return

        vp = self.torch.load(str(p), map_location="cpu")
        # Expect voice prompt as sequence of audio tokens. We try to coerce to [1, K, T].
        # Common formats: [K, T], [1, K, T], dict containing "codes".
        if isinstance(vp, dict):
            for k in ("codes", "tokens", "audio_tokens", "prompt"):
                if k in vp:
                    vp = vp[k]
                    break

        if not hasattr(vp, "shape"):
            return

        t = vp
        if len(t.shape) == 2:
            t = t.unsqueeze(0)  # [1,K,T]
        if len(t.shape) == 3 and t.shape[0] != 1:
            # take first
            t = t[:1]
        if len(t.shape) != 3:
            return
        # Prime frame by frame: t is [1,K,T]
        # LMGen.step expects [B,K,1] per step in many builds.
        # We'll feed each timestep as [1,K,1]
        for i in range(t.shape[-1]):
            frame_codes = t[:, :, i:i+1].to(self.device)
            _ = self.lm_gen.step(frame_codes)

    def step_pcm16(self, pcm16: bytes) -> Tuple[Optional[bytes], Optional[str]]:
        """
        pcm16: little-endian int16 mono, exactly frame_size samples @ 24kHz
        Returns (out_pcm16_frame, text_delta)
        """
        # Convert to float32 tensor [-1,1]
        import numpy as np
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if x.shape[0] != self.frame_size:
            # pad/truncate to frame_size
            if x.shape[0] < self.frame_size:
                x = np.pad(x, (0, self.frame_size - x.shape[0]))
            else:
                x = x[: self.frame_size]
        wav = self.torch.from_numpy(x).view(1, 1, self.frame_size).to(self.device)

        # Encode -> step -> decode
        codes_in = self.mimi.encode(wav)  # [1,K,1]
        tokens_out = self.lm_gen.step(codes_in)

        if tokens_out is None:
            return (None, None)

        # tokens_out often: [B, 1+K, 1] where tokens_out[:,0,0]=text token and [:,1:,:]=audio codes
        text_delta = None
        try:
            tok = int(tokens_out[0, 0, 0].item())
            if self._text_decoder and tok != 0:
                # decode single token; may be piece
                piece = self._text_decoder(tok)
                if piece and isinstance(piece, str):
                    text_delta = piece
        except Exception:
            pass

        try:
            audio_codes = tokens_out[:, 1:, :]  # [1,K,1]
            out_wav = self.mimi.decode(audio_codes)  # [1,1,T]
            out = out_wav.detach().float().clamp(-1, 1).cpu().numpy().reshape(-1)
            out_i16 = (out * 32767.0).astype("int16").tobytes()
            return (out_i16, text_delta)
        except Exception:
            return (None, text_delta)

    def streaming_context(self):
        return contextlib.ExitStack()


# Allow only one live session by default (stateful, heavy).
LIVE_LOCK = threading.Lock()

# --------- Flask app ---------
def create_app():
    from flask import Flask, request, make_response, jsonify, session, Response
    from flask_sock import Sock

    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))
    sock = Sock(app)

    # Session id cookie
    @app.before_request
    def _ensure_sid():
        if "sid" not in session:
            session["sid"] = secrets.token_urlsafe(16)

    @app.get("/api/status")
    def api_status():
        sid = session.get("sid")
        cache_bytes = _hf_cache_bytes_for_repo(DOWNLOAD.repo_id)
        payload = DOWNLOAD.as_dict()
        payload["hf_cache_bytes_estimate"] = cache_bytes
        payload["model_loaded"] = MODEL_LOADED
        payload["model_error"] = MODEL_ERR
        payload["voices"] = list(VOICE_PROMPTS.keys())
        payload["default_voice"] = "NATF2" if "NATF2" in VOICE_PROMPTS else (list(VOICE_PROMPTS.keys())[0] if VOICE_PROMPTS else "")
        payload["has_token"] = bool(TOKENS.get(sid, "").strip())
        return jsonify(payload)

    @app.post("/api/set_token")
    def api_set_token():
        sid = session.get("sid")
        data = request.get_json(force=True, silent=True) or {}
        token = (data.get("token") or "").strip()
        if not token:
            return jsonify({"ok": False, "error": "Missing token"}), 400
        with TOKENS_LOCK:
            TOKENS[sid] = token
        # auto-download if not ready
        if DOWNLOAD.status in ("need_token", "idle", "error"):
            start_download_if_needed(sid, DOWNLOAD.repo_id)
        return jsonify({"ok": True})

    @app.post("/api/clear_token")
    def api_clear_token():
        sid = session.get("sid")
        with TOKENS_LOCK:
            TOKENS.pop(sid, None)
        return jsonify({"ok": True})

    @app.post("/api/start_download")
    def api_start_download():
        sid = session.get("sid")
        data = request.get_json(force=True, silent=True) or {}
        repo = (data.get("repo_id") or DOWNLOAD.repo_id or DEFAULT_HF_REPO).strip()
        DOWNLOAD.repo_id = repo
        started, msg = start_download_if_needed(sid, repo)
        return jsonify({"ok": True, "started": started, "message": msg})

    @app.post("/api/load_model")
    def api_load_model():
        sid = session.get("sid")
        data = request.get_json(force=True, silent=True) or {}
        repo = (data.get("repo_id") or DOWNLOAD.repo_id or DEFAULT_HF_REPO).strip()
        ok, msg = load_models_if_ready(sid, repo)
        return jsonify({"ok": ok, "message": msg})

    @sock.route("/ws/live")
    def ws_live(ws):
        # This is a simple binary+json protocol:
        # Client sends first message as JSON text:
        #   {"type":"config","text_prompt":"...","voice":"NATF2","temp":0.8,"temp_text":0.7}
        # Then sends audio frames as bytes:
        #   b'\x01' + <pcm16 bytes for exactly frame_size samples>
        # Server sends:
        #   JSON strings for events (ready/text/error)
        #   b'\x02' + <pcm16 out frame>
        try:
            with LIVE_LOCK:
                sid = None
                # Receive config
                first = ws.receive()
                if not isinstance(first, str):
                    ws.send(json.dumps({"type": "error", "error": "Expected JSON config as first message."}))
                    return
                cfg = json.loads(first)
                if cfg.get("type") != "config":
                    ws.send(json.dumps({"type": "error", "error": "First message must be type=config."}))
                    return

                # Ensure model downloaded and loaded
                # (We don't have flask session in ws handler easily; pass token by separate /api endpoints.)
                if DOWNLOAD.status != "ready":
                    ws.send(json.dumps({"type": "error", "error": "Model not downloaded yet."}))
                    return
                if not MODEL_LOADED:
                    # We can't read flask session here reliably; ask client to call /api/load_model first.
                    ws.send(json.dumps({"type": "need_load", "message": "Call /api/load_model before starting."}))
                    return

                text_prompt = (cfg.get("text_prompt") or "").strip()
                voice = (cfg.get("voice") or "").strip()
                temp = float(cfg.get("temp") or 0.8)
                temp_text = float(cfg.get("temp_text") or 0.7)
                voice_path = VOICE_PROMPTS.get(voice) if voice else None

                session_obj = PersonaPlexLiveSession(
                    text_prompt=text_prompt,
                    voice_prompt_path=voice_path,
                    temp=temp,
                    temp_text=temp_text,
                )

                # Start streaming contexts and prime prompts if possible.
                torch = TORCH
                ws.send(json.dumps({"type": "ready", "frame_size": session_obj.frame_size, "sample_rate": 24000}))

                # Run streaming in a single loop: every inbound audio frame -> step -> send outbound
                # Use moshi streaming contexts for performance/statefulness.
                with torch.no_grad(), session_obj.lm_gen.streaming(1), session_obj.mimi.streaming(1):
                    # Apply prompts
                    with contextlib.suppress(Exception):
                        session_obj._maybe_apply_text_prompt()
                    with contextlib.suppress(Exception):
                        session_obj._prime_voice_prompt()

                    text_accum = ""
                    while True:
                        msg = ws.receive()
                        if msg is None:
                            break

                        if isinstance(msg, str):
                            # control messages
                            try:
                                j = json.loads(msg)
                            except Exception:
                                continue
                            if j.get("type") == "ping":
                                ws.send(json.dumps({"type": "pong", "t": time.time()}))
                            continue

                        # binary audio
                        b = msg
                        if not b:
                            continue
                        mtype = b[0]
                        if mtype != 0x01:
                            continue
                        pcm = b[1:]

                        out_pcm, tdelta = session_obj.step_pcm16(pcm)
                        if tdelta:
                            text_accum += tdelta
                            # send incremental (and let client render cascading pieces)
                            ws.send(json.dumps({"type": "text", "delta": tdelta, "full": text_accum[-2000:]}))

                        if out_pcm:
                            ws.send(b"\x02" + out_pcm)

        except Exception as e:
            err = "".join(traceback.format_exception_only(type(e), e)).strip()
            try:
                ws.send(json.dumps({"type": "error", "error": err}))
            except Exception:
                pass

    @app.get("/")
    def index():
        # One-page app: setup view + talk view. Start Talking triggers getUserMedia in same gesture.
        html = INDEX_HTML
        resp = make_response(html)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    return app


# --------- Frontend (single-page) ---------
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>PersonaPlex Eval</title>
  <style>
    :root{
      --bg: #0b0b0c;
      --panel: #141416;
      --panel2: #1b1b1e;
      --muted: #a8a8aa;
      --text: #f2f2f3;
      --accent: #ffae00;
      --danger: #ff4d4d;
      --ok: #39d98a;
      --ring-magenta: #ff3bd4;
      --ring-teal: #27f3ff;
      --radius: 8px;
      --pad: 14px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }
    *{ box-sizing:border-box; }
    html,body{ height:100%; }
    body{
      margin:0;
      background: radial-gradient(1200px 900px at 15% 10%, #141416 0%, #0b0b0c 55%, #070708 100%);
      color: var(--text);
      font-family: var(--mono);
      letter-spacing: 0.1px;
    }
    .wrap{
      max-width: 1080px;
      margin: 0 auto;
      padding: 22px;
    }
    .topbar{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap: 14px;
      margin-bottom: 16px;
    }
    .title{
      font-size: 18px;
      font-weight: 700;
      color: var(--text);
      display:flex;
      align-items:center;
      gap:10px;
    }
    .title .dot{
      width:10px;height:10px;border-radius:999px;
      background: var(--accent);
      box-shadow: 0 0 16px rgba(255,174,0,0.35);
    }
    .pill{
      display:inline-flex;
      align-items:center;
      gap:10px;
      padding: 10px 14px;
      border-radius: 999px;
      background: linear-gradient(180deg, #1b1b1e, #131315);
      border: 1px solid rgba(255,174,0,0.35);
      color: var(--text);
      cursor:pointer;
      user-select:none;
      transition: transform .08s ease, opacity .2s ease, filter .2s ease;
    }
    .pill:hover{ transform: translateY(-1px); filter: brightness(1.05); }
    .pill:active{ transform: translateY(0px); }
    .pill.disabled{
      opacity: 0.35;
      cursor:not-allowed;
      filter: grayscale(0.2);
    }
    .grid{
      display:grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 14px;
    }
    @media (max-width: 980px){
      .grid{ grid-template-columns: 1fr; }
    }
    .panel{
      background: linear-gradient(180deg, var(--panel), #101012);
      border: 1px solid rgba(255,174,0,0.18);
      border-radius: var(--radius);
      padding: var(--pad);
      box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    }
    .panel h3{
      margin: 0 0 10px 0;
      font-size: 13px;
      color: var(--accent);
      letter-spacing: .6px;
      text-transform: uppercase;
    }
    .row{
      display:flex;
      gap: 10px;
      align-items:center;
      flex-wrap: wrap;
    }
    input, textarea, select{
      font-family: var(--mono);
      background: var(--panel2);
      border: 1px solid rgba(255,174,0,0.18);
      color: var(--text);
      border-radius: var(--radius);
      padding: 10px 11px;
      outline: none;
      width: 100%;
    }
    textarea{ min-height: 120px; resize: vertical; }
    .btn{
      font-family: var(--mono);
      background: #232326;
      border: 1px solid rgba(255,174,0,0.35);
      color: var(--text);
      border-radius: var(--radius);
      padding: 10px 12px;
      cursor:pointer;
      transition: transform .08s ease, filter .2s ease, opacity .2s ease;
      user-select:none;
    }
    .btn:hover{ transform: translateY(-1px); filter: brightness(1.06); }
    .btn:active{ transform: translateY(0px); }
    .btn.ghost{ background: transparent; }
    .btn.danger{ border-color: rgba(255,77,77,0.45); }
    .btn.disabled{ opacity: .45; cursor:not-allowed; }
    .muted{ color: var(--muted); font-size: 12px; line-height: 1.35; }
    .kv{
      display:grid;
      grid-template-columns: 160px 1fr;
      gap: 6px 12px;
      margin-top: 8px;
      font-size: 12px;
    }
    .kv div:nth-child(odd){ color: var(--muted); }
    .kv code{
      color: var(--text);
      background: rgba(0,0,0,0.25);
      padding: 2px 6px;
      border-radius: 6px;
      border: 1px solid rgba(255,174,0,0.15);
    }
    .progress{
      height: 10px;
      width: 100%;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,174,0,0.14);
      overflow:hidden;
      margin-top: 10px;
    }
    .bar{
      height:100%;
      width: 0%;
      background: linear-gradient(90deg, rgba(255,174,0,0.20), rgba(255,174,0,0.95));
      box-shadow: 0 0 18px rgba(255,174,0,0.35);
      transition: width .25s ease;
    }
    .badge{
      display:inline-flex;
      align-items:center;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid rgba(255,174,0,0.25);
      background: rgba(255,174,0,0.08);
      font-size: 11px;
      color: var(--accent);
      margin-left: 8px;
    }
    .statusLine{
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap: 10px;
      font-size: 12px;
      margin-top: 8px;
    }
    .statusDot{
      width:10px;height:10px;border-radius:999px;
      background: rgba(255,255,255,0.2);
      box-shadow:none;
    }
    .statusDot.ok{ background: var(--ok); box-shadow: 0 0 12px rgba(57,217,138,0.35); }
    .statusDot.warn{ background: var(--accent); box-shadow: 0 0 12px rgba(255,174,0,0.35); }
    .statusDot.err{ background: var(--danger); box-shadow: 0 0 12px rgba(255,77,77,0.35); }

    /* Talk view */
    #talkView{
      position: fixed;
      inset: 0;
      display:none;
      background: radial-gradient(1100px 800px at 50% 45%, rgba(255,174,0,0.06), rgba(0,0,0,0) 55%),
                  radial-gradient(900px 700px at 50% 60%, rgba(39,243,255,0.05), rgba(0,0,0,0) 60%),
                  #050506;
      overflow:hidden;
    }
    #talkView.active{ display:block; }
    #exitBtn{
      position:absolute;
      top: 14px;
      left: 14px;
      width: 42px;
      height: 42px;
      border-radius: 12px;
      border: 1px solid rgba(255,174,0,0.28);
      background: rgba(20,20,22,0.7);
      color: var(--text);
      cursor:pointer;
      display:flex;
      align-items:center;
      justify-content:center;
      font-size: 18px;
      line-height: 1;
      user-select:none;
      transition: transform .08s ease, filter .2s ease, opacity .2s ease;
      z-index: 5;
    }
    #exitBtn:hover{ transform: translateY(-1px); filter: brightness(1.08); }
    #exitBtn:active{ transform: translateY(0px); }
    #ringWrap{
      position:absolute;
      inset:0;
      display:flex;
      align-items:center;
      justify-content:center;
      opacity: 0;
      transition: opacity .35s ease;
    }
    #ringWrap.visible{ opacity: 1; }
    #ringSvg{
      width: min(72vh, 78vw);
      height: min(72vh, 78vw);
      filter: drop-shadow(0 0 40px rgba(255,174,0,0.08));
    }
    #ringPath{
      fill: rgba(0,0,0,0.0);
      stroke: var(--ring-magenta);
      stroke-width: 10;
      stroke-linejoin: round;
      stroke-linecap: round;
      opacity: 0.95;
    }
    #ringHint{
      position:absolute;
      bottom: 26px;
      left: 50%;
      transform: translateX(-50%);
      font-size: 12px;
      color: rgba(255,255,255,0.55);
      text-align:center;
      width: min(680px, 90vw);
      pointer-events:none;
    }
    #cascade{
      position:absolute;
      left: 50%;
      bottom: 12%;
      transform: translateX(-50%);
      width: min(860px, 92vw);
      display:flex;
      flex-direction:column;
      gap: 6px;
      align-items:center;
      pointer-events:none;
      z-index: 4;
    }
    .cascadeItem{
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(18,18,20,0.72);
      border: 1px solid rgba(255,174,0,0.16);
      color: rgba(255,255,255,0.92);
      font-size: 13px;
      line-height: 1.3;
      max-width: 100%;
      overflow-wrap: anywhere;
      opacity: 0;
      transform: translateY(10px);
      animation: riseFade 3.2s ease forwards;
    }
    @keyframes riseFade{
      0%{ opacity: 0; transform: translateY(10px); }
      12%{ opacity: 0.95; transform: translateY(0px); }
      80%{ opacity: 0.62; transform: translateY(-6px); }
      100%{ opacity: 0; transform: translateY(-12px); }
    }

    .toast{
      position: fixed;
      right: 18px;
      bottom: 18px;
      max-width: min(520px, 92vw);
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(255,174,0,0.24);
      background: rgba(18,18,20,0.85);
      color: rgba(255,255,255,0.92);
      font-size: 12px;
      z-index: 999;
      opacity: 0;
      transform: translateY(8px);
      transition: opacity .2s ease, transform .2s ease;
    }
    .toast.show{ opacity: 1; transform: translateY(0px); }
  </style>
</head>
<body>
  <div id="setupView">
    <div class="wrap">
      <div class="topbar">
        <div class="title"><span class="dot"></span> PersonaPlex Eval</div>
        <div class="pill disabled" id="startTalkingBtn" title="Download + load model first">
          Start Talking
          <span class="badge" id="startBadge">locked</span>
        </div>
      </div>

      <div class="grid">
        <div class="panel">
          <h3>Hugging Face Access</h3>
          <div class="muted">
            PersonaPlex is gated. Paste your HF token (after accepting the model terms) — it is stored only in server memory for this session.
          </div>
          <div class="row" style="margin-top:10px;">
            <input id="hfToken" type="password" placeholder="HF_TOKEN (not stored in browser)"/>
            <button class="btn" id="saveTokenBtn">Save Token</button>
            <button class="btn ghost danger" id="clearTokenBtn">Clear</button>
          </div>
          <div class="statusLine">
            <div class="row" style="gap:8px;">
              <div class="statusDot warn" id="tokenDot"></div>
              <div id="tokenStatus">token: unknown</div>
            </div>
            <div class="muted" id="repoLine"></div>
          </div>
        </div>

        <div class="panel">
          <h3>Model Download</h3>
          <div class="muted">Downloads model weights into your Hugging Face cache. Auto-download triggers once token is saved.</div>

          <div class="kv">
            <div>Repo</div><div><code id="repoCode"></code></div>
            <div>Status</div><div><code id="dlStatus"></code></div>
            <div>Cache bytes</div><div><code id="cacheBytes"></code></div>
            <div>Files</div><div><code id="fileCount"></code></div>
          </div>

          <div class="progress"><div class="bar" id="dlBar"></div></div>
          <div class="statusLine">
            <div class="row" style="gap:8px;">
              <div class="statusDot warn" id="dlDot"></div>
              <div id="dlMsg">…</div>
            </div>
            <div class="muted" id="dlPct"></div>
          </div>

          <div class="row" style="margin-top:10px; justify-content:space-between;">
            <button class="btn" id="downloadBtn">Download / Resume</button>
            <button class="btn" id="loadBtn">Load Into GPU</button>
          </div>

          <div class="muted" id="errBox" style="margin-top:10px; color: rgba(255,77,77,0.95); display:none;"></div>
        </div>

        <div class="panel">
          <h3>Persona Prompts</h3>
          <div class="muted">PersonaPlex is conditioned on a voice prompt (audio tokens) and a text prompt (role/persona context).</div>

          <div class="row" style="margin-top:10px;">
            <select id="voiceSelect"></select>
          </div>

          <div class="row" style="margin-top:10px;">
            <textarea id="textPrompt" placeholder="Text prompt (role/persona), e.g.:
You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."></textarea>
          </div>

          <div class="row" style="margin-top:10px; justify-content:space-between;">
            <button class="btn" id="savePromptBtn">Save Prompt Settings</button>
            <div class="muted">Saved locally (except HF token).</div>
          </div>
        </div>

        <div class="panel">
          <h3>Notes</h3>
          <div class="muted">
            • Microphone works on <code>localhost</code> even on HTTP (secure context exception).<br/>
            • For best results you want a GPU (A100/H100-class recommended for real-time).<br/>
            • Audio is streamed at <code>24kHz</code> in <code>80ms</code> frames (matching Mimi streaming constraints).<br/>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div id="talkView">
    <div id="exitBtn" title="Back to setup">✕</div>
    <div id="ringWrap">
      <svg id="ringSvg" viewBox="0 0 1000 1000" aria-label="FFT ring">
        <path id="ringPath"></path>
      </svg>
      <div id="ringHint">Listening + speaking (full duplex). Magenta = mic energy. Teal = model energy.</div>
    </div>
    <div id="cascade"></div>
  </div>

  <div class="toast" id="toast"></div>

<script>
(() => {
  const $ = (id) => document.getElementById(id);

  const startBtn = $("startTalkingBtn");
  const startBadge = $("startBadge");
  const repoCode = $("repoCode");
  const repoLine = $("repoLine");
  const dlStatus = $("dlStatus");
  const cacheBytes = $("cacheBytes");
  const fileCount = $("fileCount");
  const dlBar = $("dlBar");
  const dlMsg = $("dlMsg");
  const dlPct = $("dlPct");
  const dlDot = $("dlDot");
  const tokenDot = $("tokenDot");
  const tokenStatus = $("tokenStatus");
  const errBox = $("errBox");

  const voiceSelect = $("voiceSelect");
  const textPrompt = $("textPrompt");

  const setupView = $("setupView");
  const talkView = $("talkView");
  const exitBtn = $("exitBtn");
  const ringWrap = $("ringWrap");
  const ringPath = $("ringPath");
  const cascade = $("cascade");
  const toast = $("toast");

  const saved = {
    voice: localStorage.getItem("pp_voice") || "",
    text: localStorage.getItem("pp_text") || "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
    temp: parseFloat(localStorage.getItem("pp_temp") || "0.8"),
    temp_text: parseFloat(localStorage.getItem("pp_temp_text") || "0.7"),
  };
  textPrompt.value = saved.text;

  function fmtBytes(n){
    n = Number(n||0);
    const units = ["B","KB","MB","GB","TB"];
    let u = 0;
    while(n >= 1024 && u < units.length-1){ n/=1024; u++; }
    return `${n.toFixed(u===0?0:2)} ${units[u]}`;
  }

  let status = null;
  let ws = null;

  function showToast(msg){
    toast.textContent = msg;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 2400);
  }

  async function postJSON(url, body){
    const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body||{})});
    const j = await r.json().catch(()=> ({}));
    if(!r.ok) throw new Error(j.error || j.message || `HTTP ${r.status}`);
    return j;
  }

  function setDot(dotEl, kind){
    dotEl.classList.remove("ok","warn","err");
    dotEl.classList.add(kind);
  }

  function setStartEnabled(enabled){
    startBtn.classList.toggle("disabled", !enabled);
    startBadge.textContent = enabled ? "ready" : "locked";
    startBadge.style.opacity = enabled ? "1" : "0.8";
  }

  async function refresh(){
    const r = await fetch("/api/status", {cache:"no-store"});
    status = await r.json();
    repoCode.textContent = status.repo_id || "";
    repoLine.textContent = `repo: ${status.repo_id || ""}`;

    // Token status
    tokenStatus.textContent = status.has_token ? "token: present" : "token: missing";
    setDot(tokenDot, status.has_token ? "ok" : "warn");

    // Download status
    dlStatus.textContent = status.status || "";
    cacheBytes.textContent = fmtBytes(status.hf_cache_bytes_estimate || 0);
    fileCount.textContent = `${status.files_done||0}/${status.files_total||0}`;

    const pct = Math.round((status.pct||0)*100);
    dlBar.style.width = `${pct}%`;
    dlPct.textContent = status.total_bytes ? `${pct}% (${fmtBytes(status.done_bytes)} / ${fmtBytes(status.total_bytes)})` : `${pct}%`;
    dlMsg.textContent = status.message || "";
    errBox.style.display = "none";

    if(status.status === "ready"){
      setDot(dlDot, "ok");
    } else if(status.status === "error"){
      setDot(dlDot, "err");
      if(status.last_error){
        errBox.style.display = "block";
        errBox.textContent = `Download error: ${status.last_error}`;
      }
    } else {
      setDot(dlDot, "warn");
    }

    // Voices
    const voices = status.voices || [];
    const defaultVoice = status.default_voice || "";
    if(voiceSelect.options.length === 0 && voices.length){
      voiceSelect.innerHTML = "";
      for(const v of voices){
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        voiceSelect.appendChild(opt);
      }
      const pick = saved.voice && voices.includes(saved.voice) ? saved.voice : defaultVoice;
      if(pick) voiceSelect.value = pick;
    }

    // Load status
    if(status.model_error){
      errBox.style.display = "block";
      errBox.textContent = `Model load error: ${status.model_error}`;
    }

    const readyToTalk = (status.status === "ready") && (status.model_loaded === true);
    setStartEnabled(readyToTalk);
  }

  // Auto-refresh loop
  setInterval(refresh, 900);
  refresh();

  // Buttons
  $("saveTokenBtn").onclick = async () => {
    const t = $("hfToken").value.trim();
    if(!t) return showToast("Paste your HF token first.");
    try{
      await postJSON("/api/set_token", {token: t});
      $("hfToken").value = "";
      showToast("Token saved (server-side). Auto-download started.");
      await refresh();
    }catch(e){
      showToast(`Token save failed: ${e.message}`);
    }
  };
  $("clearTokenBtn").onclick = async () => {
    await postJSON("/api/clear_token", {});
    showToast("Token cleared.");
    await refresh();
  };
  $("downloadBtn").onclick = async () => {
    try{
      await postJSON("/api/start_download", {repo_id: (status && status.repo_id) || ""});
      showToast("Download/resume requested.");
      await refresh();
    }catch(e){
      showToast(`Download failed to start: ${e.message}`);
    }
  };
  $("loadBtn").onclick = async () => {
    try{
      const j = await postJSON("/api/load_model", {repo_id: (status && status.repo_id) || ""});
      showToast(j.ok ? j.message : `Load failed: ${j.message}`);
      await refresh();
    }catch(e){
      showToast(`Load failed: ${e.message}`);
    }
  };
  $("savePromptBtn").onclick = () => {
    localStorage.setItem("pp_voice", voiceSelect.value || "");
    localStorage.setItem("pp_text", textPrompt.value || "");
    showToast("Prompt settings saved locally.");
  };

  // --- Talk mode (live duplex) ---
  let audioCtx = null;
  let micStream = null;
  let micSource = null;
  let micAnalyser = null;

  let outAnalyser = null;
  let scriptNodeIn = null;
  let scriptNodeOut = null;

  // output jitter queue in float32
  let outQ = [];
  let outQMax = 24000 * 3; // 3 seconds buffer cap

  // visualization
  let raf = 0;
  const fftBins = 256;
  let micFFT = new Uint8Array(fftBins);
  let outFFT = new Uint8Array(fftBins);

  function downsampleFloat32(src, srcRate, dstRate){
    if(dstRate === srcRate) return src;
    const ratio = srcRate / dstRate;
    const dstLen = Math.floor(src.length / ratio);
    const dst = new Float32Array(dstLen);
    let pos = 0;
    for(let i=0;i<dstLen;i++){
      const p = i * ratio;
      const p0 = Math.floor(p);
      const p1 = Math.min(src.length-1, p0+1);
      const frac = p - p0;
      dst[i] = src[p0]*(1-frac) + src[p1]*frac;
    }
    return dst;
  }

  function pcm16ToFloat32(bytes){
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const n = bytes.byteLength / 2;
    const out = new Float32Array(n);
    for(let i=0;i<n;i++){
      out[i] = dv.getInt16(i*2, true) / 32768.0;
    }
    return out;
  }

  function float32ToPCM16(f32){
    const out = new Int16Array(f32.length);
    for(let i=0;i<f32.length;i++){
      let v = Math.max(-1, Math.min(1, f32[i]));
      out[i] = (v * 32767) | 0;
    }
    return new Uint8Array(out.buffer);
  }

  function energyFromFFT(arr){
    let s = 0;
    for(let i=0;i<arr.length;i++){
      const v = arr[i]/255;
      s += v*v;
    }
    return Math.sqrt(s / arr.length);
  }

  function buildRingPathFromFFT(freq, cx, cy, baseR){
    const N = 240;
    const amp = baseR * 0.20;
    let d = "";
    for(let i=0;i<N;i++){
      const a = (i / N) * Math.PI * 2;
      const bin = Math.floor((i / N) * (freq.length-1));
      const v = freq[bin] / 255;
      const r = baseR + amp * Math.pow(v, 1.35);
      const x = cx + r * Math.cos(a);
      const y = cy + r * Math.sin(a);
      d += (i===0 ? `M ${x.toFixed(2)} ${y.toFixed(2)}` : ` L ${x.toFixed(2)} ${y.toFixed(2)}`);
    }
    d += " Z";
    return d;
  }

  function startViz(){
    cancelAnimationFrame(raf);
    const baseR = 330;
    const cx = 500, cy = 500;
    const tick = () => {
      raf = requestAnimationFrame(tick);

      if(micAnalyser){
        micAnalyser.getByteFrequencyData(micFFT);
      } else {
        micFFT.fill(0);
      }
      if(outAnalyser){
        outAnalyser.getByteFrequencyData(outFFT);
      } else {
        outFFT.fill(0);
      }

      const micE = energyFromFFT(micFFT);
      const outE = energyFromFFT(outFFT);

      const useOut = outE > micE && outE > 0.06;
      const useMic = micE >= outE && micE > 0.04;

      let active = null;
      if(useOut) active = "out";
      else if(useMic) active = "mic";

      const freq = active === "out" ? outFFT : micFFT;
      const stroke = 10 + 18 * Math.min(1, (active==="out"?outE:micE));
      ringPath.setAttribute("d", buildRingPathFromFFT(freq, cx, cy, baseR));
      ringPath.style.strokeWidth = stroke.toFixed(2);

      if(active === "out") ringPath.style.stroke = getComputedStyle(document.documentElement).getPropertyValue("--ring-teal").trim();
      else if(active === "mic") ringPath.style.stroke = getComputedStyle(document.documentElement).getPropertyValue("--ring-magenta").trim();
      else ringPath.style.stroke = "rgba(255,255,255,0.18)";
    };
    tick();
  }

  function stopViz(){
    cancelAnimationFrame(raf);
    raf = 0;
  }

  function addCascadeText(txt){
    const t = (txt || "").trim();
    if(!t) return;
    const div = document.createElement("div");
    div.className = "cascadeItem";
    div.textContent = t;
    cascade.appendChild(div);
    // prune
    setTimeout(() => {
      if(div && div.parentNode) div.parentNode.removeChild(div);
    }, 3400);
  }

  async function startTalking(){
    // Safety checks
    if(!status || status.status !== "ready" || status.model_loaded !== true){
      showToast("Model not ready. Download + load first.");
      return;
    }

    // Save prompt settings
    localStorage.setItem("pp_voice", voiceSelect.value || "");
    localStorage.setItem("pp_text", textPrompt.value || "");

    // Switch view
    talkView.classList.add("active");
    ringWrap.classList.remove("visible");
    cascade.innerHTML = "";

    // Start audio
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const sr = audioCtx.sampleRate;

    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });

    micSource = audioCtx.createMediaStreamSource(micStream);
    micAnalyser = audioCtx.createAnalyser();
    micAnalyser.fftSize = fftBins * 2;
    micAnalyser.smoothingTimeConstant = 0.85;
    micSource.connect(micAnalyser);

    // Output chain: ScriptProcessor pulls from outQ and drives an analyser for the ring.
    outAnalyser = audioCtx.createAnalyser();
    outAnalyser.fftSize = fftBins * 2;
    outAnalyser.smoothingTimeConstant = 0.82;

    scriptNodeOut = audioCtx.createScriptProcessor(2048, 1, 1);
    scriptNodeOut.onaudioprocess = (e) => {
      const out = e.outputBuffer.getChannelData(0);
      const need = out.length;
      for(let i=0;i<need;i++){
        out[i] = outQ.length ? outQ.shift() : 0.0;
      }
    };
    scriptNodeOut.connect(outAnalyser);
    outAnalyser.connect(audioCtx.destination);

    // WebSocket connect
    const proto = (location.protocol === "https:") ? "wss://" : "ws://";
    const wsUrl = proto + location.host + "/ws/live";
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    const cfg = {
      type: "config",
      text_prompt: localStorage.getItem("pp_text") || "",
      voice: localStorage.getItem("pp_voice") || "",
      temp: saved.temp,
      temp_text: saved.temp_text
    };

    let serverFrameSize = 1920;
    let outSampleRate = 24000;

    // mic capture: use ScriptProcessor for simplicity, downsample to 24k, frame to serverFrameSize
    let micProc = audioCtx.createScriptProcessor(2048, 1, 1);
    let micBuf24 = [];
    let micBufLen = 0;

    function pushMic24(f32){
      for(let i=0;i<f32.length;i++){
        micBuf24.push(f32[i]);
      }
      micBufLen += f32.length;
    }

    function popFrame24(n){
      if(micBufLen < n) return null;
      const frame = new Float32Array(n);
      for(let i=0;i<n;i++){
        frame[i] = micBuf24[i];
      }
      micBuf24 = micBuf24.slice(n);
      micBufLen -= n;
      return frame;
    }

    micProc.onaudioprocess = (e) => {
      if(!ws || ws.readyState !== 1) return;
      const input = e.inputBuffer.getChannelData(0);
      const down = downsampleFloat32(input, sr, 24000);
      pushMic24(down);
      while(true){
        const fr = popFrame24(serverFrameSize);
        if(!fr) break;
        const pcm = float32ToPCM16(fr);
        const pkt = new Uint8Array(1 + pcm.byteLength);
        pkt[0] = 0x01;
        pkt.set(pcm, 1);
        ws.send(pkt);
      }
    };

    micSource.connect(micProc);
    // do not connect micProc to destination (avoid feedback)
    // micProc.connect(audioCtx.destination); // intentionally omitted

    ws.onopen = () => {
      ws.send(JSON.stringify(cfg));
      showToast("Connected. Streaming…");
    };

    ws.onmessage = (ev) => {
      if(typeof ev.data === "string"){
        let j = null;
        try{ j = JSON.parse(ev.data); }catch(_){}
        if(!j) return;
        if(j.type === "ready"){
          serverFrameSize = j.frame_size || 1920;
          outSampleRate = j.sample_rate || 24000;
          ringWrap.classList.add("visible");
          startViz();
        } else if(j.type === "need_load"){
          showToast(j.message || "Model needs load.");
          addCascadeText("Backend says: model not loaded. Go back and click 'Load Into GPU'.");
        } else if(j.type === "text"){
          if(j.delta) addCascadeText(j.delta);
        } else if(j.type === "error"){
          showToast("Error: " + (j.error || "unknown"));
          addCascadeText("Error: " + (j.error || "unknown"));
        }
        return;
      }

      // binary audio: 0x02 + pcm16 @ 24kHz
      const u8 = new Uint8Array(ev.data);
      if(!u8.length) return;
      if(u8[0] !== 0x02) return;
      const pcmBytes = u8.slice(1);
      const f32_24 = pcm16ToFloat32(pcmBytes);

      // resample from 24k to audioCtx sampleRate
      const f32 = downsampleFloat32(f32_24, 24000, sr);

      // push to output queue
      for(let i=0;i<f32.length;i++){
        outQ.push(f32[i]);
      }
      if(outQ.length > outQMax){
        outQ = outQ.slice(outQ.length - outQMax);
      }
    };

    ws.onclose = () => {
      showToast("Disconnected.");
    };

    ws.onerror = () => {
      showToast("WebSocket error.");
    };

    // exit handler
    exitBtn.onclick = () => stopTalking();
    startViz();
  }

  function stopTalking(){
    // stop ws + audio
    try{ if(ws) ws.close(); }catch(_){}
    ws = null;

    stopViz();

    try{
      if(scriptNodeIn) scriptNodeIn.disconnect();
      scriptNodeIn = null;
    }catch(_){}

    try{
      if(scriptNodeOut) scriptNodeOut.disconnect();
      scriptNodeOut = null;
    }catch(_){}

    try{
      if(micSource) micSource.disconnect();
      micSource = null;
    }catch(_){}

    try{
      if(micStream){
        for(const tr of micStream.getTracks()) tr.stop();
        micStream = null;
      }
    }catch(_){}

    try{
      if(audioCtx){
        audioCtx.close();
        audioCtx = null;
      }
    }catch(_){}

    outQ = [];
    talkView.classList.remove("active");
    ringWrap.classList.remove("visible");
    cascade.innerHTML = "";
  }

  startBtn.onclick = async () => {
    if(startBtn.classList.contains("disabled")){
      showToast("Start Talking is locked (download + load first).");
      return;
    }
    try{
      await startTalking();
    }catch(e){
      showToast("Start failed: " + (e.message || e));
      addCascadeText("Start failed: " + (e.message || e));
      stopTalking();
    }
  };

})();
</script>
</body>
</html>
"""


def main():
    ensure_venv_and_deps()
    ensure_personaplex_repo()
    ensure_personaplex_moshi_installed()

    # Discover voice prompts from repo
    global VOICE_PROMPTS
    VOICE_PROMPTS = find_voice_prompts(PERSONAPLEX_DIR)
    if not VOICE_PROMPTS:
        print("Warning: no NAT*/VAR*.pt voice prompts found in repo. Voice conditioning may be unavailable.")

    app = create_app()

    # Run with gevent + gevent-websocket for real WebSocket support.
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler

    print("\nPersonaPlex Eval server running:")
    print(f"  http://{HOST}:{PORT}")
    print("Open that in your browser.\n")

    server = pywsgi.WSGIServer((HOST, PORT), app, handler_class=WebSocketHandler)
    server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
