"""Lightweight frontend server. Serves UI, stores videos, proxies processing to GPU backend.
Runs on a cheap always-on instance (Cloud Run, e2-micro, etc.)
"""
import os
import time
import json as _json
import uuid as _uuid
from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import numpy as np
import requests as http_requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
VIDEOS_DIR = os.path.join(BASE_DIR, "saved_videos")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)


# Backend URLs
CPU_BACKEND_URL = "http://localhost:5005"  # always-on local CPU backend
GPU_BACKEND_URL = None  # set when a GPU backend registers
BACKEND_URL = os.environ.get("BACKEND_URL", CPU_BACKEND_URL)

import threading as _threading
_backend_lock = _threading.Lock()



def _backend_health_check():
    """Periodically check GPU backend. Fall back to CPU if GPU is down."""
    global BACKEND_URL
    import time as _t
    while True:
        _t.sleep(10)
        with _backend_lock:
            if GPU_BACKEND_URL and BACKEND_URL == GPU_BACKEND_URL:
                try:
                    r = http_requests.get(f"{GPU_BACKEND_URL}/", timeout=3)
                    if r.status_code != 200:
                        raise Exception(f"status {r.status_code}")
                except Exception:
                    print(f"[Backend] GPU backend offline, falling back to CPU")
                    BACKEND_URL = CPU_BACKEND_URL
            elif GPU_BACKEND_URL and BACKEND_URL == CPU_BACKEND_URL:
                # GPU was registered but we fell back — check if it's back
                try:
                    r = http_requests.get(f"{GPU_BACKEND_URL}/", timeout=3)
                    if r.status_code == 200:
                        print(f"[Backend] GPU backend is back, switching to {GPU_BACKEND_URL}")
                        BACKEND_URL = GPU_BACKEND_URL
                except Exception:
                    pass

_health_thread = _threading.Thread(target=_backend_health_check, daemon=True)
_health_thread.start()

current_video = {"path": None, "fps": 30, "duration": 0, "total_frames": 0}

MAX_SAVED_VIDEOS = 10
VIDEOS_INDEX = os.path.join(VIDEOS_DIR, "index.json")


def _load_video_index():
    if os.path.exists(VIDEOS_INDEX):
        with open(VIDEOS_INDEX) as f:
            return _json.load(f)
    return []


def _save_video_index(index):
    with open(VIDEOS_INDEX, "w") as f:
        _json.dump(index, f)


def _add_to_video_index(video_id, name):
    video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
    index = _load_video_index()
    index = [v for v in index if v["id"] != video_id]
    thumb_path = os.path.join(VIDEOS_DIR, f"{video_id}_thumb.jpg")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, total // 10))
    ret, frame = cap.read()
    cap.release()
    if ret:
        thumb = cv2.resize(frame, (320, 180))
        cv2.imwrite(thumb_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
    index.insert(0, {"id": video_id, "name": name})
    while len(index) > MAX_SAVED_VIDEOS:
        old = index.pop()
        for ext in [".mp4", "_thumb.jpg"]:
            p = os.path.join(VIDEOS_DIR, f"{old['id']}{ext}")
            if os.path.exists(p):
                os.remove(p)
    _save_video_index(index)


def _load_saved_video(video_id):
    video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        return None
    current_video["path"] = video_path
    cap = cv2.VideoCapture(video_path)
    current_video["fps"] = cap.get(cv2.CAP_PROP_FPS) or 30
    current_video["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_video["duration"] = current_video["total_frames"] / current_video["fps"]
    cap.release()
    return current_video


# ── Check backend health ─────────────────────────────────────────────

def _backend_available():
    try:
        r = http_requests.get(f"{BACKEND_URL}/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ── HTML (same as server.py but fetches modes from backend) ──────────

HTML = open(os.path.join(BASE_DIR, "frontend.html")).read() if os.path.exists(os.path.join(BASE_DIR, "frontend.html")) else """
<!DOCTYPE html>
<html><body><h1>Frontend server running</h1>
<p>Create frontend.html or set up the full UI.</p>
<p>Backend: """ + BACKEND_URL + """</p></body></html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


# ── Modes (proxy from backend) ───────────────────────────────────────

@app.route("/modes")
def modes():
    try:
        r = http_requests.get(f"{BACKEND_URL}/modes", timeout=5)
        return Response(r.content, mimetype="application/json")
    except Exception:
        return jsonify(["pose", "shadow"])  # fallback


# ── Video management ─────────────────────────────────────────────────

@app.route("/upload_video", methods=["POST"])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"})
    f = request.files['video']
    video_id = str(_uuid.uuid4())[:8]
    name = os.path.splitext(f.filename)[0] if f.filename else "Uploaded video"
    out_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
    f.save(out_path)
    current_video["path"] = out_path
    cap = cv2.VideoCapture(out_path)
    current_video["fps"] = cap.get(cv2.CAP_PROP_FPS) or 30
    current_video["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_video["duration"] = current_video["total_frames"] / current_video["fps"]
    cap.release()
    _add_to_video_index(video_id, name)
    return jsonify({
        "ok": True,
        "duration": current_video["duration"],
        "fps": current_video["fps"],
        "total_frames": current_video["total_frames"],
    })


@app.route("/videos")
def list_videos():
    return jsonify(_load_video_index())


@app.route("/thumbnail/<video_id>")
def thumbnail(video_id):
    thumb_path = os.path.join(VIDEOS_DIR, f"{video_id}_thumb.jpg")
    if not os.path.exists(thumb_path):
        return "Not found", 404
    with open(thumb_path, "rb") as f:
        return Response(f.read(), mimetype="image/jpeg")


@app.route("/load_video/<video_id>")
def load_video(video_id):
    result = _load_saved_video(video_id)
    if result is None:
        return jsonify({"error": "Video not found"})
    _backend_video_synced["path"] = None  # force re-sync
    return jsonify({
        "ok": True,
        "duration": current_video["duration"],
        "fps": current_video["fps"],
        "total_frames": current_video["total_frames"],
    })


# ── Sync video to backend ─────────────────────────────────────────────

_backend_video_synced = {"path": None}  # track which video the backend has


def _sync_video_to_backend():
    """Upload current video to the GPU backend if not already synced."""
    path = current_video.get("path")
    if not path or not os.path.exists(path):
        return False
    if _backend_video_synced["path"] == path:
        return True  # already synced

    try:
        with open(path, "rb") as f:
            r = http_requests.post(
                f"{BACKEND_URL}/upload_video",
                files={"video": ("video.mp4", f, "video/mp4")},
                timeout=120)
        if r.status_code == 200 and r.json().get("ok"):
            _backend_video_synced["path"] = path
            print(f"[Sync] Video synced to backend")
            return True
    except Exception as e:
        print(f"[Sync] Failed to sync video: {e}")
    return False


# ── Frame processing (proxy to backend which has the video) ──────────

@app.route("/sync_video", methods=["POST"])
def sync_video():
    """Explicitly sync the current video to the active backend."""
    _backend_video_synced["path"] = None  # force re-sync
    if _backend_available():
        ok = _sync_video_to_backend()
        return jsonify({"ok": ok})
    # Also sync to CPU backend
    path = current_video.get("path")
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                http_requests.post(f"{CPU_BACKEND_URL}/upload_video",
                                   files={"video": ("video.mp4", f)}, timeout=120)
        except Exception:
            pass
    return jsonify({"ok": True})


@app.route("/frame/<mode>")
def frame(mode):
    if not current_video.get("path"):
        return "No video loaded", 404

    # Ensure backend has the video
    if _backend_available() and _sync_video_to_backend():
        # Proxy to backend — it reads the frame itself, no per-frame transfer
        try:
            backend_url = f"{BACKEND_URL}/frame/{mode}?{request.query_string.decode()}"
            r = http_requests.get(backend_url, timeout=30)
            return Response(r.content, mimetype="image/jpeg",
                            headers={
                                "Cache-Control": "no-store",
                                "X-Process-Ms": r.headers.get("X-Process-Ms", "?"),
                                "Access-Control-Expose-Headers": "X-Process-Ms",
                            })
        except Exception as e:
            print(f"[Frame] Backend error: {e}, falling back to CPU")

    # Fallback: process locally on CPU backend
    try:
        backend_url = f"{CPU_BACKEND_URL}/frame/{mode}?{request.query_string.decode()}"
        r = http_requests.get(backend_url, timeout=30)
        return Response(r.content, mimetype="image/jpeg",
                        headers={
                            "Cache-Control": "no-store",
                            "X-Process-Ms": r.headers.get("X-Process-Ms", "?"),
                            "Access-Control-Expose-Headers": "X-Process-Ms",
                        })
    except Exception:
        return "Backend unavailable", 503


# ── MJPEG streaming (proxy to backend) ───────────────────────────────

@app.route("/stream_stats")
def stream_stats():
    # Proxy to whichever backend is active
    try:
        r = http_requests.get(f"{BACKEND_URL}/stream_stats", timeout=2)
        return Response(r.content, mimetype="application/json")
    except Exception:
        return jsonify({"proc_ms": 0, "fps": 0})


@app.route("/stream/<mode>")
def stream(mode):
    if not current_video.get("path"):
        return "No video loaded", 404

    # Ensure backend has the video
    backend = BACKEND_URL
    if _backend_available():
        _sync_video_to_backend()
    else:
        backend = CPU_BACKEND_URL

    # Proxy the MJPEG stream from backend
    try:
        backend_url = f"{backend}/stream/{mode}?{request.query_string.decode()}"
        r = http_requests.get(backend_url, stream=True, timeout=300)
        return Response(
            r.iter_content(chunk_size=8192),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as e:
        print(f"[Stream] Error: {e}")
        return "Stream unavailable", 503


# ── Backend status ───────────────────────────────────────────────────

@app.route("/backend_direct_url")
def backend_direct_url():
    """Return the URL the browser can use to reach the backend directly."""
    # If GPU backend is on localhost via tunnel, expose via the frontend's public IP
    if GPU_BACKEND_URL and "localhost" in GPU_BACKEND_URL:
        # Browser can reach it via the same host on the tunneled port
        port = GPU_BACKEND_URL.split(":")[-1]
        return jsonify({"url": f":{port}"})  # relative to current host
    elif GPU_BACKEND_URL:
        return jsonify({"url": GPU_BACKEND_URL})
    return jsonify({"url": ""})


@app.route("/register_backend", methods=["POST"])
def register_backend():
    """Called by a GPU backend to announce itself."""
    global BACKEND_URL, GPU_BACKEND_URL
    data = request.json
    url = data.get("url", "")
    if not url:
        return jsonify({"error": "No URL provided"})
    with _backend_lock:
        GPU_BACKEND_URL = url
        BACKEND_URL = url
    print(f"[Backend] GPU backend registered: {url}")
    return jsonify({"ok": True, "url": BACKEND_URL})


@app.route("/set_backend", methods=["POST"])
def set_backend():
    global BACKEND_URL, GPU_BACKEND_URL
    data = request.json
    url = data.get("url", "")
    if not url:
        return jsonify({"error": "No URL provided"})
    with _backend_lock:
        if url == CPU_BACKEND_URL:
            GPU_BACKEND_URL = None
        else:
            GPU_BACKEND_URL = url
        BACKEND_URL = url
    return jsonify({"ok": True, "url": BACKEND_URL})


@app.route("/backend_status")
def backend_status():
    available = _backend_available()
    cuda = False
    backend_type = "cpu"
    if available:
        try:
            r = http_requests.get(f"{BACKEND_URL}/", timeout=2)
            data = r.json()
            cuda = data.get("cuda", False)
            backend_type = "gpu" if cuda else "cpu"
        except Exception:
            pass
    return jsonify({
        "available": available,
        "url": BACKEND_URL,
        "cuda": cuda,
        "type": backend_type,
        "gpu_url": GPU_BACKEND_URL,
        "cpu_url": CPU_BACKEND_URL,
    })


if __name__ == "__main__":
    print(f"=== Frontend Server ===")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Backend available: {_backend_available()}")
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, threaded=True)
