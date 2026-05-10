"""Lightweight frontend server. Serves UI, stores videos, proxies processing to GPU backend.
Runs on a cheap always-on instance (Cloud Run, e2-micro, etc.)
"""
import os
import subprocess
import time
import shutil
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

YT_DLP = shutil.which("yt-dlp") or os.path.join(BASE_DIR, "venv", "bin", "yt-dlp")

# GPU backend URL — set via env var or default to localhost
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5005")

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

@app.route("/download", methods=["POST"])
def download():
    data = request.json
    url = data.get("url", "")
    if not url:
        return jsonify({"error": "No URL provided"})
    video_id = str(_uuid.uuid4())[:8]
    out_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
    try:
        title_result = subprocess.run(
            [YT_DLP, "--print", "title", "--skip-download", "--no-playlist", url],
            capture_output=True, text=True, timeout=30
        )
        name = title_result.stdout.strip() or "YouTube video"
        subprocess.run(
            [YT_DLP, "-f", "best[height<=720][ext=mp4]/best[height<=720]/best",
             "--no-playlist", "-o", out_path, "--force-overwrites", url],
            check=True, capture_output=True, text=True, timeout=120
        )
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
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr[:500]})
    except Exception as e:
        return jsonify({"error": str(e)})


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
    return jsonify({
        "ok": True,
        "duration": current_video["duration"],
        "fps": current_video["fps"],
        "total_frames": current_video["total_frames"],
    })


# ── Frame processing (proxy to GPU backend) ──────────────────────────

@app.route("/frame/<mode>")
def frame(mode):
    path = current_video.get("path")
    if not path or not os.path.exists(path):
        return "No video loaded", 404

    t = float(request.args.get("t", 0))
    prompt = request.args.get("prompt", "")
    fps = current_video["fps"]

    # Read the frame locally
    cap = cv2.VideoCapture(path)
    frame_num = int(t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, raw_frame = cap.read()
    cap.release()

    if not ret:
        return "Frame not found", 404

    # Encode frame and send to GPU backend
    _, buf = cv2.imencode(".jpg", raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    try:
        backend_url = f"{BACKEND_URL}/process?mode={mode}"
        if prompt:
            backend_url += f"&prompt={prompt}"
        r = http_requests.post(backend_url, data=buf.tobytes(),
                               headers={"Content-Type": "image/jpeg"}, timeout=30)
        proc_ms = r.headers.get("X-Process-Ms", "?")
        return Response(r.content, mimetype="image/jpeg",
                        headers={
                            "Cache-Control": "no-store",
                            "X-Process-Ms": proc_ms,
                            "Access-Control-Expose-Headers": "X-Process-Ms",
                        })
    except Exception as e:
        # Backend unavailable — return raw frame
        return Response(buf.tobytes(), mimetype="image/jpeg",
                        headers={"X-Process-Ms": "-1"})


# ── MJPEG streaming (proxy to GPU backend) ───────────────────────────

_stream_stats = {"proc_ms": 0, "frame_num": 0, "fps": 0, "t0": 0, "count": 0}


@app.route("/stream_stats")
def stream_stats():
    return jsonify(_stream_stats)


def _generate_stream(mode, start_time=0, prompt=None):
    path = current_video.get("path")
    if not path or not os.path.exists(path):
        return

    cap = cv2.VideoCapture(path)
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / vid_fps

    if start_time > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * vid_fps))

    frame_num = int(start_time * vid_fps)
    backend_url = f"{BACKEND_URL}/process?mode={mode}"
    if prompt:
        backend_url += f"&prompt={prompt}"

    try:
        while cap.isOpened():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            try:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                r = http_requests.post(backend_url, data=buf.tobytes(),
                                       headers={"Content-Type": "image/jpeg"}, timeout=30)
                proc_ms = int(r.headers.get("X-Process-Ms", 0))

                # Update stats
                _stream_stats["proc_ms"] = proc_ms
                _stream_stats["frame_num"] = frame_num
                _stream_stats["count"] += 1
                now = time.time()
                if _stream_stats["t0"] == 0:
                    _stream_stats["t0"] = now
                elif now - _stream_stats["t0"] >= 1.0:
                    _stream_stats["fps"] = round(_stream_stats["count"] / (now - _stream_stats["t0"]), 1)
                    _stream_stats["count"] = 0
                    _stream_stats["t0"] = now

                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + r.content + b"\r\n")

                elapsed = time.time() - t0
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
            except Exception as e:
                print(f"[stream] Error: {e}")
                continue
            finally:
                frame_num += 1
    finally:
        cap.release()


@app.route("/stream/<mode>")
def stream(mode):
    start_time = float(request.args.get("t", 0))
    prompt = request.args.get("prompt", None)
    return Response(
        _generate_stream(mode, start_time=start_time, prompt=prompt),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── Backend status ───────────────────────────────────────────────────

@app.route("/backend_status")
def backend_status():
    available = _backend_available()
    return jsonify({"available": available, "url": BACKEND_URL})


if __name__ == "__main__":
    print(f"=== Frontend Server ===")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Backend available: {_backend_available()}")
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, threaded=True)
