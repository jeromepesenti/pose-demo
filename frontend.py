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

# Backend URLs
CPU_BACKEND_URL = "http://localhost:5005"  # always-on local CPU backend
GPU_BACKEND_URL = None  # set when a GPU backend registers
BACKEND_URL = os.environ.get("BACKEND_URL", CPU_BACKEND_URL)

import threading as _threading
_backend_lock = _threading.Lock()

# Lambda Labs API for on-demand GPU
LAMBDA_API_KEY = os.environ.get("LAMBDA_API_KEY", "")
LAMBDA_SSH_KEY_NAME = os.environ.get("LAMBDA_SSH_KEY_NAME", "macbook")
LAMBDA_INSTANCE_TYPE = os.environ.get("LAMBDA_INSTANCE_TYPE", "gpu_1x_a100_sxm4")
LAMBDA_REGION = os.environ.get("LAMBDA_REGION", "")  # auto-select if empty

_lambda_state = {
    "instance_id": None,
    "ip": None,
    "status": "off",       # off, starting, setup, ready, stopping
    "message": "",         # detailed status message
    "last_activity": 0,
    "idle_timeout": 900,   # 15 minutes
}

def _lambda_api(method, endpoint, data=None):
    headers = {"Authorization": f"Bearer {LAMBDA_API_KEY}"}
    if method == "GET":
        r = http_requests.get(f"https://cloud.lambdalabs.com/api/v1/{endpoint}", headers=headers, timeout=10)
    else:
        r = http_requests.post(f"https://cloud.lambdalabs.com/api/v1/{endpoint}", headers=headers, json=data, timeout=30)
    return r.json()


def _lambda_launch():
    """Launch a Lambda instance and set up the backend."""
    import subprocess
    try:
        _lambda_state["status"] = "starting"
        _lambda_state["message"] = "Finding available GPU..."

        # Find available region
        types = _lambda_api("GET", "instance-types")
        instance_info = types.get("data", {}).get(LAMBDA_INSTANCE_TYPE, {})
        regions = instance_info.get("regions_with_capacity_available", [])
        if not regions:
            _lambda_state["status"] = "off"
            _lambda_state["message"] = f"No capacity for {LAMBDA_INSTANCE_TYPE}"
            print(f"[Lambda] No capacity for {LAMBDA_INSTANCE_TYPE}")
            return False

        region = LAMBDA_REGION or regions[0]["name"]
        _lambda_state["message"] = f"Launching {LAMBDA_INSTANCE_TYPE} in {region}..."

        # Launch
        result = _lambda_api("POST", "instance-operations/launch", {
            "region_name": region,
            "instance_type_name": LAMBDA_INSTANCE_TYPE,
            "ssh_key_names": [LAMBDA_SSH_KEY_NAME],
            "quantity": 1,
        })
        instance_ids = result.get("data", {}).get("instance_ids", [])
        if not instance_ids:
            _lambda_state["status"] = "off"
            error_msg = result.get("error", {}).get("message", str(result))
            _lambda_state["message"] = f"Launch failed: {error_msg}"
            print(f"[Lambda] Launch failed: {result}")
            return False

        _lambda_state["instance_id"] = instance_ids[0]
        _lambda_state["message"] = "Waiting for instance to boot..."
        print(f"[Lambda] Launched instance {instance_ids[0]}, waiting for IP...")

        # Poll for IP
        import time as _time
        for attempt in range(60):
            _time.sleep(5)
            try:
                info = _lambda_api("GET", f"instances/{_lambda_state['instance_id']}")
                instance = info.get("data", {})
                ip = instance.get("ip")
                status = instance.get("status")
                _lambda_state["message"] = f"Instance status: {status or 'pending'}... ({attempt*5}s)"
                if ip and status == "active":
                    _lambda_state["ip"] = ip
                    break
            except Exception as e:
                _lambda_state["message"] = f"Polling... ({attempt*5}s)"
        else:
            _lambda_state["status"] = "off"
            _lambda_state["message"] = "Timed out waiting for instance"
            print("[Lambda] Timed out waiting for instance")
            return False

        ip = _lambda_state["ip"]
        print(f"[Lambda] Instance ready at {ip}")

        # Wait for SSH
        _lambda_state["status"] = "setup"
        _lambda_state["message"] = f"Waiting for SSH on {ip}..."
        for attempt in range(24):
            try:
                r = subprocess.run(
                    ["ssh", "-i", os.path.expanduser("~/.ssh/id_ed25519"), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                     f"ubuntu@{ip}", "echo ready"],
                    capture_output=True, timeout=10, text=True)
                if r.returncode == 0:
                    break
            except Exception:
                pass
            _lambda_state["message"] = f"Waiting for SSH... ({attempt*5}s)"
            _time.sleep(5)
        else:
            _lambda_state["status"] = "off"
            _lambda_state["message"] = "SSH connection failed"
            return False

        # Copy files
        _lambda_state["message"] = "Copying project files..."
        base = os.path.dirname(os.path.abspath(__file__))
        files = [f for f in [
            f"{base}/backend_gpu.py", f"{base}/controlnet_gpu.py",
            f"{base}/setup_lambda.sh"
        ] if os.path.exists(f)]
        subprocess.run(
            ["scp", "-i", os.path.expanduser("~/.ssh/id_ed25519"), "-o", "StrictHostKeyChecking=no"] + files +
            [f"ubuntu@{ip}:~/pose-demo/"],
            capture_output=True, timeout=60)

        # Run setup
        _lambda_state["message"] = "Installing dependencies (~2 min)..."
        r = subprocess.run(
            ["ssh", "-i", os.path.expanduser("~/.ssh/id_ed25519"), "-o", "StrictHostKeyChecking=no", f"ubuntu@{ip}",
             "bash ~/pose-demo/setup_lambda.sh"],
            capture_output=True, timeout=600, text=True)
        if r.returncode != 0:
            print(f"[Lambda] Setup failed: {r.stderr[-500:]}")
            _lambda_state["message"] = "Setup failed — check logs"

        # Start backend
        _lambda_state["message"] = "Starting backend server..."
        instance_id = _lambda_state["instance_id"]
        subprocess.run(
            ["ssh", "-i", os.path.expanduser("~/.ssh/id_ed25519"), "-o", "StrictHostKeyChecking=no", f"ubuntu@{ip}",
             f"cd ~/pose-demo && LAMBDA_API_KEY={LAMBDA_API_KEY} LAMBDA_INSTANCE_ID={instance_id} "
             f"IDLE_TIMEOUT={_lambda_state['idle_timeout']} "
             "nohup python3 backend_gpu.py > backend.log 2>&1 &"],
            capture_output=True, timeout=10)
        _time.sleep(8)

        # Verify backend is running
        _lambda_state["message"] = "Verifying backend..."
        try:
            r = http_requests.get(f"http://{ip}:5005/", timeout=5)
            if r.status_code != 200:
                raise Exception(f"Backend returned {r.status_code}")
        except Exception as e:
            _lambda_state["message"] = f"Backend failed to start: {e}"
            print(f"[Lambda] Backend verification failed: {e}")
            # Don't return false — it might just need more time

        # Connect frontend
        global BACKEND_URL
        BACKEND_URL = f"http://{ip}:5005"
        _lambda_state["status"] = "ready"
        _lambda_state["last_activity"] = time.time()
        _lambda_state["message"] = ""
        print(f"[Lambda] Backend ready at {BACKEND_URL}")
        return True

    except Exception as e:
        _lambda_state["status"] = "off"
        _lambda_state["message"] = f"Error: {str(e)}"
        print(f"[Lambda] Launch error: {e}")
        import traceback; traceback.print_exc()
        return False


def _lambda_terminate():
    """Terminate the Lambda instance."""
    if not _lambda_state["instance_id"]:
        return
    _lambda_state["status"] = "stopping"
    result = _lambda_api("POST", "instance-operations/terminate", {
        "instance_ids": [_lambda_state["instance_id"]],
    })
    print(f"[Lambda] Terminated: {result}")
    global BACKEND_URL
    BACKEND_URL = "http://localhost:5005"
    _lambda_state["instance_id"] = None
    _lambda_state["ip"] = None
    _lambda_state["status"] = "off"


def _idle_monitor():
    """Background thread: terminate Lambda instance after idle timeout."""
    import time
    while True:
        time.sleep(30)
        if (_lambda_state["status"] == "ready" and
            _lambda_state["last_activity"] > 0 and
            time.time() - _lambda_state["last_activity"] > _lambda_state["idle_timeout"]):
            print(f"[Lambda] Idle for {_lambda_state['idle_timeout']}s, terminating...")
            _lambda_terminate()

if LAMBDA_API_KEY:
    _idle_thread = _threading.Thread(target=_idle_monitor, daemon=True)
    _idle_thread.start()


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

    # Track activity for idle timeout
    _lambda_state["last_activity"] = time.time()

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

@app.route("/gpu/status")
def gpu_status():
    return jsonify(_lambda_state)


@app.route("/gpu/start", methods=["POST"])
def gpu_start():
    if not LAMBDA_API_KEY:
        return jsonify({"error": "No LAMBDA_API_KEY configured"})
    if _lambda_state["status"] not in ("off",):
        return jsonify({"error": f"GPU is {_lambda_state['status']}"})
    # Launch in background thread
    _threading.Thread(target=_lambda_launch, daemon=True).start()
    return jsonify({"ok": True, "status": "starting"})


@app.route("/gpu/stop", methods=["POST"])
def gpu_stop():
    if not LAMBDA_API_KEY:
        return jsonify({"error": "No LAMBDA_API_KEY configured"})
    _lambda_terminate()
    return jsonify({"ok": True, "status": "off"})


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
