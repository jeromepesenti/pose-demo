import os
import subprocess
import time
import threading
import traceback
from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions,
    ImageSegmenter, ImageSegmenterOptions,
    FaceLandmarker, FaceLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
    ObjectDetector, ObjectDetectorOptions,
    FaceDetector, FaceDetectorOptions,
)
from rtmlib import Body, Wholebody

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

current_video = {"path": None, "fps": 30, "duration": 0, "total_frames": 0}
YT_DLP = os.path.join(BASE_DIR, "venv", "bin", "yt-dlp")

# Processor cache: keep one processor per mode alive to avoid reload cost
_processor_cache = {}
_processor_lock = threading.Lock()

MODES = {
    "pose":          {"label": "Pose Estimation (MediaPipe)",   "group": "MediaPipe"},
    "shadow":        {"label": "Shadow Silhouette",              "group": "MediaPipe"},
    "face_mesh":     {"label": "Face Mesh",                      "group": "MediaPipe"},
    "hands":         {"label": "Hand Tracking",                  "group": "MediaPipe"},
    "face_detect":   {"label": "Face Detection",                 "group": "MediaPipe"},
    "objects":       {"label": "Object Detection",               "group": "MediaPipe"},
    "rtmpose_body":  {"label": "RTMPose Body",                   "group": "RTMPose (MMPose)"},
    "rtmpose_whole": {"label": "RTMPose Wholebody (133 pts)",    "group": "RTMPose (MMPose)"},
}

# ── Connection definitions ───────────────────────────────────────────

POSE_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
    (15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

FACE_OVAL = [
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),
    (356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),
    (379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),
    (149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),
    (234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10),
]
LIPS = [
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),
    (321,375),(375,291),(291,409),(409,270),(270,269),(269,267),(267,0),
    (0,37),(37,39),(39,40),(40,185),(185,61),
]
LEFT_EYE = [(33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),(155,133),(133,173),(173,157),(157,158),(158,159),(159,160),(160,161),(161,246),(246,33)]
RIGHT_EYE = [(362,382),(382,381),(381,380),(380,374),(374,373),(373,390),(390,249),(249,263),(263,466),(466,388),(388,387),(387,386),(386,385),(385,384),(384,398),(398,362)]
LEFT_BROW = [(70,63),(63,105),(105,66),(66,107),(107,55),(55,65),(65,52),(52,53),(53,46),(46,70)]
RIGHT_BROW = [(300,293),(293,334),(334,296),(296,336),(336,285),(285,295),(295,282),(282,283),(283,276),(276,300)]
FACE_CONNECTIONS = FACE_OVAL + LIPS + LEFT_EYE + RIGHT_EYE + LEFT_BROW + RIGHT_BROW

RTMPOSE_BODY_CONNECTIONS = [
    (0,1),(0,14),(0,15),(1,2),(1,5),(1,8),(1,11),
    (2,3),(3,4),(5,6),(6,7),(8,9),(9,10),(11,12),(12,13),
    (14,16),(15,17),
]

RTMPOSE_WHOLEBODY_BODY = [
    (0,1),(0,2),(1,3),(2,4),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
RTMPOSE_WHOLEBODY_FEET = [(15,17),(15,18),(15,19),(16,20),(16,21),(16,22)]
RTMPOSE_WHOLEBODY_FACE = [(23+i, 23+i+1) for i in range(16)] + \
                          [(23+17+i, 23+17+i+1) for i in range(4)] + \
                          [(23+22+i, 23+22+i+1) for i in range(4)] + \
                          [(23+27+i, 23+27+i+1) for i in range(4)] + \
                          [(23+36+i, 23+36+i+1) for i in range(5)] + \
                          [(23+42+i, 23+42+i+1) for i in range(5)] + \
                          [(23+48+i, 23+48+i+1) for i in range(11)] + \
                          [(23+60+i, 23+60+i+1) for i in range(7)]
RTMPOSE_WHOLEBODY_LHAND = [(91+i, 91+i+1) for i in range(3)] + \
                           [(91, 91+4)] + [(91+4+i, 91+4+i+1) for i in range(3)] + \
                           [(91, 91+8)] + [(91+8+i, 91+8+i+1) for i in range(3)] + \
                           [(91, 91+12)] + [(91+12+i, 91+12+i+1) for i in range(3)] + \
                           [(91, 91+16)] + [(91+16+i, 91+16+i+1) for i in range(3)]
RTMPOSE_WHOLEBODY_RHAND = [(112+i, 112+i+1) for i in range(3)] + \
                           [(112, 112+4)] + [(112+4+i, 112+4+i+1) for i in range(3)] + \
                           [(112, 112+8)] + [(112+8+i, 112+8+i+1) for i in range(3)] + \
                           [(112, 112+12)] + [(112+12+i, 112+12+i+1) for i in range(3)] + \
                           [(112, 112+16)] + [(112+16+i, 112+16+i+1) for i in range(3)]

COLORS = [
    (233, 105, 0), (15, 52, 96), (22, 199, 154), (245, 166, 35),
    (189, 16, 224), (80, 227, 194), (255, 107, 107), (78, 205, 196),
]

# ── HTML ─────────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YouTube Vision Demo</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #1a1a2e; color: #eee;
    font-family: system-ui, sans-serif;
    min-height: 100vh;
  }
  #layout {
    display: flex; min-height: 100vh;
  }

  /* ── Sidebar ── */
  #sidebar {
    width: 260px; min-width: 260px;
    background: #16213e;
    padding: 16px;
    display: flex; flex-direction: column; gap: 12px;
    border-right: 1px solid #0f3460;
    overflow-y: auto;
  }
  #sidebar h2 {
    font-size: 0.9rem; color: #aaa;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 4px;
  }
  .video-card {
    cursor: pointer; border-radius: 8px;
    overflow: hidden; background: #1a1a2e;
    border: 2px solid transparent;
    transition: border-color 0.2s, transform 0.15s;
  }
  .video-card:hover { border-color: #e94560; transform: scale(1.02); }
  .video-card.active { border-color: #e94560; }
  .video-card img {
    width: 100%; aspect-ratio: 16/9; object-fit: cover; display: block;
  }
  .video-card .card-title {
    padding: 6px 8px; font-size: 0.75rem; color: #ccc;
    line-height: 1.3;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }

  /* ── Main ── */
  #main {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; padding: 20px;
    overflow-y: auto;
  }
  h1 { margin-bottom: 10px; font-size: 1.4rem; }
  #info { font-size: 0.85rem; color: #aaa; margin-bottom: 16px; }
  .row {
    display: flex; gap: 8px; margin-bottom: 12px;
    width: 100%; max-width: 800px; align-items: center;
  }
  #url-input {
    flex: 1; padding: 10px 14px; font-size: 1rem;
    border: 1px solid #333; border-radius: 6px;
    background: #16213e; color: #eee; outline: none;
  }
  #url-input:focus { border-color: #e94560; }
  select {
    padding: 10px 14px; font-size: 1rem;
    border: 1px solid #333; border-radius: 6px;
    background: #16213e; color: #eee; outline: none;
    cursor: pointer; flex: 1;
  }
  button {
    padding: 10px 24px; font-size: 1rem; border: none;
    border-radius: 6px; background: #e94560; color: #fff; cursor: pointer;
    white-space: nowrap;
  }
  button:hover { background: #c73650; }
  button:disabled { background: #555; cursor: wait; }
  #status { font-size: 0.9rem; color: #aaa; margin-bottom: 12px; min-height: 1.2em; }
  #output { text-align: center; width: 100%; max-width: 800px; }
  #output img {
    border-radius: 8px; width: 100%; background: #000; display: block;
  }
  #controls {
    display: flex; align-items: center; gap: 10px;
    margin-top: 10px; width: 100%;
  }
  #play-btn {
    padding: 8px 16px; font-size: 1.2rem; min-width: 44px;
  }
  #slider {
    flex: 1; height: 6px; cursor: pointer;
    accent-color: #e94560;
  }
  #time-display {
    font-size: 0.85rem; color: #aaa; min-width: 100px; text-align: right;
    font-variant-numeric: tabular-nums;
  }

  @media (max-width: 768px) {
    #layout { flex-direction: column; }
    #sidebar {
      width: 100%; min-width: unset;
      flex-direction: row; overflow-x: auto;
      border-right: none; border-bottom: 1px solid #0f3460;
      padding: 10px;
    }
    #sidebar h2 { display: none; }
    .video-card { min-width: 160px; }
  }
</style>
</head>
<body>

<div id="layout">

  <!-- Sidebar with example videos -->
  <div id="sidebar">
    <h2>Example Videos</h2>
    <div class="video-card" onclick="pickVideo('https://www.youtube.com/watch?v=aqz7rZ3Ys4E', this)">
      <img src="https://i.ytimg.com/vi/aqz7rZ3Ys4E/mqdefault.jpg" alt="Dance Moves">
      <div class="card-title">Best of Favorite Dance Moves (2024)</div>
    </div>
    <div class="video-card" onclick="pickVideo('https://www.youtube.com/watch?v=iUkX7y_0dsQ', this)">
      <img src="https://i.ytimg.com/vi/iUkX7y_0dsQ/mqdefault.jpg" alt="Female Dancer">
      <div class="card-title">Female Dancer White Background</div>
    </div>
    <div class="video-card" onclick="pickVideo('https://www.youtube.com/watch?v=r2jd613Rdhs', this)">
      <img src="https://i.ytimg.com/vi/r2jd613Rdhs/mqdefault.jpg" alt="iDancers">
      <div class="card-title">iDancers - Beautiful Girls Dancing</div>
    </div>
  </div>

  <!-- Main content -->
  <div id="main">
    <h1>YouTube Vision Demo</h1>
    <div id="info">Pick an example video or paste a YouTube URL</div>

    <div class="row">
      <input id="url-input" type="text" placeholder="Paste YouTube URL here...">
      <button id="go-btn" onclick="go()">Go</button>
    </div>

    <div class="row">
      <select id="mode-select">
        <optgroup label="MediaPipe">
          <option value="pose">Pose Estimation (MediaPipe)</option>
          <option value="shadow">Shadow Silhouette</option>
          <option value="face_mesh">Face Mesh</option>
          <option value="hands">Hand Tracking</option>
          <option value="face_detect">Face Detection</option>
          <option value="objects">Object Detection</option>
        </optgroup>
        <optgroup label="RTMPose (MMPose)">
          <option value="rtmpose_body">RTMPose Body</option>
          <option value="rtmpose_whole">RTMPose Wholebody (133 pts)</option>
        </optgroup>
      </select>
    </div>

    <div id="status"></div>

    <div id="output" style="display:none;">
      <img id="frame-img">
      <div id="controls">
        <button id="play-btn" onclick="togglePlay()">&#9654;</button>
        <input id="slider" type="range" min="0" max="1000" value="0">
        <span id="time-display">0:00 / 0:00</span>
      </div>
    </div>
  </div>

</div>

<script>
let currentUrl = '';
let playing = false;
let duration = 0;
let fps = 30;
let currentTime = 0;
let pendingFrame = false;
let rafId = null;
let seeking = false;

const img = document.getElementById('frame-img');
const slider = document.getElementById('slider');
const playBtn = document.getElementById('play-btn');
const timeDisplay = document.getElementById('time-display');
const statusEl = document.getElementById('status');
const output = document.getElementById('output');

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + ':' + String(sec).padStart(2, '0');
}

function getMode() {
  return document.getElementById('mode-select').value;
}

function pickVideo(url, card) {
  document.getElementById('url-input').value = url;
  document.querySelectorAll('.video-card').forEach(c => c.classList.remove('active'));
  if (card) card.classList.add('active');
  go();
}

async function go() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) return;
  const btn = document.getElementById('go-btn');
  btn.disabled = true;
  stop();
  output.style.display = 'none';

  // Highlight matching sidebar card
  document.querySelectorAll('.video-card').forEach(c => {
    const onclick = c.getAttribute('onclick') || '';
    if (onclick.includes(url)) c.classList.add('active');
    else c.classList.remove('active');
  });

  if (url !== currentUrl) {
    statusEl.textContent = 'Downloading video...';
    try {
      const resp = await fetch('/download', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({url})
      });
      const data = await resp.json();
      if (data.error) {
        statusEl.textContent = 'Error: ' + data.error;
        btn.disabled = false;
        return;
      }
      currentUrl = url;
      duration = data.duration;
      fps = data.fps;
    } catch (e) {
      statusEl.textContent = 'Error: ' + e.message;
      btn.disabled = false;
      return;
    }
  }

  currentTime = 0;
  slider.value = 0;
  updateTimeDisplay();
  output.style.display = 'block';
  statusEl.textContent = 'Ready — press play';
  btn.disabled = false;

  fetchFrame(0);
}

function fetchFrame(t) {
  if (pendingFrame) return;
  pendingFrame = true;
  const mode = getMode();
  const newImg = new Image();
  newImg.onload = () => {
    img.src = newImg.src;
    pendingFrame = false;
  };
  newImg.onerror = () => {
    pendingFrame = false;
  };
  newImg.src = '/frame/' + mode + '?t=' + t.toFixed(3) + '&_=' + Date.now();
}

function updateTimeDisplay() {
  timeDisplay.textContent = fmtTime(currentTime) + ' / ' + fmtTime(duration);
}

function togglePlay() {
  if (playing) {
    stop();
  } else {
    play();
  }
}

function play() {
  playing = true;
  playBtn.innerHTML = '&#9646;&#9646;';
  statusEl.textContent = 'Playing — ' + document.getElementById('mode-select').selectedOptions[0].text;
  let lastTs = performance.now();

  function tick(now) {
    if (!playing) return;
    const dt = (now - lastTs) / 1000;
    lastTs = now;
    currentTime += dt;

    if (currentTime >= duration) {
      currentTime = duration;
      stop();
      return;
    }

    if (!seeking) {
      slider.value = Math.round((currentTime / duration) * 1000);
    }
    updateTimeDisplay();
    fetchFrame(currentTime);
    rafId = requestAnimationFrame(tick);
  }
  rafId = requestAnimationFrame(tick);
}

function stop() {
  playing = false;
  playBtn.innerHTML = '&#9654;';
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  if (statusEl.textContent.startsWith('Playing'))
    statusEl.textContent = 'Paused';
}

slider.addEventListener('input', () => {
  seeking = true;
  currentTime = (slider.value / 1000) * duration;
  updateTimeDisplay();
  fetchFrame(currentTime);
});

slider.addEventListener('change', () => {
  seeking = false;
  currentTime = (slider.value / 1000) * duration;
  updateTimeDisplay();
  fetchFrame(currentTime);
});

document.getElementById('mode-select').addEventListener('change', () => {
  if (currentUrl) {
    fetchFrame(currentTime);
    if (playing) {
      statusEl.textContent = 'Playing — ' + document.getElementById('mode-select').selectedOptions[0].text;
    }
  }
});
</script>
</body>
</html>
"""


# ── Video download ───────────────────────────────────────────────────

def download_video(url):
    out_path = os.path.join(DOWNLOAD_DIR, "video.mp4")
    try:
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
        return None
    except subprocess.CalledProcessError as e:
        return e.stderr[:500]
    except Exception as e:
        return str(e)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/download", methods=["POST"])
def download():
    data = request.json
    url = data.get("url", "")
    if not url:
        return jsonify({"error": "No URL provided"})
    err = download_video(url)
    if err:
        return jsonify({"error": err})
    return jsonify({
        "ok": True,
        "duration": current_video["duration"],
        "fps": current_video["fps"],
        "total_frames": current_video["total_frames"],
    })


# ── Drawing helpers ──────────────────────────────────────────────────

def draw_landmarks(frame, landmarks, connections, color, w, h, thickness=2, radius=3):
    pts = []
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        vis = lm.visibility if hasattr(lm, 'visibility') and lm.visibility else 1.0
        pts.append((px, py, vis))
    for a, b in connections:
        if a < len(pts) and b < len(pts):
            if pts[a][2] > 0.3 and pts[b][2] > 0.3:
                cv2.line(frame, pts[a][:2], pts[b][:2], color, thickness)
    for px, py, vis in pts:
        if vis > 0.3:
            cv2.circle(frame, (px, py), radius, color, -1)
            cv2.circle(frame, (px, py), radius, (255, 255, 255), 1)


def draw_keypoints_array(frame, keypoints, scores, connections, color,
                         threshold=0.3, thickness=2, radius=4):
    n = len(keypoints)
    for a, b in connections:
        if a < n and b < n and scores[a] > threshold and scores[b] > threshold:
            pt1 = (int(keypoints[a][0]), int(keypoints[a][1]))
            pt2 = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(frame, pt1, pt2, color, thickness)
    for i in range(n):
        if scores[i] > threshold:
            pt = (int(keypoints[i][0]), int(keypoints[i][1]))
            cv2.circle(frame, pt, radius, color, -1)
            cv2.circle(frame, pt, radius, (255, 255, 255), 1)


def draw_bbox(frame, bbox, label, color, w, h):
    x, y = int(bbox.origin_x), int(bbox.origin_y)
    bw, bh = int(bbox.width), int(bbox.height)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
    cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ── Processor creation & caching ─────────────────────────────────────

def get_processor(mode):
    with _processor_lock:
        if mode in _processor_cache:
            return _processor_cache[mode]
        processor = _create_processor(mode)
        _processor_cache[mode] = processor
        return processor


def _create_processor(mode):
    # MediaPipe modes use IMAGE mode for single-frame processing
    image_mode = mp.tasks.vision.RunningMode.IMAGE

    if mode == "pose":
        base = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "pose_landmarker_lite.task"))
        return PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=base, running_mode=image_mode, num_poses=6,
        ))
    elif mode == "shadow":
        base = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "selfie_multiclass.tflite"))
        return ImageSegmenter.create_from_options(ImageSegmenterOptions(
            base_options=base, running_mode=image_mode,
            output_category_mask=True, output_confidence_masks=True,
        ))
    elif mode == "face_mesh":
        base = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "face_landmarker.task"))
        return FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=base, running_mode=image_mode,
            num_faces=4, output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        ))
    elif mode == "hands":
        base = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "hand_landmarker.task"))
        return HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=base, running_mode=image_mode, num_hands=6,
        ))
    elif mode == "face_detect":
        base = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "face_detector.tflite"))
        return FaceDetector.create_from_options(FaceDetectorOptions(
            base_options=base, running_mode=image_mode,
        ))
    elif mode == "objects":
        base = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "efficientdet.tflite"))
        return ObjectDetector.create_from_options(ObjectDetectorOptions(
            base_options=base, running_mode=image_mode,
            max_results=10, score_threshold=0.3,
        ))
    elif mode == "rtmpose_body":
        return Body(mode='lightweight', to_openpose=True, backend='onnxruntime')
    elif mode == "rtmpose_whole":
        return Wholebody(mode='lightweight', to_openpose=False, backend='onnxruntime')


# ── Frame processing ─────────────────────────────────────────────────

def process_frame(mode, processor, frame, mp_image):
    h, w = frame.shape[:2]

    if mode == "pose":
        result = processor.detect(mp_image)
        out = frame.copy()
        if result.pose_landmarks:
            for i, person in enumerate(result.pose_landmarks):
                draw_landmarks(out, person, POSE_CONNECTIONS,
                               COLORS[i % len(COLORS)], w, h, thickness=3, radius=5)
        return out

    elif mode == "shadow":
        result = processor.segment(mp_image)
        if result.category_mask is not None:
            mask = np.squeeze(result.category_mask.numpy_view())
            condition = (mask > 0)
        elif result.confidence_masks:
            first = np.squeeze(result.confidence_masks[0].numpy_view())
            combined = np.zeros_like(first, dtype=np.float32)
            for i in range(1, len(result.confidence_masks)):
                combined += np.squeeze(result.confidence_masks[i].numpy_view())
            condition = (combined > 0.5)
        else:
            condition = np.zeros((h, w), dtype=bool)
        fg = np.full_like(frame, (60, 30, 30), dtype=np.uint8)
        bg = np.full_like(frame, (230, 220, 220), dtype=np.uint8)
        out = np.where(condition[:, :, np.newaxis], fg, bg).astype(np.uint8)
        return cv2.GaussianBlur(out, (7, 7), 0)

    elif mode == "face_mesh":
        result = processor.detect(mp_image)
        out = frame.copy()
        if result.face_landmarks:
            for i, face in enumerate(result.face_landmarks):
                draw_landmarks(out, face, FACE_CONNECTIONS,
                               COLORS[i % len(COLORS)], w, h, thickness=1, radius=1)
        return out

    elif mode == "hands":
        result = processor.detect(mp_image)
        out = frame.copy()
        if result.hand_landmarks:
            for i, hand in enumerate(result.hand_landmarks):
                draw_landmarks(out, hand, HAND_CONNECTIONS,
                               COLORS[i % len(COLORS)], w, h, thickness=2, radius=4)
        return out

    elif mode == "face_detect":
        result = processor.detect(mp_image)
        out = frame.copy()
        if result.detections:
            for i, det in enumerate(result.detections):
                color = COLORS[i % len(COLORS)]
                score = det.categories[0].score if det.categories else 0
                draw_bbox(out, det.bounding_box, f"Face {score:.0%}", color, w, h)
                for kp in det.keypoints or []:
                    cv2.circle(out, (int(kp.x * w), int(kp.y * h)), 4, color, -1)
        return out

    elif mode == "objects":
        result = processor.detect(mp_image)
        out = frame.copy()
        if result.detections:
            for i, det in enumerate(result.detections):
                color = COLORS[i % len(COLORS)]
                cat = det.categories[0] if det.categories else None
                label = f"{cat.category_name} {cat.score:.0%}" if cat else "?"
                draw_bbox(out, det.bounding_box, label, color, w, h)
        return out

    elif mode == "rtmpose_body":
        out = frame.copy()
        keypoints, scores = processor(frame)
        if keypoints is not None and len(keypoints) > 0:
            for i in range(len(keypoints)):
                draw_keypoints_array(out, keypoints[i], scores[i],
                                     RTMPOSE_BODY_CONNECTIONS,
                                     COLORS[i % len(COLORS)],
                                     threshold=0.3, thickness=3, radius=5)
        return out

    elif mode == "rtmpose_whole":
        out = frame.copy()
        keypoints, scores = processor(frame)
        if keypoints is not None and len(keypoints) > 0:
            for i in range(len(keypoints)):
                color = COLORS[i % len(COLORS)]
                kp, sc = keypoints[i], scores[i]
                draw_keypoints_array(out, kp, sc,
                                     RTMPOSE_WHOLEBODY_BODY + RTMPOSE_WHOLEBODY_FEET,
                                     color, threshold=0.3, thickness=3, radius=4)
                draw_keypoints_array(out, kp, sc, RTMPOSE_WHOLEBODY_FACE,
                                     (200, 200, 200), threshold=0.3, thickness=1, radius=1)
                draw_keypoints_array(out, kp, sc,
                                     RTMPOSE_WHOLEBODY_LHAND + RTMPOSE_WHOLEBODY_RHAND,
                                     (80, 227, 194), threshold=0.3, thickness=2, radius=3)
        return out

    return frame.copy()


# ── Single frame endpoint ────────────────────────────────────────────

@app.route("/frame/<mode>")
def frame(mode):
    if mode not in MODES:
        return "Invalid mode", 400

    path = current_video.get("path")
    if not path or not os.path.exists(path):
        return "No video loaded", 404

    t = float(request.args.get("t", 0))
    fps = current_video["fps"]

    cap = cv2.VideoCapture(path)
    frame_num = int(t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, raw_frame = cap.read()
    cap.release()

    if not ret:
        return "Frame not found", 404

    try:
        processor = get_processor(mode)
        is_rtm = mode.startswith("rtmpose")

        if is_rtm:
            out = process_frame(mode, processor, raw_frame, None)
        else:
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            out = process_frame(mode, processor, raw_frame, mp_image)

        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    except Exception as e:
        print(f"[{mode}] Frame error: {e}")
        traceback.print_exc()
        _, buf = cv2.imencode(".jpg", raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), mimetype="image/jpeg")


# ── Legacy MJPEG stream (kept for compatibility) ─────────────────────

def generate_frames(mode):
    path = current_video.get("path")
    if not path or not os.path.exists(path):
        return

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = 1.0 / fps

    is_rtm = mode.startswith("rtmpose")
    processor = get_processor(mode)

    try:
        while cap.isOpened():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            try:
                if is_rtm:
                    out = process_frame(mode, processor, frame, None)
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    out = process_frame(mode, processor, frame, mp_image)
                _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            except Exception as e:
                print(f"[{mode}] Frame error: {e}")
                traceback.print_exc()
                continue
            elapsed = time.time() - t0
            if elapsed < delay:
                time.sleep(delay - elapsed)
    finally:
        cap.release()


@app.route("/stream/<mode>")
def stream(mode):
    if mode not in MODES:
        return "Invalid mode", 400
    return Response(
        generate_frames(mode),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    print("Starting server at http://localhost:5001")
    print("Available modes:")
    for k, v in MODES.items():
        print(f"  [{v['group']}] {k}: {v['label']}")
    app.run(host="0.0.0.0", port=5001, threaded=True)
