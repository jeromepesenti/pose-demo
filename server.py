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
from ultralytics import YOLO

# ControlNet: only available with CUDA
_HAS_CUDA = False
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

if _HAS_CUDA:
    from controlnet_gpu import generate as controlnet_generate

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

current_video = {"path": None, "fps": 30, "duration": 0, "total_frames": 0}
import shutil
YT_DLP = shutil.which("yt-dlp") or os.path.join(BASE_DIR, "venv", "bin", "yt-dlp")

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
    "rtmpose_shadow":{"label": "RTMPose Body Shadow",            "group": "RTMPose (MMPose)"},
    "rtmpose_costume":{"label": "RTMPose Costume Transfer",      "group": "RTMPose (MMPose)"},
    "yolo_shadow":   {"label": "YOLO Shadow (per-person)",       "group": "YOLO"},
}

if _HAS_CUDA:
    MODES["controlnet"] = {"label": "ControlNet Pose (SD 1.5)", "group": "Generative"}

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
  #stats {
    display: flex; gap: 16px; justify-content: center;
    margin-top: 8px; font-size: 0.8rem; font-variant-numeric: tabular-nums;
  }
  .stat { padding: 4px 10px; border-radius: 4px; background: #16213e; }
  .stat-label { color: #777; }
  .stat-value { color: #eee; font-weight: 600; }
  .stat-good { color: #16c79a; }
  .stat-warn { color: #f5a623; }
  .stat-bad { color: #e94560; }
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
    <div class="video-card" onclick="pickVideo('https://www.youtube.com/watch?v=2DiQUX11YaY', this)">
      <img src="https://i.ytimg.com/vi/2DiQUX11YaY/mqdefault.jpg" alt="Flashmob">
      <div class="card-title">Crazy Uptown Funk Flashmob in Sydney</div>
    </div>
  </div>

  <!-- Main content -->
  <div id="main">
    <h1>YouTube Vision Demo</h1>
    <div id="info">Pick an example video or paste a YouTube URL</div>

    <div class="row">
      <input id="url-input" type="text" placeholder="Paste YouTube URL here...">
      <button id="go-btn" onclick="go()">Go</button>
      <span style="color:#555;">or</span>
      <label style="cursor:pointer;">
        <button onclick="document.getElementById('vid-upload').click(); return false;">Upload</button>
        <input id="vid-upload" type="file" accept="video/*" style="display:none;" onchange="uploadVideo(this)">
      </label>
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
          <option value="rtmpose_shadow">RTMPose Body Shadow</option>
          <option value="rtmpose_costume">RTMPose Costume Transfer</option>
        </optgroup>
        <optgroup label="YOLO">
          <option value="yolo_shadow">YOLO Shadow (per-person)</option>
        </optgroup>
        <optgroup label="Generative" id="gen-group" style="display:none;">
          <option value="controlnet">ControlNet Pose (SD 1.5)</option>
        </optgroup>
      </select>
    </div>

    <div class="row" id="prompt-row" style="display:none;">
      <input id="prompt-input" type="text" placeholder="Describe the person..." value="person dancing, professional photo, studio lighting, high quality"
        style="flex:1; padding:10px 14px; font-size:1rem; border:1px solid #333; border-radius:6px; background:#16213e; color:#eee; outline:none;">
    </div>

    <div class="row" id="ref-row" style="display:none;">
      <label style="flex:1; display:flex; align-items:center; gap:8px; cursor:pointer;
        padding:10px 14px; border:1px dashed #555; border-radius:6px; background:#16213e; color:#aaa;">
        <span id="ref-label">Upload reference person image...</span>
        <input id="ref-input" type="file" accept="image/*" style="display:none;" onchange="uploadRef(this)">
      </label>
      <img id="ref-preview" style="display:none; height:48px; border-radius:4px;">
    </div>

    <div id="status"></div>

    <div id="output" style="display:none;">
      <img id="frame-img">
      <div id="controls">
        <button id="play-btn" onclick="togglePlay()">&#9654;</button>
        <input id="slider" type="range" min="0" max="1000" value="0">
        <span id="time-display">0:00 / 0:00</span>
      </div>
      <div id="stats">
        <div class="stat"><span class="stat-label">Process: </span><span class="stat-value" id="stat-proc">—</span></div>
        <div class="stat"><span class="stat-label">Round-trip: </span><span class="stat-value" id="stat-rtt">—</span></div>
        <div class="stat"><span class="stat-label">FPS: </span><span class="stat-value" id="stat-fps">—</span></div>
        <div class="stat"><span class="stat-label">Status: </span><span class="stat-value" id="stat-status">—</span></div>
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

let refUploaded = false;

function getMode() {
  const mode = document.getElementById('mode-select').value;
  document.getElementById('ref-row').style.display =
    (mode === 'rtmpose_costume') ? 'flex' : 'none';
  document.getElementById('prompt-row').style.display =
    (mode === 'controlnet') ? 'flex' : 'none';
  return mode;
}

// Check if controlnet is available and show the option group
fetch('/modes').then(r => r.json()).then(modes => {
  if (modes.includes('controlnet')) {
    document.getElementById('gen-group').style.display = '';
  }
});

async function uploadRef(input) {
  if (!input.files[0]) return;
  const formData = new FormData();
  formData.append('image', input.files[0]);
  document.getElementById('ref-label').textContent = 'Uploading...';
  try {
    const resp = await fetch('/upload_reference', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.error) {
      document.getElementById('ref-label').textContent = 'Error: ' + data.error;
      return;
    }
    document.getElementById('ref-label').textContent = data.filename + ' (' + data.parts + ')';
    document.getElementById('ref-preview').src = '/reference_preview?' + Date.now();
    document.getElementById('ref-preview').style.display = 'block';
    refUploaded = true;
    if (currentUrl) fetchFrame(currentTime);
  } catch (e) {
    document.getElementById('ref-label').textContent = 'Upload failed';
  }
}

function pickVideo(url, card) {
  document.getElementById('url-input').value = url;
  document.querySelectorAll('.video-card').forEach(c => c.classList.remove('active'));
  if (card) card.classList.add('active');
  go();
}

async function uploadVideo(input) {
  if (!input.files[0]) return;
  const btn = document.getElementById('go-btn');
  btn.disabled = true;
  stop();
  output.style.display = 'none';
  statusEl.textContent = 'Uploading video (' + (input.files[0].size / 1e6).toFixed(1) + ' MB)...';

  const formData = new FormData();
  formData.append('video', input.files[0]);
  try {
    const resp = await fetch('/upload_video', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.error) {
      statusEl.textContent = 'Error: ' + data.error;
      btn.disabled = false;
      return;
    }
    currentUrl = '__uploaded__';
    duration = data.duration;
    fps = data.fps;
    currentTime = 0;
    slider.value = 0;
    updateTimeDisplay();
    output.style.display = 'block';
    getMode();
    statusEl.textContent = 'Ready — press play';
    btn.disabled = false;
    fetchFrame(0);
  } catch (e) {
    statusEl.textContent = 'Upload failed: ' + e.message;
    btn.disabled = false;
  }
  input.value = '';
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
  getMode();
  btn.disabled = false;

  fetchFrame(0);
}

let lastFrameTime = 0;
let fpsCount = 0;
let fpsTimer = performance.now();
let currentFps = 0;

function fetchFrame(t) {
  if (pendingFrame) return;
  pendingFrame = true;
  const mode = getMode();
  let frameUrl = '/frame/' + mode + '?t=' + t.toFixed(3) + '&_=' + Date.now();
  if (mode === 'controlnet') {
    frameUrl += '&prompt=' + encodeURIComponent(
      document.getElementById('prompt-input').value.trim());
  }

  const rttStart = performance.now();

  fetch(frameUrl).then(resp => {
    const procMs = resp.headers.get('X-Process-Ms');
    return resp.blob().then(blob => ({ blob, procMs }));
  }).then(({ blob, procMs }) => {
    const rttMs = Math.round(performance.now() - rttStart);
    const url = URL.createObjectURL(blob);
    img.src = url;

    // Update stats
    if (procMs) {
      const pm = parseInt(procMs);
      const procEl = document.getElementById('stat-proc');
      procEl.textContent = pm + 'ms';
      procEl.className = 'stat-value ' + (pm < 33 ? 'stat-good' : pm < 100 ? 'stat-warn' : 'stat-bad');
    }

    const rttEl = document.getElementById('stat-rtt');
    rttEl.textContent = rttMs + 'ms';
    rttEl.className = 'stat-value ' + (rttMs < 50 ? 'stat-good' : rttMs < 150 ? 'stat-warn' : 'stat-bad');

    // FPS calculation
    fpsCount++;
    const now = performance.now();
    if (now - fpsTimer >= 1000) {
      currentFps = fpsCount;
      fpsCount = 0;
      fpsTimer = now;
    }
    const fpsEl = document.getElementById('stat-fps');
    fpsEl.textContent = currentFps || '...';
    fpsEl.className = 'stat-value ' + (currentFps >= 24 ? 'stat-good' : currentFps >= 10 ? 'stat-warn' : 'stat-bad');

    // Real-time status
    const statusEl2 = document.getElementById('stat-status');
    const videoFps = fps || 30;
    if (currentFps >= videoFps * 0.9) {
      statusEl2.textContent = 'Real-time';
      statusEl2.className = 'stat-value stat-good';
    } else if (currentFps >= videoFps * 0.5) {
      statusEl2.textContent = 'Slight lag';
      statusEl2.className = 'stat-value stat-warn';
    } else if (currentFps > 0) {
      statusEl2.textContent = 'Lagging (' + Math.round(currentFps/videoFps*100) + '% speed)';
      statusEl2.className = 'stat-value stat-bad';
    } else {
      statusEl2.textContent = '—';
      statusEl2.className = 'stat-value';
    }

    pendingFrame = false;
  }).catch(() => {
    pendingFrame = false;
  });
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


@app.route("/modes")
def modes():
    return jsonify(list(MODES.keys()))


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


@app.route("/upload_video", methods=["POST"])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"})
    f = request.files['video']
    out_path = os.path.join(DOWNLOAD_DIR, "video.mp4")
    f.save(out_path)
    current_video["path"] = out_path
    cap = cv2.VideoCapture(out_path)
    current_video["fps"] = cap.get(cv2.CAP_PROP_FPS) or 30
    current_video["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_video["duration"] = current_video["total_frames"] / current_video["fps"]
    cap.release()
    return jsonify({
        "ok": True,
        "duration": current_video["duration"],
        "fps": current_video["fps"],
        "total_frames": current_video["total_frames"],
    })


@app.route("/upload_reference", methods=["POST"])
def upload_reference():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})
    f = request.files['image']
    img_bytes = np.frombuffer(f.read(), np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Could not decode image"})

    ok, msg = process_reference_image(image)
    if not ok:
        return jsonify({"error": msg})
    return jsonify({"ok": True, "filename": f.filename, "parts": msg})


@app.route("/reference_preview")
def reference_preview():
    if _reference_data["image"] is None:
        return "No reference", 404
    # Draw the extracted parts overlay on the reference
    vis = _reference_data["image"].copy()
    kp = _reference_data["keypoints"]
    sc = _reference_data["scores"]
    body_scale = _get_body_scale(kp, sc)

    # Draw skeleton
    for a_name, b_name in [("r_sho","r_elb"),("r_elb","r_wri"),("l_sho","l_elb"),("l_elb","l_wri"),
                            ("r_hip","r_kne"),("r_kne","r_ank"),("l_hip","l_kne"),("l_kne","l_ank"),
                            ("r_sho","l_sho"),("r_hip","l_hip"),("r_sho","r_hip"),("l_sho","l_hip")]:
        a = _pt(kp, sc, a_name)
        b = _pt(kp, sc, b_name)
        if a and b:
            cv2.line(vis, a, b, (0, 255, 0), 2)

    for name in _OP:
        pt = _pt(kp, sc, name)
        if pt:
            cv2.circle(vis, pt, 4, (0, 0, 255), -1)

    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype="image/jpeg")


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
    elif mode == "rtmpose_shadow":
        return Body(mode='lightweight', to_openpose=True, backend='onnxruntime')
    elif mode == "rtmpose_costume":
        return Body(mode='lightweight', to_openpose=True, backend='onnxruntime')
    elif mode == "yolo_shadow":
        return YOLO("yolov8n-seg.pt")


# ── Body shadow drawing ─────────────────────────────────────────────

# OpenPose COCO-18 keypoint indices
_OP = {
    "nose": 0, "neck": 1,
    "r_sho": 2, "r_elb": 3, "r_wri": 4,
    "l_sho": 5, "l_elb": 6, "l_wri": 7,
    "r_hip": 8, "r_kne": 9, "r_ank": 10,
    "l_hip": 11, "l_kne": 12, "l_ank": 13,
    "r_eye": 14, "l_eye": 15, "r_ear": 16, "l_ear": 17,
}

def _pt(keypoints, scores, name, threshold=0.3):
    """Get (x, y) for a keypoint, or None if below threshold."""
    idx = _OP[name]
    if idx < len(scores) and scores[idx] > threshold:
        return (int(keypoints[idx][0]), int(keypoints[idx][1]))
    return None

def _midpoint(a, b):
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)

def _limb_polygon(p1, p2, width):
    """Create a thick polygon (rectangle) between two points."""
    if p1 is None or p2 is None:
        return None
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = max(1, (dx*dx + dy*dy) ** 0.5)
    nx = -dy / length * width / 2
    ny = dx / length * width / 2
    return np.array([
        [int(p1[0] + nx), int(p1[1] + ny)],
        [int(p1[0] - nx), int(p1[1] - ny)],
        [int(p2[0] - nx), int(p2[1] - ny)],
        [int(p2[0] + nx), int(p2[1] + ny)],
    ], dtype=np.int32)

def draw_body_shadow(frame, keypoints, scores, color=(40, 30, 30)):
    """Draw a filled body silhouette from OpenPose keypoints."""
    kp, sc = keypoints, scores
    p = lambda name: _pt(kp, sc, name)

    nose = p("nose")
    neck = p("neck")
    r_sho = p("r_sho"); l_sho = p("l_sho")
    r_elb = p("r_elb"); l_elb = p("l_elb")
    r_wri = p("r_wri"); l_wri = p("l_wri")
    r_hip = p("r_hip"); l_hip = p("l_hip")
    r_kne = p("r_kne"); l_kne = p("l_kne")
    r_ank = p("r_ank"); l_ank = p("l_ank")

    # Estimate body scale from shoulder width or torso height
    body_scale = 30  # default
    if r_sho and l_sho:
        body_scale = max(20, int(((r_sho[0]-l_sho[0])**2 + (r_sho[1]-l_sho[1])**2)**0.5 * 0.4))

    # Head: circle at nose or midpoint of eyes
    head_center = nose or _midpoint(p("r_eye"), p("l_eye"))
    if head_center:
        head_r = max(12, int(body_scale * 0.7))
        cv2.circle(frame, head_center, head_r, color, -1)

    # Neck to head connection
    if neck and head_center:
        poly = _limb_polygon(neck, head_center, int(body_scale * 0.5))
        if poly is not None:
            cv2.fillPoly(frame, [poly], color)

    # Torso: polygon from shoulders to hips
    torso_pts = [pt for pt in [r_sho, l_sho, l_hip, r_hip] if pt is not None]
    if len(torso_pts) >= 3:
        cv2.fillPoly(frame, [np.array(torso_pts, dtype=np.int32)], color)

    # Upper body fill: neck area
    if neck and r_sho and l_sho:
        cv2.fillPoly(frame, [np.array([neck, r_sho, l_sho], dtype=np.int32)], color)

    # Limbs as thick polygons
    limb_w = max(10, int(body_scale * 0.45))
    arm_w = max(8, int(body_scale * 0.35))
    calf_w = max(8, int(body_scale * 0.35))

    limbs = [
        (r_sho, r_elb, arm_w), (r_elb, r_wri, arm_w),
        (l_sho, l_elb, arm_w), (l_elb, l_wri, arm_w),
        (r_hip, r_kne, limb_w), (r_kne, r_ank, calf_w),
        (l_hip, l_kne, limb_w), (l_kne, l_ank, calf_w),
    ]
    for a, b, w in limbs:
        poly = _limb_polygon(a, b, w)
        if poly is not None:
            cv2.fillPoly(frame, [poly], color)

    # Joint circles for smoother connections
    joint_r = max(5, int(body_scale * 0.22))
    for pt in [neck, r_sho, l_sho, r_elb, l_elb, r_hip, l_hip, r_kne, l_kne]:
        if pt:
            cv2.circle(frame, pt, joint_r, color, -1)

    # Hands and feet (smaller circles)
    extremity_r = max(4, int(body_scale * 0.18))
    for pt in [r_wri, l_wri, r_ank, l_ank]:
        if pt:
            cv2.circle(frame, pt, extremity_r, color, -1)


# ── Costume transfer ─────────────────────────────────────────────────

# Body part definitions: each part is defined by the keypoints that form its region
# Using OpenPose COCO-18 indices
BODY_PARTS = {
    "head":       {"keypoints": ["nose", "r_eye", "l_eye", "r_ear", "l_ear"], "expand": 1.6},
    "torso":      {"keypoints": ["r_sho", "l_sho", "l_hip", "r_hip"]},
    "upper_torso":{"keypoints": ["neck", "r_sho", "l_sho"]},
    "r_upper_arm":{"keypoints": ["r_sho", "r_elb"], "width_scale": 0.4},
    "r_lower_arm":{"keypoints": ["r_elb", "r_wri"], "width_scale": 0.3},
    "l_upper_arm":{"keypoints": ["l_sho", "l_elb"], "width_scale": 0.4},
    "l_lower_arm":{"keypoints": ["l_elb", "l_wri"], "width_scale": 0.3},
    "r_upper_leg":{"keypoints": ["r_hip", "r_kne"], "width_scale": 0.45},
    "r_lower_leg":{"keypoints": ["r_kne", "r_ank"], "width_scale": 0.35},
    "l_upper_leg":{"keypoints": ["l_hip", "l_kne"], "width_scale": 0.45},
    "l_lower_leg":{"keypoints": ["l_kne", "l_ank"], "width_scale": 0.35},
}

# Reference image data (populated on upload)
_reference_data = {
    "image": None,         # original image (BGR)
    "mask": None,          # person segmentation mask
    "keypoints": None,     # (N, 2) array
    "scores": None,        # (N,) array
    "points": None,        # control points for triangulation
    "indices": None,       # which keypoint index each point maps to
    "triangles": None,     # Delaunay triangle index triples
}


def _get_body_scale(kp, sc):
    r_sho = _pt(kp, sc, "r_sho")
    l_sho = _pt(kp, sc, "l_sho")
    if r_sho and l_sho:
        return max(20, int(((r_sho[0]-l_sho[0])**2 + (r_sho[1]-l_sho[1])**2)**0.5))
    return 60


def _segment_person(image):
    """Use MediaPipe to segment the person from the background."""
    seg_options = ImageSegmenterOptions(
        base_options=BaseOptions(
            model_asset_path=os.path.join(BASE_DIR, "selfie_multiclass.tflite")
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_category_mask=True,
        output_confidence_masks=False,
    )
    segmenter = ImageSegmenter.create_from_options(seg_options)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = segmenter.segment(mp_image)
    segmenter.close()

    mask = np.squeeze(result.category_mask.numpy_view())
    return (mask > 0).astype(np.uint8) * 255


def _collect_control_points(kp, sc, h, w):
    """Collect valid keypoints + boundary points for triangulation."""
    points = []
    indices = []  # which _OP index each point came from (-1 for boundary)

    for name, idx in _OP.items():
        pt = _pt(kp, sc, name)
        if pt is not None:
            points.append([float(pt[0]), float(pt[1])])
            indices.append(idx)

    # Add corner and edge points so triangulation covers the whole image
    boundary = [
        [0, 0], [w//2, 0], [w-1, 0],
        [0, h//2], [w-1, h//2],
        [0, h-1], [w//2, h-1], [w-1, h-1],
    ]
    for bp in boundary:
        points.append(bp)
        indices.append(-1)

    return np.float32(points), indices


def _triangulate(points, w, h):
    """Compute Delaunay triangulation, return list of index triples."""
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for pt in points:
        x, y = float(pt[0]), float(pt[1])
        # Clamp to rect
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        subdiv.insert((x, y))

    tri_list = subdiv.getTriangleList()
    triangles = []

    for t in tri_list:
        pts_tri = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        for pt in pts_tri:
            # Find matching point index
            dists = np.sum((points - np.array(pt)) ** 2, axis=1)
            idx.append(int(np.argmin(dists)))
        # Skip degenerate triangles
        if len(set(idx)) == 3:
            triangles.append(tuple(idx))

    return triangles


def _warp_triangle(src_img, src_mask, dst_img, src_tri, dst_tri):
    """Warp a single triangle from src to dst using affine transform."""
    src_tri = np.float32(src_tri)
    dst_tri = np.float32(dst_tri)

    # Bounding rects
    sr = cv2.boundingRect(src_tri)
    dr = cv2.boundingRect(dst_tri)

    # Clip to image bounds
    sh, sw = src_img.shape[:2]
    dh, dw = dst_img.shape[:2]

    sr = (max(0, sr[0]), max(0, sr[1]),
          min(sw - max(0, sr[0]), sr[2]), min(sh - max(0, sr[1]), sr[3]))
    dr = (max(0, dr[0]), max(0, dr[1]),
          min(dw - max(0, dr[0]), dr[2]), min(dh - max(0, dr[1]), dr[3]))

    if sr[2] <= 0 or sr[3] <= 0 or dr[2] <= 0 or dr[3] <= 0:
        return

    # Offset triangles to their bounding rects
    src_tri_rect = [(p[0] - sr[0], p[1] - sr[1]) for p in src_tri]
    dst_tri_rect = [(p[0] - dr[0], p[1] - dr[1]) for p in dst_tri]

    # Crop source
    src_crop = src_img[sr[1]:sr[1]+sr[3], sr[0]:sr[0]+sr[2]]
    mask_crop = src_mask[sr[1]:sr[1]+sr[3], sr[0]:sr[0]+sr[2]]

    if src_crop.size == 0:
        return

    # Affine transform
    M = cv2.getAffineTransform(np.float32(src_tri_rect), np.float32(dst_tri_rect))
    warped = cv2.warpAffine(src_crop, M, (dr[2], dr[3]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warped_mask = cv2.warpAffine(mask_crop, M, (dr[2], dr[3]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Create triangle mask in destination rect
    tri_mask = np.zeros((dr[3], dr[2]), dtype=np.uint8)
    cv2.fillConvexPoly(tri_mask, np.int32(dst_tri_rect), 255)

    # Combine: only pixels inside triangle AND person mask
    combined_mask = cv2.bitwise_and(tri_mask, warped_mask)
    alpha = combined_mask.astype(np.float32)[:, :, None] / 255.0

    # Blend into destination
    roi = dst_img[dr[1]:dr[1]+dr[3], dr[0]:dr[0]+dr[2]]
    if roi.shape[:2] == warped.shape[:2]:
        dst_img[dr[1]:dr[1]+dr[3], dr[0]:dr[0]+dr[2]] = \
            (roi * (1 - alpha) + warped * alpha).astype(np.uint8)


def process_reference_image(image):
    """Process uploaded reference: segment person, detect pose, build triangulation."""
    body_detector = Body(mode='lightweight', to_openpose=True, backend='onnxruntime')
    keypoints, scores = body_detector(image)

    if keypoints is None or len(keypoints) == 0:
        return False, "No person detected in reference image"

    kp, sc = keypoints[0], scores[0]
    h, w = image.shape[:2]

    # Segment person from background
    person_mask = _segment_person(image)

    # Collect control points and triangulate
    points, indices = _collect_control_points(kp, sc, h, w)
    triangles = _triangulate(points, w, h)

    _reference_data["image"] = image
    _reference_data["mask"] = person_mask
    _reference_data["keypoints"] = kp
    _reference_data["scores"] = sc
    _reference_data["points"] = points
    _reference_data["indices"] = indices
    _reference_data["triangles"] = triangles

    n_body = sum(1 for i in indices if i >= 0)
    return True, f"{n_body} keypoints, {len(triangles)} triangles"


def render_costume(frame, keypoints, scores):
    """Render reference image warped to match detected pose via triangulation."""
    if "triangles" not in _reference_data or not _reference_data["triangles"]:
        return frame.copy()

    h, w = frame.shape[:2]
    canvas = np.full((h, w, 3), (220, 220, 230), dtype=np.uint8)

    src_img = _reference_data["image"]
    src_mask = _reference_data["mask"]
    src_points = _reference_data["points"]
    indices = _reference_data["indices"]
    triangles = _reference_data["triangles"]

    kp, sc = keypoints, scores

    # Build destination points: map each source point to its target location
    dst_points = []
    for i, idx in enumerate(indices):
        if idx >= 0:
            # This is a body keypoint — find its position in the target
            name = [n for n, v in _OP.items() if v == idx][0]
            pt = _pt(kp, sc, name)
            if pt is not None:
                dst_points.append([float(pt[0]), float(pt[1])])
            else:
                # Keypoint not visible in target — keep source position scaled
                dst_points.append([float(src_points[i][0]), float(src_points[i][1])])
        else:
            # Boundary point — scale to target dimensions
            src_h, src_w = src_img.shape[:2]
            dst_points.append([
                src_points[i][0] / max(1, src_w - 1) * (w - 1),
                src_points[i][1] / max(1, src_h - 1) * (h - 1),
            ])

    dst_points = np.float32(dst_points)

    # Warp each triangle
    for tri_idx in triangles:
        i0, i1, i2 = tri_idx
        src_tri = [src_points[i0], src_points[i1], src_points[i2]]
        dst_tri = [dst_points[i0], dst_points[i1], dst_points[i2]]

        try:
            _warp_triangle(src_img, src_mask, canvas, src_tri, dst_tri)
        except Exception:
            continue

    return canvas


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

    elif mode == "rtmpose_shadow":
        h, w = frame.shape[:2]
        out = np.full((h, w, 3), (220, 220, 230), dtype=np.uint8)
        keypoints, scores = processor(frame)
        if keypoints is not None and len(keypoints) > 0:
            for i in range(len(keypoints)):
                draw_body_shadow(out, keypoints[i], scores[i], color=(40, 30, 30))
            out = cv2.GaussianBlur(out, (5, 5), 0)
        return out

    elif mode == "rtmpose_costume":
        keypoints, scores = processor(frame)
        if keypoints is not None and len(keypoints) > 0 and _reference_data.get("triangles"):
            out = render_costume(frame, keypoints[0], scores[0])
            return out
        elif keypoints is not None and len(keypoints) > 0:
            # No reference uploaded, fall back to shadow
            h, w = frame.shape[:2]
            out = np.full((h, w, 3), (220, 220, 230), dtype=np.uint8)
            for i in range(len(keypoints)):
                draw_body_shadow(out, keypoints[i], scores[i], color=(40, 30, 30))
            return out
        return frame.copy()

    elif mode == "yolo_shadow":
        h, w = frame.shape[:2]
        results = processor(frame, classes=[0], verbose=False)
        r = results[0]

        out = np.full((h, w, 3), (220, 220, 230), dtype=np.uint8)

        if r.masks is not None:
            shadow_colors = [
                (40, 30, 30), (30, 40, 60), (50, 25, 25),
                (25, 35, 50), (45, 30, 40), (30, 30, 50),
            ]
            for i, mask_tensor in enumerate(r.masks.data):
                mask = mask_tensor.cpu().numpy()
                # Resize mask to frame dimensions
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                color = shadow_colors[i % len(shadow_colors)]
                person_mask = (mask > 0.5)
                out[person_mask] = color

            out = cv2.GaussianBlur(out, (5, 5), 0)

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
        t_start = time.time()

        if mode == "controlnet" and _HAS_CUDA:
            prompt = request.args.get("prompt", "person dancing, professional photo")
            out = controlnet_generate(raw_frame, prompt=prompt)
        else:
            processor = get_processor(mode)
            is_rtm = mode.startswith("rtmpose")
            is_yolo = mode.startswith("yolo")

            if is_rtm or is_yolo:
                out = process_frame(mode, processor, raw_frame, None)
            else:
                rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                out = process_frame(mode, processor, raw_frame, mp_image)

        proc_ms = (time.time() - t_start) * 1000

        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), mimetype="image/jpeg",
                        headers={
                            "Cache-Control": "no-store",
                            "X-Process-Ms": str(int(proc_ms)),
                            "Access-Control-Expose-Headers": "X-Process-Ms",
                        })
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
