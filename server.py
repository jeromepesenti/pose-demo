import os
import subprocess
import time
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

current_video = {"path": None}
YT_DLP = os.path.join(BASE_DIR, "venv", "bin", "yt-dlp")

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

# RTMPose COCO-18 skeleton (OpenPose format from rtmlib)
RTMPOSE_BODY_CONNECTIONS = [
    (0,1),(0,14),(0,15),(1,2),(1,5),(1,8),(1,11),
    (2,3),(3,4),(5,6),(6,7),(8,9),(9,10),(11,12),(12,13),
    (14,16),(15,17),
]

# RTMPose Wholebody 133-point skeleton connections
# 0-16: body, 17-22: feet, 23-90: face, 91-111: left hand, 112-132: right hand
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
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 20px;
  }
  h1 { margin-bottom: 10px; font-size: 1.4rem; }
  #info { font-size: 0.85rem; color: #aaa; margin-bottom: 16px; }
  .row {
    display: flex; gap: 8px; margin-bottom: 12px;
    width: 100%; max-width: 750px; align-items: center;
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
  #output { text-align: center; }
  #output img {
    border-radius: 8px; max-width: 800px; width: 100%; background: #000;
  }
</style>
</head>
<body>

<h1>YouTube Vision Demo</h1>
<div id="info">Paste a YouTube URL, pick a processing mode, and hit Go</div>

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
  <img id="stream-img">
</div>

<script>
let currentUrl = '';

async function go() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) return;
  const btn = document.getElementById('go-btn');
  const status = document.getElementById('status');
  const output = document.getElementById('output');
  btn.disabled = true;
  output.style.display = 'none';

  if (url !== currentUrl) {
    status.textContent = 'Downloading video...';
    try {
      const resp = await fetch('/download', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({url})
      });
      const data = await resp.json();
      if (data.error) {
        status.textContent = 'Error: ' + data.error;
        btn.disabled = false;
        return;
      }
      currentUrl = url;
    } catch (e) {
      status.textContent = 'Error: ' + e.message;
      btn.disabled = false;
      return;
    }
  }

  const mode = document.getElementById('mode-select').value;
  status.textContent = 'Loading model & processing...';
  output.style.display = 'block';
  document.getElementById('stream-img').src = '/stream/' + mode + '?' + Date.now();
  status.textContent = 'Streaming — ' + document.getElementById('mode-select').selectedOptions[0].text;
  btn.disabled = false;
}

document.getElementById('mode-select').addEventListener('change', () => {
  if (currentUrl) go();
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
    return jsonify({"ok": True})


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
    """Draw keypoints from numpy arrays (N,2) and scores (N,)."""
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


# ── Processor creation ───────────────────────────────────────────────

def create_processor(mode):
    video_mode = mp.tasks.vision.RunningMode.VIDEO

    if mode == "pose":
        base = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "pose_landmarker_lite.task"))
        return PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=base, running_mode=video_mode, num_poses=6,
        ))
    elif mode == "shadow":
        base = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "selfie_multiclass.tflite"))
        return ImageSegmenter.create_from_options(ImageSegmenterOptions(
            base_options=base, running_mode=video_mode,
            output_category_mask=True, output_confidence_masks=True,
        ))
    elif mode == "face_mesh":
        base = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "face_landmarker.task"))
        return FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=base, running_mode=video_mode,
            num_faces=4, output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        ))
    elif mode == "hands":
        base = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "hand_landmarker.task"))
        return HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=base, running_mode=video_mode, num_hands=6,
        ))
    elif mode == "face_detect":
        base = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "face_detector.tflite"))
        return FaceDetector.create_from_options(FaceDetectorOptions(
            base_options=base, running_mode=video_mode,
        ))
    elif mode == "objects":
        base = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "efficientdet.tflite"))
        return ObjectDetector.create_from_options(ObjectDetectorOptions(
            base_options=base, running_mode=video_mode,
            max_results=10, score_threshold=0.3,
        ))
    elif mode == "rtmpose_body":
        return Body(mode='lightweight', to_openpose=True, backend='onnxruntime')
    elif mode == "rtmpose_whole":
        return Wholebody(mode='lightweight', to_openpose=False, backend='onnxruntime')


# ── Frame processing ─────────────────────────────────────────────────

def process_frame(mode, processor, frame, mp_image, timestamp_ms):
    h, w = frame.shape[:2]

    if mode == "pose":
        result = processor.detect_for_video(mp_image, timestamp_ms)
        out = frame.copy()
        if result.pose_landmarks:
            for i, person in enumerate(result.pose_landmarks):
                draw_landmarks(out, person, POSE_CONNECTIONS,
                               COLORS[i % len(COLORS)], w, h, thickness=3, radius=5)
        return out

    elif mode == "shadow":
        result = processor.segment_for_video(mp_image, timestamp_ms)
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
        result = processor.detect_for_video(mp_image, timestamp_ms)
        out = frame.copy()
        if result.face_landmarks:
            for i, face in enumerate(result.face_landmarks):
                draw_landmarks(out, face, FACE_CONNECTIONS,
                               COLORS[i % len(COLORS)], w, h, thickness=1, radius=1)
        return out

    elif mode == "hands":
        result = processor.detect_for_video(mp_image, timestamp_ms)
        out = frame.copy()
        if result.hand_landmarks:
            for i, hand in enumerate(result.hand_landmarks):
                draw_landmarks(out, hand, HAND_CONNECTIONS,
                               COLORS[i % len(COLORS)], w, h, thickness=2, radius=4)
        return out

    elif mode == "face_detect":
        result = processor.detect_for_video(mp_image, timestamp_ms)
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
        result = processor.detect_for_video(mp_image, timestamp_ms)
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
                # Body + feet (thick)
                draw_keypoints_array(out, kp, sc,
                                     RTMPOSE_WHOLEBODY_BODY + RTMPOSE_WHOLEBODY_FEET,
                                     color, threshold=0.3, thickness=3, radius=4)
                # Face (thin)
                draw_keypoints_array(out, kp, sc, RTMPOSE_WHOLEBODY_FACE,
                                     (200, 200, 200), threshold=0.3, thickness=1, radius=1)
                # Hands (medium)
                draw_keypoints_array(out, kp, sc,
                                     RTMPOSE_WHOLEBODY_LHAND + RTMPOSE_WHOLEBODY_RHAND,
                                     (80, 227, 194), threshold=0.3, thickness=2, radius=3)
        return out

    return frame.copy()


# ── Stream generator ─────────────────────────────────────────────────

def generate_frames(mode):
    path = current_video.get("path")
    if not path or not os.path.exists(path):
        return

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = 1.0 / fps
    timestamp_ms = 0
    frame_duration_ms = int(1000 / fps)

    is_rtm = mode.startswith("rtmpose")
    processor = create_processor(mode)

    try:
        while cap.isOpened():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            try:
                if is_rtm:
                    out = process_frame(mode, processor, frame, None, None)
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    out = process_frame(mode, processor, frame, mp_image, timestamp_ms)

                timestamp_ms += frame_duration_ms

                _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            except Exception as e:
                print(f"[{mode}] Frame error: {e}")
                traceback.print_exc()
                timestamp_ms += frame_duration_ms
                continue

            elapsed = time.time() - t0
            if elapsed < delay:
                time.sleep(delay - elapsed)
    finally:
        cap.release()
        if hasattr(processor, 'close'):
            processor.close()


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
