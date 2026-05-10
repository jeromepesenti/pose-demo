"""GPU Backend: receives frames, processes them, returns results.
Runs on a GPU machine (Lambda, GCP, etc.)
"""
import os
import time
import traceback
import threading
from flask import Flask, Response, request, jsonify
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

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check for CUDA
_HAS_CUDA = False
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

if _HAS_CUDA:
    from controlnet_gpu import generate as controlnet_generate
    from controlnet_gpu import generate_video_frame

# Try detectron2
_HAS_D2 = False
try:
    from detectron2_modes import process_keypoint, process_panoptic, process_densepose
    _HAS_D2 = True
except ImportError:
    pass

# ── Processor cache ──────────────────────────────────────────────────

_processor_cache = {}
_processor_lock = threading.Lock()


def get_processor(mode):
    with _processor_lock:
        if mode in _processor_cache:
            return _processor_cache[mode]
        processor = _create_processor(mode)
        _processor_cache[mode] = processor
        return processor


def _create_processor(mode):
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
    elif mode in ("rtmpose_body", "rtmpose_shadow", "rtmpose_costume"):
        return Body(mode='lightweight', to_openpose=True, backend='onnxruntime')
    elif mode == "rtmpose_whole":
        return Wholebody(mode='lightweight', to_openpose=False, backend='onnxruntime')
    elif mode == "yolo_shadow":
        return YOLO("yolov8n-seg.pt")


# ── Import process_frame logic from server.py ────────────────────────
# We import the drawing helpers and process_frame function
# For now, we'll use exec to load them from server.py to avoid duplication

# Lazy import of the processing logic
_process_module = None

def _get_process_module():
    global _process_module
    if _process_module is not None:
        return _process_module
    import importlib.util
    spec = importlib.util.spec_from_file_location("server", os.path.join(BASE_DIR, "server.py"))
    mod = importlib.util.module_from_spec(spec)
    # We only need the processing functions, not the Flask app
    # So we'll just import what we need directly
    _process_module = True
    return True


# ── Available modes ──────────────────────────────────────────────────

@app.route("/")
def index():
    return jsonify({"status": "GPU backend running", "cuda": _HAS_CUDA, "detectron2": _HAS_D2})


@app.route("/modes")
def modes():
    available = [
        "pose", "shadow", "face_mesh", "hands", "face_detect", "objects",
        "rtmpose_body", "rtmpose_whole", "rtmpose_shadow", "rtmpose_costume",
        "yolo_shadow",
    ]
    if _HAS_D2:
        available.extend(["d2_keypoint", "d2_panoptic", "d2_densepose"])
    if _HAS_CUDA:
        available.extend(["controlnet", "controlnet_video"])
    return jsonify(available)


# ── Process a single frame ───────────────────────────────────────────

@app.route("/process", methods=["POST"])
def process():
    """Receive a JPEG frame, process it, return processed JPEG.

    Query params:
        mode: processing mode
        prompt: (optional) for controlnet
    """
    mode = request.args.get("mode", "pose")
    prompt = request.args.get("prompt", "person dancing, professional photo")

    # Decode input frame
    img_bytes = np.frombuffer(request.data, np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return "Bad image", 400

    t0 = time.time()

    try:
        if mode == "controlnet" and _HAS_CUDA:
            out = controlnet_generate(frame, prompt=prompt)
        elif mode.startswith("d2_") and _HAS_D2:
            if mode == "d2_keypoint":
                out = process_keypoint(frame)
            elif mode == "d2_panoptic":
                out = process_panoptic(frame)
            elif mode == "d2_densepose":
                out = process_densepose(frame)
            else:
                out = frame
        else:
            # Use the processor cache for other modes
            processor = get_processor(mode)
            is_rtm = mode.startswith("rtmpose")
            is_yolo = mode.startswith("yolo")

            if is_rtm or is_yolo:
                out = _process_frame_simple(mode, processor, frame)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                out = _process_frame_mp(mode, processor, frame, mp_image)

        proc_ms = int((time.time() - t0) * 1000)
        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), mimetype="image/jpeg",
                        headers={
                            "X-Process-Ms": str(proc_ms),
                            "Access-Control-Expose-Headers": "X-Process-Ms",
                        })
    except Exception as e:
        print(f"[{mode}] Error: {e}")
        traceback.print_exc()
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), mimetype="image/jpeg",
                        headers={"X-Process-Ms": "-1"})


# ── Simplified process_frame (avoids importing the full server.py) ───

# Drawing helpers (duplicated from server.py to keep backend self-contained)

POSE_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
    (15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]
FACE_OVAL = [(10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),(356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),(379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),(149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),(234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10)]
LIPS = [(61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),(321,375),(375,291),(291,409),(409,270),(270,269),(269,267),(267,0),(0,37),(37,39),(39,40),(40,185),(185,61)]
LEFT_EYE = [(33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),(155,133),(133,173),(173,157),(157,158),(158,159),(159,160),(160,161),(161,246),(246,33)]
RIGHT_EYE = [(362,382),(382,381),(381,380),(380,374),(374,373),(373,390),(390,249),(249,263),(263,466),(466,388),(388,387),(387,386),(386,385),(385,384),(384,398),(398,362)]
FACE_CONNECTIONS_ALL = FACE_OVAL + LIPS + LEFT_EYE + RIGHT_EYE

RTMPOSE_BODY_CONNECTIONS = [(0,1),(0,14),(0,15),(1,2),(1,5),(1,8),(1,11),(2,3),(3,4),(5,6),(6,7),(8,9),(9,10),(11,12),(12,13),(14,16),(15,17)]
RTMPOSE_WHOLEBODY_BODY = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

COLORS = [(233,105,0),(15,52,96),(22,199,154),(245,166,35),(189,16,224),(80,227,194),(255,107,107),(78,205,196)]


def _draw_landmarks(frame, landmarks, connections, color, w, h, thickness=2, radius=3):
    pts = []
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        vis = lm.visibility if hasattr(lm, 'visibility') and lm.visibility else 1.0
        pts.append((px, py, vis))
    for a, b in connections:
        if a < len(pts) and b < len(pts) and pts[a][2] > 0.3 and pts[b][2] > 0.3:
            cv2.line(frame, pts[a][:2], pts[b][:2], color, thickness)
    for px, py, vis in pts:
        if vis > 0.3:
            cv2.circle(frame, (px, py), radius, color, -1)
            cv2.circle(frame, (px, py), radius, (255, 255, 255), 1)


def _draw_kp_array(frame, kp, sc, conns, color, thr=0.3, thick=2, rad=4):
    n = len(kp)
    for a, b in conns:
        if a < n and b < n and sc[a] > thr and sc[b] > thr:
            cv2.line(frame, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])), color, thick)
    for i in range(n):
        if sc[i] > thr:
            cv2.circle(frame, (int(kp[i][0]), int(kp[i][1])), rad, color, -1)
            cv2.circle(frame, (int(kp[i][0]), int(kp[i][1])), rad, (255,255,255), 1)


def _draw_bbox(frame, bbox, label, color):
    x, y, bw, bh = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
    cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
    cv2.putText(frame, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _process_frame_mp(mode, processor, frame, mp_image):
    h, w = frame.shape[:2]
    if mode == "pose":
        result = processor.detect(mp_image)
        out = frame.copy()
        if result.pose_landmarks:
            for i, p in enumerate(result.pose_landmarks):
                _draw_landmarks(out, p, POSE_CONNECTIONS, COLORS[i%len(COLORS)], w, h, 3, 5)
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
        fg = np.full_like(frame, (60,30,30), dtype=np.uint8)
        bg = np.full_like(frame, (230,220,220), dtype=np.uint8)
        return cv2.GaussianBlur(np.where(condition[:,:,np.newaxis], fg, bg).astype(np.uint8), (7,7), 0)
    elif mode == "face_mesh":
        result = processor.detect(mp_image)
        out = frame.copy()
        if result.face_landmarks:
            for i, f in enumerate(result.face_landmarks):
                _draw_landmarks(out, f, FACE_CONNECTIONS_ALL, COLORS[i%len(COLORS)], w, h, 1, 1)
        return out
    elif mode == "hands":
        result = processor.detect(mp_image)
        out = frame.copy()
        if result.hand_landmarks:
            for i, hand in enumerate(result.hand_landmarks):
                _draw_landmarks(out, hand, HAND_CONNECTIONS, COLORS[i%len(COLORS)], w, h, 2, 4)
        return out
    elif mode == "face_detect":
        result = processor.detect(mp_image)
        out = frame.copy()
        if result.detections:
            for i, det in enumerate(result.detections):
                color = COLORS[i%len(COLORS)]
                score = det.categories[0].score if det.categories else 0
                _draw_bbox(out, det.bounding_box, f"Face {score:.0%}", color)
                for kp in det.keypoints or []:
                    cv2.circle(out, (int(kp.x*w), int(kp.y*h)), 4, color, -1)
        return out
    elif mode == "objects":
        result = processor.detect(mp_image)
        out = frame.copy()
        if result.detections:
            for i, det in enumerate(result.detections):
                color = COLORS[i%len(COLORS)]
                cat = det.categories[0] if det.categories else None
                label = f"{cat.category_name} {cat.score:.0%}" if cat else "?"
                _draw_bbox(out, det.bounding_box, label, color)
        return out
    return frame.copy()


def _process_frame_simple(mode, processor, frame):
    h, w = frame.shape[:2]
    if mode == "rtmpose_body":
        out = frame.copy()
        kps, scs = processor(frame)
        if kps is not None:
            for i in range(len(kps)):
                _draw_kp_array(out, kps[i], scs[i], RTMPOSE_BODY_CONNECTIONS, COLORS[i%len(COLORS)], thick=3, rad=5)
        return out
    elif mode == "rtmpose_whole":
        out = frame.copy()
        kps, scs = processor(frame)
        if kps is not None:
            for i in range(len(kps)):
                _draw_kp_array(out, kps[i], scs[i], RTMPOSE_WHOLEBODY_BODY, COLORS[i%len(COLORS)], thick=3, rad=4)
        return out
    elif mode == "rtmpose_shadow":
        out = np.full((h, w, 3), (220,220,230), dtype=np.uint8)
        kps, scs = processor(frame)
        if kps is not None:
            # Simple shadow: thick skeleton
            for i in range(len(kps)):
                _draw_kp_array(out, kps[i], scs[i], RTMPOSE_BODY_CONNECTIONS, (40,30,30), thick=15, rad=12)
            out = cv2.GaussianBlur(out, (11,11), 0)
        return out
    elif mode == "yolo_shadow":
        out = np.full((h, w, 3), (220,220,230), dtype=np.uint8)
        results = processor(frame, classes=[0], verbose=False)
        r = results[0]
        if r.masks is not None:
            shadow_colors = [(40,30,30),(30,40,60),(50,25,25),(25,35,50),(45,30,40),(30,30,50)]
            for i, mt in enumerate(r.masks.data):
                mask = mt.cpu().numpy()
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                out[mask > 0.5] = shadow_colors[i % len(shadow_colors)]
            out = cv2.GaussianBlur(out, (5,5), 0)
        return out
    return frame.copy()


if __name__ == "__main__":
    print("=== GPU Backend ===")
    print(f"CUDA: {_HAS_CUDA}")
    print(f"Detectron2: {_HAS_D2}")
    app.run(host="0.0.0.0", port=5005, threaded=True)
