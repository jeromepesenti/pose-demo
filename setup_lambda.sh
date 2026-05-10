#!/bin/bash
set -e
echo "=== Lambda GPU Backend Setup (~2 min) ==="
START=$(date +%s)

# System deps
echo "[1/5] System packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 libegl1 libgles2 unzip 2>/dev/null

# Python deps (all pip wheels, no compilation)
echo "[2/5] Python packages..."
pip install --quiet flask mediapipe opencv-python yt-dlp rtmlib \
    onnxruntime-gpu ultralytics diffusers transformers accelerate \
    peft safetensors 'Pillow>=9.1' 'numpy<2' requests

# Deno for yt-dlp
echo "[3/5] Deno..."
if [ ! -f ~/.deno/bin/deno ]; then
    curl -fsSL https://deno.land/install.sh | sh 2>/dev/null
fi

# Project setup
echo "[4/5] Project files..."
cd ~/pose-demo
mkdir -p downloads saved_videos

# Download MediaPipe models
for model in pose_landmarker_lite.task face_landmarker.task hand_landmarker.task; do
    [ -f "$model" ] && continue
    echo "  Downloading $model..."
    case $model in
        pose_landmarker_lite.task) curl -sL -o $model "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" ;;
        face_landmarker.task) curl -sL -o $model "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" ;;
        hand_landmarker.task) curl -sL -o $model "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" ;;
    esac
done
for model in selfie_multiclass.tflite face_detector.tflite efficientdet.tflite; do
    [ -f "$model" ] && continue
    echo "  Downloading $model..."
    case $model in
        selfie_multiclass.tflite) curl -sL -o $model "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite" ;;
        face_detector.tflite) curl -sL -o $model "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite" ;;
        efficientdet.tflite) curl -sL -o $model "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite" ;;
    esac
done

# Pre-download ML models
echo "[5/5] ML models..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')" 2>/dev/null
python3 -c "from rtmlib import Body; Body(mode='lightweight', to_openpose=True, backend='onnxruntime')" 2>/dev/null

# Detectron2 + DensePose (pre-built wheels, ~15s)
if [ "${INSTALL_DETECTRON2:-1}" = "1" ]; then
    echo "[6/6] Installing Detectron2 + DensePose..."
    WHEEL_BASE="https://github.com/jeromepesenti/pose-demo/releases/download/v0.1-wheels"
    PY_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    curl -sL -o /tmp/detectron2.whl "${WHEEL_BASE}/detectron2-0.6-${PY_VER}-${PY_VER}-linux_x86_64.whl"
    curl -sL -o /tmp/densepose.whl "${WHEEL_BASE}/detectron2_densepose-0.6-py3-none-any.whl"
    pip install --quiet --no-deps /tmp/detectron2.whl /tmp/densepose.whl
    pip install --quiet fvcore iopath pycocotools tabulate yacs omegaconf cloudpickle av opencv-python-headless pybind11
    # DensePose needs the config files from the repo
    if [ ! -d /tmp/detectron2 ]; then
        git clone --quiet --depth 1 https://github.com/facebookresearch/detectron2.git /tmp/detectron2
    fi
fi

ELAPSED=$(($(date +%s) - START))
echo ""
echo "=== Done in ${ELAPSED}s ==="
echo "Start backend:  cd ~/pose-demo && python3 backend_gpu.py"
