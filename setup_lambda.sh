#!/bin/bash
set -e
echo "=== Lambda GPU Backend Setup (~2 min) ==="
START=$(date +%s)

# System deps
echo "[1/5] System packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 unzip 2>/dev/null

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

# Detectron2 (optional, adds ~1-2 min)
if [ "${INSTALL_DETECTRON2:-1}" = "1" ]; then
    echo "[6/6] Installing Detectron2 + DensePose via conda..."
    if ! command -v conda &>/dev/null; then
        echo "  Installing Miniconda..."
        curl -sL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash /tmp/miniconda.sh -b -p $HOME/miniconda 2>/dev/null
        export PATH=$HOME/miniconda/bin:$PATH
    fi
    conda install -y -c conda-forge detectron2 2>/dev/null || {
        echo "  Conda install failed, trying pip with pre-built..."
        pip install --quiet --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git' 2>/dev/null
    }
    # DensePose
    if [ -d /tmp/detectron2 ]; then
        pip install --quiet --no-build-isolation -e /tmp/detectron2/projects/DensePose 2>/dev/null
    else
        git clone --quiet https://github.com/facebookresearch/detectron2.git /tmp/detectron2 2>/dev/null
        pip install --quiet --no-build-isolation -e /tmp/detectron2/projects/DensePose 2>/dev/null
    fi
fi

ELAPSED=$(($(date +%s) - START))
echo ""
echo "=== Done in ${ELAPSED}s ==="
echo "Start backend:  cd ~/pose-demo && python3 backend_gpu.py"
