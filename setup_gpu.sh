#!/bin/bash
set -e
echo "=== Pose Demo GPU Setup ==="

# System deps
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 python3-dev unzip

# Python deps
echo "[2/6] Installing Python packages..."
pip install --quiet flask mediapipe opencv-python yt-dlp rtmlib \
    onnxruntime-gpu ultralytics diffusers transformers accelerate \
    peft safetensors 'Pillow>=9.1' 'numpy<2' pybind11

# Detectron2
echo "[3/6] Installing Detectron2 + DensePose (compiling CUDA extensions)..."
if [ ! -d /tmp/detectron2 ]; then
    git clone --quiet https://github.com/facebookresearch/detectron2.git /tmp/detectron2
fi
cd /tmp/detectron2
export CPATH=$(python3 -m pybind11 --includes | sed 's/-I//g' | tr ' ' ':')
pip install --quiet --no-build-isolation -e .
pip install --quiet --no-build-isolation -e projects/DensePose

# Deno (for yt-dlp YouTube support)
echo "[4/6] Installing Deno..."
if [ ! -f ~/.deno/bin/deno ]; then
    curl -fsSL https://deno.land/install.sh | sh
fi

# Project files
echo "[5/6] Setting up project..."
cd ~/pose-demo
mkdir -p downloads saved_videos

# Download MediaPipe models if missing
for model in pose_landmarker_lite.task face_landmarker.task hand_landmarker.task; do
    if [ ! -f "$model" ]; then
        echo "  Downloading $model..."
        case $model in
            pose_landmarker_lite.task)
                curl -sL -o $model "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" ;;
            face_landmarker.task)
                curl -sL -o $model "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" ;;
            hand_landmarker.task)
                curl -sL -o $model "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" ;;
        esac
    fi
done
for model in selfie_multiclass.tflite face_detector.tflite efficientdet.tflite; do
    if [ ! -f "$model" ]; then
        echo "  Downloading $model..."
        case $model in
            selfie_multiclass.tflite)
                curl -sL -o $model "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite" ;;
            face_detector.tflite)
                curl -sL -o $model "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite" ;;
            efficientdet.tflite)
                curl -sL -o $model "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite" ;;
        esac
    fi
done

# Pre-download ML models
echo "[6/6] Pre-downloading ML models..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')" 2>/dev/null
python3 -c "from rtmlib import Body; Body(mode='lightweight', to_openpose=True, backend='onnxruntime')" 2>/dev/null

echo ""
echo "=== Setup complete! ==="
echo "Start the server with:"
echo "  export PATH=\$HOME/.deno/bin:\$HOME/.local/bin:\$PATH"
echo "  cd ~/pose-demo && python3 server.py"
echo ""
echo "Then SSH tunnel from your Mac:"
echo "  ssh -L 5002:localhost:5001 ubuntu@<IP>"
echo "  Open http://localhost:5002"
