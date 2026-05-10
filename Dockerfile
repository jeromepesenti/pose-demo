FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:/root/.deno/bin:$PATH"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git curl unzip \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip3 install --no-cache-dir \
    flask mediapipe opencv-python yt-dlp rtmlib \
    onnxruntime-gpu ultralytics \
    diffusers transformers accelerate peft safetensors \
    torch torchvision \
    'Pillow>=9.1' 'numpy<2' pybind11

# Detectron2 (with CUDA compilation)
RUN git clone https://github.com/facebookresearch/detectron2.git /opt/detectron2 \
    && cd /opt/detectron2 \
    && CPATH=$(python3 -m pybind11 --includes | sed 's/-I//g' | tr ' ' ':') \
       pip3 install --no-build-isolation -e . \
    && pip3 install --no-build-isolation -e projects/DensePose

# Deno (for yt-dlp)
RUN curl -fsSL https://deno.land/install.sh | sh

# Pre-download commonly used models
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"
RUN python3 -c "from rtmlib import Body, Wholebody; Body(mode='lightweight', to_openpose=True, backend='onnxruntime'); Wholebody(mode='lightweight', to_openpose=False, backend='onnxruntime')"

WORKDIR /app
COPY server.py controlnet_gpu.py detectron2_modes.py index.html restart.sh ./
COPY *.task *.tflite ./
RUN mkdir -p downloads saved_videos

EXPOSE 5001
CMD ["python3", "server.py"]
