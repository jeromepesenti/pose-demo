FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir flask opencv-python-headless requests

WORKDIR /app
COPY frontend.py frontend.html ./
RUN mkdir -p downloads saved_videos

ENV PORT=8080
EXPOSE 8080
CMD ["python3", "frontend.py"]
