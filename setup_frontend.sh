#!/bin/bash
set -e
echo "=== Frontend Setup (lightweight) ==="

# Only needs opencv for video reading + thumbnails, and flask
pip install --quiet flask opencv-python yt-dlp requests

# Deno for yt-dlp
if [ ! -f ~/.deno/bin/deno ]; then
    curl -fsSL https://deno.land/install.sh | sh 2>/dev/null
fi

echo ""
echo "=== Frontend ready! ==="
echo "Start with:"
echo "  BACKEND_URL=http://<gpu-ip>:5005 python3 frontend.py"
