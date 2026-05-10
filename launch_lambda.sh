#!/bin/bash
# Launch a Lambda Labs GPU backend and connect it to the GCP frontend.
# Usage: ./launch_lambda.sh <lambda-ip>
#
# Prerequisites:
#   - Lambda instance running with SSH access
#   - GCP frontend running at 34.10.252.140:5001

set -e

IP=${1:?"Usage: ./launch_lambda.sh <lambda-ip>"}
FRONTEND="34.10.252.140"

echo "=== Setting up Lambda backend at $IP ==="

# 1. Copy project files
echo "[1/4] Copying files..."
scp -o StrictHostKeyChecking=no \
    backend_gpu.py controlnet_gpu.py detectron2_modes.py \
    setup_lambda.sh \
    ubuntu@${IP}:~/pose-demo/ 2>/dev/null

# 2. Run setup
echo "[2/4] Installing dependencies (~2 min)..."
ssh ubuntu@${IP} "bash ~/pose-demo/setup_lambda.sh"

# 3. Start backend
echo "[3/4] Starting backend..."
ssh ubuntu@${IP} "cd ~/pose-demo && pkill -f backend_gpu.py 2>/dev/null; export PATH=\$HOME/.deno/bin:\$HOME/.local/bin:\$PATH && nohup python3 backend_gpu.py > backend.log 2>&1 &"
sleep 5

# 4. Point frontend to this backend
echo "[4/4] Connecting frontend to backend..."
curl -s -X POST "http://${FRONTEND}:5001/set_backend" \
    -H "Content-Type: application/json" \
    -d "{\"url\": \"http://${IP}:5005\"}"

echo ""
echo "=== Done! ==="
echo "Frontend: http://${FRONTEND}:5001"
echo "Backend:  http://${IP}:5005"
echo ""
echo "To disconnect: curl -X POST http://${FRONTEND}:5001/set_backend -H 'Content-Type: application/json' -d '{\"url\": \"http://localhost:5005\"}'"
