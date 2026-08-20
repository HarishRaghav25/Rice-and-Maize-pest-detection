#!/bin/bash
# ============================================================
#  Jetson Nano 2GB Setup Script for YOLO Rice Pest Detection
#  Compatible with JetPack 4.6.x (Jetson Nano 2GB Developer Kit)
# ============================================================

set -e

echo "============================================================"
echo "  YOLO Rice Pest Detection -- Jetson Nano 2GB Setup"
echo "============================================================"

# 1. Setup swap (critical for 2GB model — prevents OOM during model loading)
echo "[1/7] Setting up 4GB swap for Jetson Nano 2GB..."
if [ ! -f /var/swapfile ]; then
    sudo fallocate -l 4G /var/swapfile
    sudo chmod 600 /var/swapfile
    sudo mkswap /var/swapfile
    sudo swapon /var/swapfile
    echo '/var/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "  Swap created and enabled (4GB)"
else
    sudo swapon /var/swapfile 2>/dev/null || true
    echo "  Swap already exists, enabled"
fi

# 2. Set maximum performance mode
echo "[2/7] Setting Jetson Nano 2GB to maximum performance mode..."
sudo nvpmodel -m 0 2>/dev/null || echo "  (nvpmodel not available -- skipping)"
sudo jetson_clocks 2>/dev/null || echo "  (jetson_clocks not available -- skipping)"

# 3. Update pip
echo "[3/7] Upgrading pip..."
pip3 install --upgrade pip

# 4. Install Python dependencies
echo "[4/7] Installing Python dependencies..."
pip3 install --upgrade numpy opencv-python Pillow

# 5. Install ONNX Runtime with GPU support
echo "[5/7] Installing ONNX Runtime..."
# Try GPU version first (requires CUDA), fall back to CPU
pip3 install onnxruntime-gpu 2>/dev/null || {
    echo "  GPU version failed, installing CPU version..."
    pip3 install onnxruntime
}

# 6. Verify installation
echo "[6/7] Verifying installation..."
python3 -c "
import cv2
import numpy as np
import onnxruntime as ort

print(f'  OpenCV     : {cv2.__version__}')
print(f'  NumPy      : {np.__version__}')
print(f'  ORT        : {ort.__version__}')
print(f'  Providers  : {ort.get_available_providers()}')
print()
print('  All dependencies verified successfully!')
"

# 7. Check for CUDA
echo "[7/7] Checking GPU availability..."
python3 -c "
try:
    import onnxruntime as ort
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print('  GPU acceleration available (CUDA)')
    elif 'TensorrtExecutionProvider' in ort.get_available_providers():
        print('  GPU acceleration available (TensorRT)')
    else:
        print('  WARNING: GPU acceleration NOT available -- inference will use CPU')
        print('     For GPU support, install: pip3 install onnxruntime-gpu')
except Exception as e:
    print(f'  Check failed: {e}')
"

echo ""
echo "============================================================"
echo "  Setup Complete! (Jetson Nano 2GB)"
echo ""
echo "  Usage:"
echo "    # Run on a single image:"
echo "    python3 jetson_inference.py --model pest_detector.onnx --source photo.jpg"
echo ""
echo "    # Run with USB camera:"
echo "    python3 jetson_inference.py --model pest_detector.onnx --source 0"
echo ""
echo "    # Run with CSI camera (Jetson Nano):"
echo "    python3 jetson_inference.py --model pest_detector.onnx --source csi"
echo ""
echo "  Memory tips for 2GB kit:"
echo "    - Use imgsz=320 (default) for best memory/speed balance"
echo "    - Close all other applications before running inference"
echo "    - The 4GB swap file is already configured"
echo "============================================================"
