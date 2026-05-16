#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$PROJECT_DIR/Hunyuan3D-2.1"

echo "=== image-to-3d setup ==="
echo "Project dir: $PROJECT_DIR"

# ── 1. Clone Hunyuan3D-2.1 ──────────────────────────────────────────────
if [ -d "$REPO_DIR" ]; then
    echo "Hunyuan3D-2.1 already cloned, pulling latest..."
    git -C "$REPO_DIR" pull --ff-only || true
else
    echo "Cloning Hunyuan3D-2.1..."
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$REPO_DIR"
fi

# ── 2. Create conda environment ─────────────────────────────────────────
if conda env list | grep -q "^image-to-3d "; then
    echo "Conda env 'image-to-3d' already exists, skipping create."
else
    echo "Creating conda env 'image-to-3d' with Python 3.10..."
    conda create -y -n image-to-3d python=3.10
fi

echo "Activating image-to-3d env..."
eval "$(conda shell.bash hook)"
conda activate image-to-3d

# ── 3. Install PyTorch (CUDA 12.4) ──────────────────────────────────────
echo "Installing PyTorch 2.5.1 + CUDA 12.4..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ── 4. Install Hunyuan3D-2.1 dependencies ───────────────────────────────
echo "Installing Hunyuan3D-2.1 requirements..."
pip install -r "$REPO_DIR/requirements.txt"

# ── 5. Build C++ extensions ─────────────────────────────────────────────
echo "Building custom rasterizer..."
cd "$REPO_DIR/hy3dpaint/custom_rasterizer"
pip install -e .

echo "Building differentiable renderer..."
cd "$REPO_DIR/hy3dpaint/DifferentiableRenderer"
bash compile_mesh_painter.sh

cd "$PROJECT_DIR"

# ── 6. Download Real-ESRGAN weights ─────────────────────────────────────
ESRGAN_PATH="$REPO_DIR/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
if [ ! -f "$ESRGAN_PATH" ]; then
    echo "Downloading Real-ESRGAN weights..."
    mkdir -p "$(dirname "$ESRGAN_PATH")"
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
        -O "$ESRGAN_PATH"
else
    echo "Real-ESRGAN weights already present."
fi

# ── 7. HuggingFace login check ──────────────────────────────────────────
echo ""
echo "Model weights auto-download from HuggingFace on first run (~15 GB)."
echo "You must accept the license at: https://huggingface.co/tencent/Hunyuan3D-2.1"
echo ""
if ! python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    echo "Not logged into HuggingFace. Run: huggingface-cli login"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate image-to-3d"
echo "Generate with: python generate.py input/photo.png -o output/model.glb"
