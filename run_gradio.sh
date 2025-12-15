#!/bin/bash
# Hunyuan3D-2.1 Gradio Web App Launcher
# Environment: hunyuan3d2.1
# Access at: http://localhost:8082

cd "$(dirname "$0")"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate hunyuan3d2.1

python3 gradio_app.py \
  --model_path tencent/Hunyuan3D-2.1 \
  --subfolder hunyuan3d-dit-v2-1 \
  --texgen_model_path tencent/Hunyuan3D-2.1 \
  --low_vram_mode
