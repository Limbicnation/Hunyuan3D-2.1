# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hunyuan3D-2.1 is a production-ready AI system for generating textured 3D assets from single images. Two-stage pipeline:
- **Hunyuan3D-Shape (3.3B params)**: Image-to-3D mesh via Diffusion Transformer with flow matching
- **Hunyuan3D-Paint (2B params)**: PBR texture synthesis via multiview diffusion (albedo, metallic, roughness, normal maps)

VRAM: 10GB (shape only), 21GB (texture only), 29GB (full pipeline)

## Build & Setup

```bash
# 1. PyTorch with CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 2. Dependencies
pip install -r requirements.txt

# 3. Build CUDA rasterizer (required for texture generation)
cd hy3dpaint/custom_rasterizer && pip install -e . && cd ../..

# 4. Build mesh painter
cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh && cd ../..

# 5. Download RealESRGAN weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt
```

## Running the Pipeline

```bash
# Quick test - full pipeline
python demo.py

# Web UI (Gradio)
python gradio_app.py --model_path tencent/Hunyuan3D-2.1 --low_vram_mode

# Batch processing
python batch_process.py --input_dir ./my_images --output_dir ./output --resume

# REST API server
python api_server.py  # http://localhost:8081/docs

# Shape-only test
python hy3dshape/minimal_demo.py

# Texture-only test
python hy3dpaint/demo.py

# API tests
python test_api_server.py
```

## Architecture

### Pipeline Flow
```
Input Image → Background Removal (rembg) → DINOv2 Encoder → DiT Denoiser → ShapeVAE → Marching Cubes → 3D Mesh
     ↓
Mesh + Image → UV Unwrap (xatlas) → Multiview Rendering → UNet2.5D Diffusion → PBR Textures → GLB Export
```

### Key Modules

**hy3dshape/** - Shape generation
- `hy3dshape/pipelines.py` - Main pipeline: `Hunyuan3DDiTFlowMatchingPipeline`
- `hy3dshape/models/denoisers/hunyuan3ddit.py` - DiT architecture with MOE layers
- `hy3dshape/models/autoencoders/model.py` - ShapeVAE decoder
- `hy3dshape/models/autoencoders/surface_extractors.py` - Marching Cubes variants

**hy3dpaint/** - Texture generation
- `textureGenPipeline.py` - Main pipeline: `Hunyuan3DPaintPipeline`
- `hunyuanpaintpbr/pipeline.py` - HunyuanPaintPipeline (extends StableDiffusion)
- `hunyuanpaintpbr/unet/model.py` - UNet2.5D with position-aware RoPE attention
- `DifferentiableRenderer/MeshRender.py` - Custom renderer with texture baking
- `custom_rasterizer/` - CUDA rasterizer kernel

### Entry Points
```
demo.py              - Simple full pipeline demo
gradio_app.py        - Web UI (port 7860)
batch_process.py     - Batch processing with resume
api_server.py        - REST API (port 8081)
hy3dshape/main.py    - Shape model training
hy3dpaint/train.py   - Texture model training
```

## Critical Implementation Details

### 1. Path Configuration (REQUIRED)
All scripts must configure paths before imports:
```python
import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')
```

### 2. Torchvision Compatibility Fix
Apply early in scripts:
```python
try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    pass
```

### 3. Model Loading (CRITICAL)
```python
# CORRECT - pass device to from_pretrained
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2.1',
    device='cuda',
    dtype=torch.float16
)

# WRONG - .to() returns None
# shape_pipeline = shape_pipeline.to('cuda')  # DON'T DO THIS
```

### 4. Intermediate Mesh Format (CRITICAL)
Texture pipeline requires OBJ format for intermediate meshes (not GLB):
```python
# Shape output - use OBJ
mesh.export('output.obj')  # NOT .glb

# Texture pipeline reads with pymeshlab
paint_pipeline(mesh_path='output.obj', ...)

# Final output is GLB with PBR materials
```

### 5. CUDA Memory Management
```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# After each inference in batch processing:
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

## Training

### Shape Model
```bash
cd hy3dshape
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export config=configs/hunyuandit-finetuning-flowmatching-dinol518-bf16-lr1e5-4096.yaml
export output_dir=output_folder/dit/finetuning
bash scripts/train_deepspeed.sh 1 0 8 0.0.0.0 $config $output_dir
```

Data structure: `dataset/preprocessed/{uid}/geo_data/` (SDF volumes) + `render_cond/` (24 views)

### Texture Model
```bash
python3 hy3dpaint/train.py \
  --base hy3dpaint/cfgs/hunyuan-paint-pbr.yaml \
  --name experiment_name \
  --logdir logs/
```

Data structure: `train_examples/{uid}/render_tex/` (RGB, albedo, metallic-roughness, normal, position maps)

## Configuration Presets

### Shape Generation
```python
# Fast: num_inference_steps=5, octree_resolution=128
# Balanced: num_inference_steps=50, octree_resolution=256
# High quality: num_inference_steps=50, octree_resolution=512
```

### Texture Generation
```python
Hunyuan3DPaintConfig(
    max_num_view=6,    # 6-12 (more views = better quality, more VRAM)
    resolution=512,    # 512 or 768
)
```

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: hunyuanpaintpbr` | Missing sys.path | Add `sys.path.insert(0, './hy3dpaint')` |
| `TypeError: 'NoneType' object is not callable` | Using `.to('cuda')` on pipeline | Pass `device='cuda'` to `from_pretrained()` |
| `ValueError: incorrect header on GLB` | pymeshlab can't read trimesh GLB | Use OBJ for intermediate files |
| Custom rasterizer not found | Not built | `cd hy3dpaint/custom_rasterizer && pip install -e .` |
| Mesh inpainting error | C++ not compiled | `cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh` |
| CUDA OOM | High VRAM settings | Reduce `max_num_view`, `resolution`, or `octree_resolution` |

## Docker

```bash
cd docker
docker build -t hunyuan3d21:latest .  # ~70GB, 1+ hour
docker run -it --name hy3d21 -p 7860:7860 --gpus all hunyuan3d21 python gradio_app.py --port 7860
```

## API Endpoints (api_server.py)

- `POST /generate` - Synchronous generation (base64 image → GLB)
- `POST /send` - Async generation (returns task ID)
- `GET /status/{uid}` - Check task status
- `GET /health` - Health check

Interactive docs: http://localhost:8081/docs
