#!/bin/bash

# Hunyuan3D 2.1 Debug Launch Script
# WSL compatibility fixes for RTX 4090 24GB with segfault debugging

set -e  # Exit on any error

# Parse command line arguments for different modes
MODE="full"
DEBUG=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --debug)
      DEBUG=1
      shift
      ;;
    --safe)
      MODE="safe"
      shift
      ;;
    --no-bpy)
      MODE="no-bpy"
      shift
      ;;
    --cpu-only)
      MODE="cpu-only"
      shift
      ;;
    --minimal)
      MODE="minimal"
      shift
      ;;
    *)
      break
      ;;
  esac
done

# Set environment variables for Pydantic compatibility
export PYDANTIC_PRIVATE_ALLOW_UNHANDLED_SCHEMA_TYPES=1

# CUDA debugging environment variables
if [ "$DEBUG" = "1" ]; then
  export CUDA_LAUNCH_BLOCKING=1
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export PYTHONFAULTHANDLER=1
  echo "🐛 Debug mode enabled with CUDA debugging"
fi

# WSL2 GPU compatibility
export CUDA_VISIBLE_DEVICES=0
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Memory management for WSL2
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}

echo "🚀 Starting Hunyuan3D 2.1 in '$MODE' mode..."
echo "✓ Pydantic environment variable set"

# Build command based on mode
CMD="python3 gradio_app.py"
CMD="$CMD --model_path tencent/Hunyuan3D-2.1"
CMD="$CMD --subfolder hunyuan3d-dit-v2-1"
CMD="$CMD --texgen_model_path tencent/Hunyuan3D-2.1"
CMD="$CMD --host 0.0.0.0 --port 7860"

case $MODE in
  "safe")
    CMD="$CMD --low_vram_mode --safe_mode"
    echo "✓ Safe mode: minimal GPU usage"
    ;;
  "no-bpy")
    CMD="$CMD --low_vram_mode --disable_bpy"
    echo "✓ No-bpy mode: Blender functionality disabled"
    ;;
  "cpu-only")
    CMD="$CMD --cpu_mode"
    echo "✓ CPU-only mode: no GPU rendering"
    ;;
  "minimal")
    CMD="$CMD --low_vram_mode --disable_rasterizer --disable_bpy"
    echo "✓ Minimal mode: custom rasterizer and bpy disabled"
    ;;
  *)
    CMD="$CMD --low_vram_mode"
    echo "✓ Full mode: all features enabled"
    echo "✓ Real bpy package installed for full Blender functionality"
    ;;
esac

echo "📝 Executing: $CMD"
echo "🔧 Use --debug for detailed debugging, --safe for minimal GPU usage"
echo "🔧 Other modes: --no-bpy, --cpu-only, --minimal"

# Create error handling wrapper
handle_error() {
  local exit_code=$?
  echo "❌ Hunyuan3D crashed with exit code $exit_code"
  if [ $exit_code -eq 139 ]; then
    echo "💥 Segmentation fault detected! Try these solutions:"
    echo "   1. Run with --safe mode: $0 --safe"
    echo "   2. Try --no-bpy mode: $0 --no-bpy"
    echo "   3. Use --minimal mode: $0 --minimal"
    echo "   4. Enable debugging: $0 --debug"
  fi
  exit $exit_code
}

trap handle_error ERR

# Launch with timeout to prevent hanging
timeout 300 $CMD || handle_error

echo "🎯 Hunyuan3D should now be accessible at http://localhost:7860"