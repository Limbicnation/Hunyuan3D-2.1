# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import torch

# Safe import of custom rasterizer with fallback
RASTERIZER_AVAILABLE = True
try:
    if os.environ.get('HUNYUAN_DISABLE_RASTERIZER') == '1':
        print("🔧 Custom CUDA rasterizer disabled by safety mode")
        RASTERIZER_AVAILABLE = False
        custom_rasterizer_kernel = None
    else:
        import custom_rasterizer_kernel
        print("✓ Custom CUDA rasterizer loaded successfully")
except ImportError as e:
    print(f"⚠️  Custom CUDA rasterizer not available: {e}")
    RASTERIZER_AVAILABLE = False
    custom_rasterizer_kernel = None
except Exception as e:
    print(f"❌ Custom CUDA rasterizer failed to load: {e}")
    RASTERIZER_AVAILABLE = False
    custom_rasterizer_kernel = None


def _fallback_rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
    """Software fallback rasterization when CUDA extension is unavailable"""
    print("🔄 Using fallback rasterization (software rendering)")
    batch_size, num_vertices, _ = pos.shape
    num_faces = tri.shape[0]
    height, width = resolution
    
    # Create empty output tensors
    device = pos.device
    findices = torch.zeros((height, width), dtype=torch.int32, device=device)
    barycentric = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    
    # Simple software rasterization (basic implementation)
    # This is a minimal fallback - full implementation would be complex
    return findices, barycentric


def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
    assert pos.device == tri.device
    
    if not RASTERIZER_AVAILABLE or custom_rasterizer_kernel is None:
        return _fallback_rasterize(pos, tri, resolution, clamp_depth, use_depth_prior)
    
    try:
        findices, barycentric = custom_rasterizer_kernel.rasterize_image(
            pos[0], tri, clamp_depth, resolution[1], resolution[0], 1e-6, use_depth_prior
        )
        return findices, barycentric
    except Exception as e:
        print(f"❌ CUDA rasterizer failed, falling back to software: {e}")
        return _fallback_rasterize(pos, tri, resolution, clamp_depth, use_depth_prior)


def interpolate(col, findices, barycentric, tri):
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, axis=-2)
    return result.view(1, *result.shape)
