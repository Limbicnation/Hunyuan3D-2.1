#!/usr/bin/env python3
"""
Batch Processing Script for Hunyuan3D-2.1
Generates 3D meshes with PBR textures from multiple input images.

Usage:
    python batch_process.py --input_dir ./my_images --output_dir ./output
    python batch_process.py --input_dir ./images --output_dir ./output --resume
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import time
import csv
import traceback
from typing import List, Dict, Tuple

# Set PyTorch CUDA memory allocator to use expandable segments to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add project paths
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Apply torchvision compatibility fix
try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix not found")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

import torch
from PIL import Image
from tqdm import tqdm

# Import Hunyuan3D pipelines
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dpaint.convert_utils import create_glb_with_pbr_materials


def quick_convert_with_obj2gltf(obj_path: str, glb_path: str) -> bool:
    # Execute conversion
    textures = {
        'albedo': obj_path.replace('.obj', '.jpg'),
        'metallic': obj_path.replace('.obj', '_metallic.jpg'),
        'roughness': obj_path.replace('.obj', '_roughness.jpg')
    }
    create_glb_with_pbr_materials(obj_path, textures, glb_path)



# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# Quality preset: Balanced
QUALITY_PRESET = {
    'resolution': 1024,
    'max_num_view': 12,
    'num_inference_steps': 50,
    'guidance_scale': 7.5,
}


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_file = output_dir / 'batch_errors.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def discover_images(input_dir: Path) -> List[Path]:
    """Discover all supported image files in input directory."""
    images = []
    for ext in SUPPORTED_FORMATS:
        images.extend(input_dir.glob(f'*{ext}'))
        images.extend(input_dir.glob(f'*{ext.upper()}'))

    return sorted(set(images))


def filter_processed_images(images: List[Path], output_dir: Path, resume: bool) -> List[Path]:
    """Filter out already-processed images if resume mode is enabled."""
    if not resume:
        return images

    to_process = []
    for img_path in images:
        output_name = img_path.stem + '.glb'
        output_file = output_dir / output_name

        if not output_file.exists():
            to_process.append(img_path)

    return to_process


def process_single_image(
    image_path: Path,
    output_dir: Path,
    shape_pipeline,
    paint_pipeline,
    bg_remover,
    apply_bg_removal: bool,
    logger: logging.Logger
) -> Dict:
    """
    Process a single image through shape and texture generation.

    Returns:
        Dict with processing stats: status, time, error, output_file, file_size
    """
    start_time = time.time()
    result = {
        'image': image_path.name,
        'status': 'failed',
        'processing_time': 0,
        'error': None,
        'output_file': None,
        'file_size_mb': 0
    }

    try:
        # Load image
        image = Image.open(image_path).convert("RGBA")

        # Apply background removal if requested
        if apply_bg_removal:
            if image.mode == 'RGB':
                image = image.convert('RGBA')
            image = bg_remover(image)

        # Generate 3D shape
        logger.info(f"Generating shape for {image_path.name}...")
        mesh = shape_pipeline(
            image=image,
            num_inference_steps=QUALITY_PRESET['num_inference_steps'],
            guidance_scale=QUALITY_PRESET['guidance_scale']
        )[0]

        # Clear CUDA cache after shape generation to prevent memory fragmentation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Save intermediate shape mesh (OBJ format for pymeshlab compatibility)
        temp_mesh_path = output_dir / f"{image_path.stem}_shape_temp.obj"
        mesh.export(str(temp_mesh_path))

        # Generate textures
        logger.info(f"Generating textures for {image_path.name}...")
        output_glb = output_dir / f"{image_path.stem}.glb"
        temp_textured_mesh_path = output_dir / f"{image_path.stem}_textured_temp.obj"

        paint_pipeline(
            mesh_path=str(temp_mesh_path),
            image_path=image,
            output_mesh_path=str(temp_textured_mesh_path),
            save_glb=False
        )

        # Convert textured OBJ to GLB
        quick_convert_with_obj2gltf(str(temp_textured_mesh_path), str(output_glb))


        # Aggressive memory cleanup after texture generation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Clean up temporary files
        if temp_mesh_path.exists():
            temp_mesh_path.unlink()
        if temp_textured_mesh_path.exists():
            temp_textured_mesh_path.unlink()
            # Also remove associated texture files
            for ext in ['.jpg', '_metallic.jpg', '_roughness.jpg']:
                tex_file = Path(str(temp_textured_mesh_path).replace('.obj', ext))
                if tex_file.exists():
                    tex_file.unlink()

        # Calculate file size
        if output_glb.exists():
            file_size_mb = output_glb.stat().st_size / (1024 * 1024)
            result['file_size_mb'] = round(file_size_mb, 2)
            result['output_file'] = output_glb.name
            result['status'] = 'success'

        # Clear CUDA cache
        torch.cuda.empty_cache()

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        result['error'] = error_msg
        logger.error(f"Failed to process {image_path.name}: {error_msg}")
        logger.debug(traceback.format_exc())

        # Clear CUDA cache even on error
        torch.cuda.empty_cache()

    finally:
        result['processing_time'] = round(time.time() - start_time, 2)

    return result


def generate_csv_report(results: List[Dict], output_dir: Path):
    """Generate CSV report with processing statistics."""
    report_path = output_dir / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(report_path, 'w', newline='') as csvfile:
        fieldnames = ['image', 'status', 'processing_time', 'file_size_mb', 'output_file', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    return report_path


def print_summary(results: List[Dict], total_time: float):
    """Print processing summary statistics."""
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = total - successful

    avg_time = sum(r['processing_time'] for r in results) / total if total > 0 else 0
    total_size = sum(r['file_size_mb'] for r in results if r['status'] == 'success')

    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total images processed: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"Average time per image: {avg_time:.2f}s")
    print(f"Total processing time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Total output size: {total_size:.2f} MB")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Batch process images to generate 3D meshes with textures'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing input images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory for output GLB files'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='tencent/Hunyuan3D-2.1',
        help='HuggingFace model path for shape generation'
    )
    parser.add_argument(
        '--texgen_model_path',
        type=str,
        default='tencent/Hunyuan3D-2.1',
        help='HuggingFace model path for texture generation'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip already-processed images (resume interrupted batch)'
    )
    parser.add_argument(
        '--no_bg_removal',
        action='store_true',
        help='Disable automatic background removal'
    )
    parser.add_argument(
        '--low_vram_mode',
        action='store_true',
        help='Enable low VRAM mode for GPUs with limited memory'
    )

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("="*60)
    logger.info("Starting Hunyuan3D Batch Processing")
    logger.info("="*60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Quality preset: Balanced (1024px, 12 views, 50 steps)")
    logger.info(f"Background removal: {'Disabled' if args.no_bg_removal else 'Enabled'}")
    logger.info(f"Resume mode: {'Enabled' if args.resume else 'Disabled'}")

    # Discover images
    print("\nDiscovering images...")
    all_images = discover_images(input_dir)
    if not all_images:
        print(f"No images found in {input_dir}")
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        sys.exit(1)

    print(f"Found {len(all_images)} images")

    # Filter already-processed images if resume mode
    images_to_process = filter_processed_images(all_images, output_dir, args.resume)

    if args.resume and len(images_to_process) < len(all_images):
        skipped = len(all_images) - len(images_to_process)
        print(f"Resume mode: Skipping {skipped} already-processed images")

    if not images_to_process:
        print("All images already processed. Nothing to do.")
        sys.exit(0)

    print(f"Processing {len(images_to_process)} images...\n")

    # Initialize pipelines
    print("Loading models...")
    logger.info("Initializing shape generation pipeline...")

    # Setup shape pipeline
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        subfolder="hunyuan3d-dit-v2-1",
        torch_dtype=torch.float16 if args.low_vram_mode else torch.float32,
        device="cuda",  # Already moved to CUDA during initialization
    )

    # Setup texture pipeline
    logger.info("Initializing texture generation pipeline...")
    config = Hunyuan3DPaintConfig(
        max_num_view=QUALITY_PRESET['max_num_view'],
        resolution=QUALITY_PRESET['resolution'],
    )
    config.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    config.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    config.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    config.paint_model_path = args.texgen_model_path

    paint_pipeline = Hunyuan3DPaintPipeline(config)

    # Initialize background remover
    bg_remover = None
    if not args.no_bg_removal:
        logger.info("Initializing background remover...")
        bg_remover = BackgroundRemover()

    print("Models loaded successfully!\n")

    # Process images with progress bar
    results = []
    total_start_time = time.time()

    with tqdm(images_to_process, desc="Processing images", unit="image") as pbar:
        for image_path in pbar:
            pbar.set_description(f"Processing {image_path.name}")

            result = process_single_image(
                image_path,
                output_dir,
                shape_pipeline,
                paint_pipeline,
                bg_remover,
                not args.no_bg_removal,
                logger
            )

            results.append(result)

            # Update progress bar with status
            status_icon = "✓" if result['status'] == 'success' else "✗"
            pbar.set_postfix({
                'status': status_icon,
                'time': f"{result['processing_time']:.1f}s"
            })

    total_time = time.time() - total_start_time

    # Generate CSV report
    print("\nGenerating report...")
    report_path = generate_csv_report(results, output_dir)
    print(f"Report saved to: {report_path}")

    # Print summary
    print_summary(results, total_time)

    logger.info("Batch processing completed!")


if __name__ == "__main__":
    main()
