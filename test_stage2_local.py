#!/usr/bin/env python3
"""
Local test script for Stage-2 Dense Prediction Decoder on M1118880788RC

This script:
1. Loads the NAC image M1118880788RC
2. Extracts a 256x256 tile (or uses the full image)
3. Runs it through DINOv3 to get spatial tokens
4. Processes through Stage-2 decoder
5. Validates results with physics
6. Saves visualization
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Import Luna modules
from luna.models.dinov3 import DINOEncoder
from luna.models.stage2_decoder import Stage2DenseDecoder, Stage2Config, PhysicsValidator
from luna.io.nac_reader import read_nac


def main():
    print("=" * 60)
    print("STAGE-2 DENSE PREDICTION DECODER - LOCAL TEST")
    print("Testing on NAC: M1118880788RC")
    print("=" * 60)
    print()
    
    # Configuration - use CPU for stability to avoid MPS FP16 issues
    device = "cpu"  # Force CPU to avoid MPS FP16 type conflicts
    print(f"Using device: {device} (forced CPU for stability)")
    print()
    
    # NAC image path
    nac_path = project_root / "data/_scratch" / "M1118880788RC.IMG"
    if not nac_path.exists():
        print(f"NAC image not found at {nac_path}")
        # Try alternative locations
        alt_paths = [
            project_root / "M1118880788RC.IMG",
            project_root / "data" / "M1118880788RC.IMG",
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                nac_path = alt_path
                break
    
    if not nac_path.exists():
        print("ERROR: NAC image not found. Please ensure M1118880788RC.IMG is available.")
        return 1
    
    print(f"Loading NAC image from: {nac_path}")
    
    # Read NAC image
    try:
        print("Reading NAC image...")
        start_time = time.time()
        nac_img = read_nac(nac_path)
        read_time = time.time() - start_time
        print(f"NAC image loaded in {read_time:.2f}s")
        print(f"  Image shape: {nac_img.pixels.shape}")
        print(f"  Pixel scale: {getattr(nac_img, 'pixel_scale', 'unknown')} m/px")
        print(f"  Sub-solar azimuth: {getattr(nac_img, 'sub_solar_azimuth', 'unknown')}°")
        print(f"  Incidence angle: {getattr(nac_img, 'incidence_angle', 'unknown')}°")
        
        # Get metadata
        image_data = nac_img.pixels.astype(np.float32)
        sub_solar_azimuth = getattr(nac_img, 'sub_solar_azimuth', 180.0)
        incidence_angle = getattr(nac_img, 'incidence_angle', 45.0)
        pixel_scale = getattr(nac_img, 'pixel_scale', 0.5)
        
    except Exception as e:
        print(f"ERROR reading NAC image: {e}")
        return 1
    
    print()
    
    # Extract a 256x256 tile for testing (center of image)
    print("Extracting 256x256 tile from center...")
    h, w = image_data.shape
    y_start = (h - 256) // 2
    x_start = (w - 256) // 2
    tile = image_data[y_start:y_start+256, x_start:x_start+256]
    
    print(f"  Original image: {h}x{w}")
    print(f"  Extracted tile: 256x256 from ({x_start}, {y_start})")
    
    # Normalize tile
    lo, hi = tile.min(), tile.max()
    if hi - lo > 0:
        tile_normalized = (tile - lo) / (hi - lo)
    else:
        tile_normalized = tile
    
    # Convert to uint8 for display
    tile_display = ((tile_normalized - tile_normalized.min()) / 
                   (tile_normalized.max() - tile_normalized.min() + 1e-6) * 255).astype(np.uint8)
    
    # Save original tile for reference
    output_dir = project_root / "test_output_stage2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the original tile image
    Image.fromarray(tile_display).save(output_dir / "M1118880788RC_tile_original.png")
    print(f"  Saved original tile to: {output_dir / 'M1118880788RC_tile_original.png'}")
    
    print()
    
    # Initialize DINOv3 encoder (uses FP16 on MPS/CUDA)
    print("Initializing DINOv3 encoder...")
    from luna.config import HF_REPO_ID
    try:
        encoder = DINOEncoder(
            lora_dir=HF_REPO_ID,
            base_weights_path=HF_REPO_ID,
            matryoshka_dim=384,
            device=device,
        )
        print(f"  DINOv3 encoder initialized with {HF_REPO_ID}")
    except Exception as e:
        print(f"  WARNING: Could not initialize DINOv3 encoder: {e}")
        print("  Using dummy tokens for Stage-2 test")
        encoder = None
    
    print()
    
    # Initialize Stage-2 decoder
    print("Initializing Stage-2 Dense Decoder...")
    config = Stage2Config()
    decoder = Stage2DenseDecoder(config=config, device=device)
    print(f"  Stage-2 decoder initialized with {decoder.get_parameter_count():,} parameters")
    
    print()
    
    # Process tile through DINOv3 (if available) or use dummy tokens
    if encoder is not None:
        print("Running DINOv3 forward pass...")
        start_time = time.time()
        
        # Prepare input for DINOv3 (use FP16 consistently on MPS/CUDA)
        tile_tensor = torch.from_numpy(tile_normalized).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16 if device in ["mps", "cuda"] else torch.float32)
        if tile_tensor.shape[1] == 1:
            tile_tensor = tile_tensor.repeat(1, 3, 1, 1)
        
        # Normalize for DINOv3
        from torchvision import transforms
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        tile_normalized_dino = normalize(tile_tensor)
        
        # Get features
        with torch.no_grad():
            features = encoder._backbone_module.forward_features(tile_normalized_dino)
            print(f"  Features keys: {list(features.keys())}")
            
            # Extract spatial tokens
            spatial_tokens = decoder.token_extractor.extract_spatial_tokens(features)
            print(f"  Spatial tokens shape: {spatial_tokens.shape}")
        
        dino_time = time.time() - start_time
        print(f"  DINOv3 forward pass completed in {dino_time:.2f}s")
        print(f"  Spatial tokens shape: {spatial_tokens.shape}")
    else:
        # Use dummy tokens
        spatial_tokens = torch.randn(1, 256, 1024, device=device)
        print(f"  Using dummy spatial tokens: {spatial_tokens.shape}")
    
    print()
    
    # Run Stage-2 decoder
    print("Running Stage-2 Dense Decoder...")
    start_time = time.time()
    
    with torch.no_grad():
        # Get probability masks
        prob_masks = decoder.get_probability_masks(spatial_tokens)
        logits = decoder(spatial_tokens)
    
    stage2_time = time.time() - start_time
    print(f"  Stage-2 forward pass completed in {stage2_time:.2f}s")
    print(f"  Probability masks shape: {prob_masks.shape}")
    
    # Convert to numpy
    prob_masks_np = prob_masks.cpu().numpy()[0]  # Remove batch dim: (3, 256, 256)
    
    # Get class predictions
    class_preds = np.argmax(prob_masks_np, axis=0)
    
    # Extract individual class masks
    regolith_mask = prob_masks_np[0, :, :]
    shadow_mask = prob_masks_np[1, :, :]  # True Void / Pit
    edge_mask = prob_masks_np[2, :, :]    # Obstacle / Boulder
    
    print(f"  Class distribution:")
    print(f"    Regolith: {np.sum(class_preds == 0):6d} pixels ({100*np.sum(class_preds == 0)/256/256:.1f}%)")
    print(f"    Pit Shadow: {np.sum(class_preds == 1):6d} pixels ({100*np.sum(class_preds == 1)/256/256:.1f}%)")
    print(f"    Boulder: {np.sum(class_preds == 2):6d} pixels ({100*np.sum(class_preds == 2)/256/256:.1f}%)")
    
    print()
    
    # Physics validation
    print("Running Physics Validation...")
    validator = PhysicsValidator()
    
    # Create binary masks (threshold at 0.5)
    shadow_binary = shadow_mask > 0.5
    edge_binary = edge_mask > 0.5
    
    result = validator.validate_detection(
        shadow_mask=shadow_binary,
        edge_mask=edge_binary,
        sub_solar_azimuth=sub_solar_azimuth,
        incidence_angle_deg=incidence_angle,
        pixel_scale=pixel_scale,
    )
    
    print(f"  Physics Validation Results:")
    print(f"    Valid Pit: {result.is_valid_pit}")
    print(f"    Is Boulder: {result.is_boulder}")
    print(f"    Depth Estimate: {result.depth_estimate_meters:.2f}m")
    print(f"    Shadow Vector: {result.shadow_vector_degrees:.1f}°")
    print(f"    Solar Vector: {result.solar_vector_degrees:.1f}°")
    print(f"    Alignment Error: {result.alignment_error_degrees:.1f}°")
    print(f"    Shadow Width: {result.width_pixels}px")
    print(f"    Shadow Height: {result.height_pixels}px")
    print(f"    Confidence: {result.class_confidence:.3f}")
    
    print()
    
    # Save visualization
    print("Saving visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original tile
    axes[0, 0].imshow(tile_display, cmap='gray')
    axes[0, 0].set_title(f'Original NAC Tile\nM1118880788RC [{x_start}:{x_start+256}, {y_start}:{y_start+256}]')
    axes[0, 0].axis('off')
    
    # Class predictions
    try:
        cmap = plt.cm.get_cmap('viridis', 3)
    except AttributeError:
        from matplotlib.colors import ListedColormap
        cmap = plt.get_cmap('viridis', 3) if hasattr(plt, 'get_cmap') else plt.cm.viridis
    class_show = axes[0, 1].imshow(class_preds, cmap=cmap, vmin=0, vmax=2)
    axes[0, 1].set_title('Class Predictions\n(0=Regolith, 1=Pit Shadow, 2=Boulder)')
    axes[0, 1].axis('off')
    plt.colorbar(class_show, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Shadow probability
    shadow_show = axes[0, 2].imshow(shadow_mask, cmap='Blues', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Pit Shadow Probability\nMax: {shadow_mask.max():.3f}')
    axes[0, 2].axis('off')
    plt.colorbar(shadow_show, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Edge probability
    edge_show = axes[1, 0].imshow(edge_mask, cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Boulder Probability\nMax: {edge_mask.max():.3f}')
    axes[1, 0].axis('off')
    plt.colorbar(edge_show, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Physics info
    info_text = (
        f"Physics Validation Results:\n\n"
        f"  Valid Pit Detection: {result.is_valid_pit}\n"
        f"  Boulder Detection: {result.is_boulder}\n"
        f"  Estimated Depth: {result.depth_estimate_meters:.2f} m\n"
        f"  Shadow Vector: {result.shadow_vector_degrees:.1f}°\n"
        f"  Solar Azimuth: {result.solar_vector_degrees:.1f}°\n"
        f"  Alignment Error: {result.alignment_error_degrees:.1f}°\n"
        f"  Shadow Size: {result.width_pixels}×{result.height_pixels} px\n"
        f"  Confidence: {result.class_confidence:.3f}"
    )
    axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes, 
                   va='center', ha='left', fontsize=10)
    axes[1, 1].set_title('Physics Validation')
    axes[1, 1].axis('off')
    
    # Combined overlay
    overlay = np.zeros((256, 256, 3), dtype=np.uint8)
    overlay[:, :, 0] = (edge_mask * 255).astype(np.uint8)  # Red for boulder
    overlay[:, :, 2] = (shadow_mask * 255).astype(np.uint8)  # Blue for pit
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Overlay: Red=Boulder, Blue=Pit')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "M1118880788RC_stage2_results.png", dpi=150)
    plt.close()
    
    print(f"  Visualization saved to: {output_dir / 'M1118880788RC_stage2_results.png'}")
    
    # Also save individual masks
    Image.fromarray((shadow_mask * 255).astype(np.uint8)).save(output_dir / "M1118880788RC_shadow_mask.png")
    Image.fromarray((edge_mask * 255).astype(np.uint8)).save(output_dir / "M1118880788RC_edge_mask.png")
    Image.fromarray(class_preds.astype(np.uint8) * 127).save(output_dir / "M1118880788RC_class_preds.png")
    
    print(f"  Individual masks saved to: {output_dir}")
    print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"NAC Image: M1118880788RC")
    print(f"Tile Position: ({x_start}, {y_start}) to ({x_start+256}, {y_start+256})")
    print(f"Sub-Solar Azimuth: {sub_solar_azimuth}°")
    print(f"Incidence Angle: {incidence_angle}°")
    print(f"Pixel Scale: {pixel_scale} m/px")
    print()
    print(f"Processing Times:")
    print(f"  NAC Read: {read_time:.2f}s")
    if encoder is not None:
        print(f"  DINOv3: {dino_time:.2f}s")
    print(f"  Stage-2: {stage2_time:.2f}s")
    print()
    print(f"Results:")
    print(f"  Valid Pit Detected: {result.is_valid_pit}")
    print(f"  Estimated Depth: {result.depth_estimate_meters:.2f}m")
    print(f"  Alignment Error: {result.alignment_error_degrees:.1f}°")
    print(f"  Confidence: {result.class_confidence:.3f}")
    print()
    print(f"Output Directory: {output_dir.absolute()}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())