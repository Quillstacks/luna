"""Stage-2 Dense Prediction Decoder for LUNA.

Native Stage-2 implements a dense prediction decoder that operates directly on 
DINOv3 spatial patch tokens in unified memory, bypassing GeoTIFF bottlenecks.

Architecture (5 Phases):
1. Ingestion: Extract spatial patch tokens from DINOv3 forward pass
2. Reconstruction: Unflatten tokens to 2D grid + channel projection
3. Progressive Resolution: Feature Pyramid Network (FPN) upsampling cascade
4. Classification: 3-class semantic segmentation head
5. Physics Gate: Solar angle validation for pit vs. boulder discrimination

Input: DINOv3 spatial patch tokens (B, 256, 1024) for 256x256 input with 16x16 patches
Output: Pixel-wise semantic masks (B, 256, 256, 3) + physics-validated detections
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Stage2Config:
    """Configuration for Stage-2 Dense Prediction Decoder."""
    
    # Input dimensions
    input_image_size: int = 256        # Input image size in pixels
    patch_size: int = 16               # DINOv3 patch size
    dino_hidden_dim: int = 1024        # DINOv3 hidden dimension
    
    # Phase 2: Channel projection
    projection_dim: int = 256          # Target channels after projection
    
    # Phase 3: FPN dimensions
    fpn_dims: list = None              # Channel dims at each FPN level
    
    # Phase 4: Classification
    num_classes: int = 3               # Regolith, True Void, Obstacle
    
    # Phase 5: Physics
    min_depth_to_width_ratio: float = 0.1  # Minimum depth/width for valid pit
    
    def __post_init__(self):
        if self.fpn_dims is None:
            # Default FPN channel dimensions: 4 levels for 16->32->64->128->256 upsampling
            self.fpn_dims = [
                self.projection_dim,        # 16x16 -> 32x32
                self.projection_dim // 2,    # 32x32 -> 64x64
                self.projection_dim // 4,    # 64x64 -> 128x128
                self.projection_dim // 8,    # 128x128 -> 256x256
            ]


DEFAULT_CONFIG = Stage2Config()


# ---------------------------------------------------------------------------
# Phase 1: Token Extractor (hooks into DINOv3 forward pass)
# ---------------------------------------------------------------------------

class SpatialTokenExtractor:
    """Extracts spatial patch tokens from DINOv3 forward pass.
    
    DINOv3 processes input as:
    - Input: (B, 3, H, W) where H=W=256 for standard tiles
    - Patches: (H/patch_size) * (W/patch_size) = 16*16 = 256 spatial patches
    - Output tokens: (B, num_patches + 1, hidden_dim) where +1 is CLS token
    - Spatial tokens: last 256 tokens in sequence (excluding CLS)
    """
    
    def __init__(self, patch_size: int = 16, hidden_dim: int = 1024):
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
    
    def extract_spatial_tokens(self, features: dict) -> torch.Tensor:
        """Extract spatial patch tokens from DINOv3 forward_features output.
        
        Args:
            features: Output dict from backbone.forward_features()
                     Contains 'x' key with all tokens (B, num_patches+1, hidden_dim)
        
        Returns:
            Spatial tokens only: (B, num_spatial_patches, hidden_dim)
        """
        # DINOv3 forward_features returns dict with 'x' containing all tokens
        all_tokens = features['x']  # Shape: (B, num_patches+1, hidden_dim)
        
        # Exclude CLS token (first token) and return spatial tokens
        # Spatial tokens are from index 1 onwards
        spatial_tokens = all_tokens[:, 1:, :]  # Shape: (B, num_patches, hidden_dim)
        
        return spatial_tokens
    
    def get_spatial_grid_size(self, image_size: int) -> Tuple[int, int]:
        """Calculate spatial grid dimensions from image size and patch size."""
        h = w = image_size // self.patch_size
        return h, w


# ---------------------------------------------------------------------------
# Phase 2: Geometric Reconstruction Module
# ---------------------------------------------------------------------------

class GeometricReconstructor(nn.Module):
    """Reshapes flat token sequence back to 2D spatial grid and projects channels."""
    
    def __init__(self, 
                 input_dim: int = 1024, 
                 output_dim: int = 256,
                 grid_size: Tuple[int, int] = (16, 16)):
        super().__init__()
        self.grid_size = grid_size
        self.projection = nn.Linear(input_dim, output_dim)
        
        # Initialize projection with identity-like weights for stability
        nn.init.kaiming_normal_(self.projection.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """Unflatten and project spatial tokens.
        
        Args:
            spatial_tokens: (B, num_patches, input_dim) where num_patches = grid_h * grid_w
        
        Returns:
            Reconstructed 2D feature map: (B, output_dim, grid_h, grid_w)
        """
        batch_size, num_patches, _ = spatial_tokens.shape
        
        # Project channels first: (B, num_patches, input_dim) -> (B, num_patches, output_dim)
        projected = self.projection(spatial_tokens)
        
        # Reshape to 2D spatial grid: (B, output_dim, grid_h, grid_w)
        # Note: We need to permute dimensions from (B, N, C) to (B, C, H, W)
        grid_h, grid_w = self.grid_size
        feature_map = projected.permute(0, 2, 1).reshape(batch_size, -1, grid_h, grid_w)
        
        return feature_map


# ---------------------------------------------------------------------------
# Phase 3: Feature Pyramid Network (FPN) - Progressive Resolution Cascade
# ---------------------------------------------------------------------------

class FPNBlock(nn.Module):
    """Single FPN upsampling block with optional skip connection."""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 upscale_factor: int = 2,
                 use_skip: bool = False,
                 skip_channels: Optional[int] = None):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.use_skip = use_skip and skip_channels is not None
        
        # Upsampling + convolution
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Skip connection path (if enabled)
        if self.use_skip:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(skip_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
            self.fuse = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through FPN block.
        
        Args:
            x: Input feature map (B, in_channels, H, W)
            skip: Optional skip connection (B, skip_channels, H*scale, W*scale)
        
        Returns:
            Output feature map (B, out_channels, H*scale, W*scale)
        """
        x = self.upsample(x)
        
        if self.use_skip and skip is not None:
            # Upsample skip to match spatial dimensions if needed
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
            skip = self.skip_conv(skip)
            x = torch.cat([x, skip], dim=1)
            x = self.fuse(x)
        
        return x


class FeaturePyramidNetwork(nn.Module):
    """Progressive resolution cascade from 16x16 to 256x256.
    
    Architecture:
    - Stage 1: 16x16 -> 32x32 (bilinear upsample + conv)
    - Stage 2: 32x32 -> 128x128 (with skip connection from early transformer layers)
    - Stage 3: 128x128 -> 256x256 (final native resolution)
    """
    
    def __init__(self, 
                 start_size: Tuple[int, int] = (16, 16),
                 target_size: Tuple[int, int] = (256, 256),
                 start_channels: int = 256,
                 fpn_dims: list = None):
        super().__init__()
        
        if fpn_dims is None:
            fpn_dims = [256, 128, 64]  # Default: reduce channels as we go up
        
        self.start_size = start_size
        self.target_size = target_size
        
        # Calculate intermediate sizes
        # 16x16 -> 32x32 -> 128x128 -> 256x256
        self.intermediate_sizes = [
            (start_size[0] * 2, start_size[1] * 2),      # 32x32
            (start_size[0] * 8, start_size[1] * 8),      # 128x128
            target_size,                                    # 256x256
        ]
        
        # Build FPN blocks
        # We need: 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        # That's 4 upsampling steps with factor 2 each
        self.blocks = nn.ModuleList()
        
        # Ensure we have enough FPN dimensions
        while len(fpn_dims) < 4:
            fpn_dims.append(fpn_dims[-1] // 2 if fpn_dims else start_channels // 4)
        
        # Block 1: 16x16 -> 32x32
        self.blocks.append(FPNBlock(
            in_channels=start_channels,
            out_channels=fpn_dims[0],
            upscale_factor=2,
            use_skip=False,
        ))
        
        # Block 2: 32x32 -> 64x64
        self.blocks.append(FPNBlock(
            in_channels=fpn_dims[0],
            out_channels=fpn_dims[1],
            upscale_factor=2,
            use_skip=False,
        ))
        
        # Block 3: 64x64 -> 128x128
        self.blocks.append(FPNBlock(
            in_channels=fpn_dims[1],
            out_channels=fpn_dims[2],
            upscale_factor=2,
            use_skip=False,
        ))
        
        # Block 4: 128x128 -> 256x256
        self.blocks.append(FPNBlock(
            in_channels=fpn_dims[2],
            out_channels=fpn_dims[3],
            upscale_factor=2,
            use_skip=False,
        ))
        
        # Final refinement convolution
        # Use the last FPN dimension (output of final block)
        final_channels = fpn_dims[-1]
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_channels, final_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FPN cascade.
        
        Args:
            x: Input feature map (B, C, 16, 16)
        
        Returns:
            Output feature map (B, out_channels, 256, 256)
        """
        for block in self.blocks:
            x = block(x)
        
        x = self.final_conv(x)
        return x


# ---------------------------------------------------------------------------
# Phase 4: Three-Class Classification Head
# ---------------------------------------------------------------------------

class ThreeClassSegmentationHead(nn.Module):
    """Semantic segmentation head for 3 lunar surface classes.
    
    Classes:
    - Class 0: Regolith / Flat Terrain
    - Class 1: True Void (pit interior shadow)
    - Class 2: Obstacle (boulder/edge shadow)
    """
    
    def __init__(self, in_channels: int = 64, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        
        # Use progressive reduction with intermediate supervision
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict pixel-wise class probabilities.
        
        Args:
            features: Input feature map (B, C, H, W)
        
        Returns:
            Class logits (before softmax): (B, num_classes, H, W)
        """
        logits = self.classifier(features)
        return logits


# ---------------------------------------------------------------------------
# Phase 5: Physics Gate (Solar Angle Validation)
# ---------------------------------------------------------------------------

@dataclass
class PhysicsValidationResult:
    """Result of physics-based validation for a detection."""
    is_valid_pit: bool
    is_boulder: bool
    depth_estimate_meters: float
    shadow_vector_degrees: float
    solar_vector_degrees: float
    alignment_error_degrees: float
    width_pixels: int
    height_pixels: int
    class_confidence: float


class PhysicsValidator:
    """Validates detections using solar geometry from NASA metadata.
    
    Uses SubSolarAzimuth and IncidenceAngle to determine if shadow placement
    is physically consistent with a pit vs. a boulder.
    """
    
    def __init__(self, 
                 pixel_scale_meters: float = 0.5,  # 0.5 m/pixel native
                 min_depth_to_width_ratio: float = 0.1):
        self.pixel_scale = pixel_scale_meters
        self.min_depth_ratio = min_depth_to_width_ratio
    
    def validate_detection(self, 
                          shadow_mask: np.ndarray,
                          edge_mask: np.ndarray,
                          sub_solar_azimuth: float,
                          incidence_angle_deg: float,
                          pixel_scale: float = 0.5) -> PhysicsValidationResult:
        """Validate a potential pit detection using physics.
        
        Args:
            shadow_mask: Binary mask of shadow pixels (H, W)
            edge_mask: Binary mask of edge/obstacle pixels (H, W)
            sub_solar_azimuth: Solar azimuth from NASA metadata (degrees)
            incidence_angle_deg: Solar incidence angle (degrees from vertical)
            pixel_scale: Meters per pixel
        
        Returns:
            PhysicsValidationResult with validation outcome
        """
        # Convert azimuth to radians for calculation
        solar_azimuth_rad = np.radians(sub_solar_azimuth)
        incidence_rad = np.radians(incidence_angle_deg)
        
        # Calculate shadow centroid (in pixel coordinates)
        shadow_y, shadow_x = np.where(shadow_mask > 0)
        if len(shadow_x) == 0:
            return PhysicsValidationResult(
                is_valid_pit=False,
                is_boulder=False,
                depth_estimate_meters=0.0,
                shadow_vector_degrees=0.0,
                solar_vector_degrees=sub_solar_azimuth,
                alignment_error_degrees=180.0,
                width_pixels=0,
                height_pixels=0,
                class_confidence=0.0,
            )
        
        shadow_cx = np.mean(shadow_x)
        shadow_cy = np.mean(shadow_y)
        
        # Calculate edge centroid
        edge_y, edge_x = np.where(edge_mask > 0)
        if len(edge_x) == 0:
            return PhysicsValidationResult(
                is_valid_pit=False,
                is_boulder=False,
                depth_estimate_meters=0.0,
                shadow_vector_degrees=0.0,
                solar_vector_degrees=sub_solar_azimuth,
                alignment_error_degrees=180.0,
                width_pixels=0,
                height_pixels=0,
                class_confidence=0.0,
            )
        
        edge_cx = np.mean(edge_x)
        edge_cy = np.mean(edge_y)
        
        # Calculate vector from edge to shadow
        vec_x = shadow_cx - edge_cx
        vec_y = shadow_cy - edge_cy
        
        # Convert to angle (degrees)
        # In image coordinates: y increases downward, x increases right
        # arctan2 returns angle from positive x-axis, counterclockwise
        # Result is in range [-180, 180]
        shadow_vector_deg = np.degrees(np.arctan2(vec_y, vec_x))
        
        # Solar azimuth is measured clockwise from North in NAC metadata
        # NAC azimuth 0° = North, 90° = East, 180° = South, 270° = West
        # In image coordinates: 0° = East (right), 90° = South (down), 180° = West (left), 270° = North (up)
        # Conversion from compass azimuth (clockwise from North) to image math angle (counterclockwise from East):
        # image_angle = 90° - compass_azimuth
        # But since y increases downward in images, we need to invert the y-component:
        # In image space: angle from positive x-axis, clockwise (because y increases down)
        # So: image_solar_angle = compass_azimuth - 90°
        # This gives: compass 90° (East) -> image 0° (right)
        #            compass 0° (North) -> image -90° or 270° (up in image = down in math)
        #            compass 180° (South) -> image 90° (down)
        #            compass 270° (West) -> image 180° (left)
        solar_vector_deg = sub_solar_azimuth - 90.0
        # Normalize to [-180, 180]
        solar_vector_deg = solar_vector_deg % 360
        if solar_vector_deg > 180:
            solar_vector_deg -= 360
        
        # Calculate alignment error (smallest angle between the two vectors)
        alignment_error = abs(shadow_vector_deg - solar_vector_deg) % 360
        alignment_error = min(alignment_error, 360 - alignment_error)
        alignment_error = float(alignment_error)
        
        # Calculate shadow dimensions (bounding box)
        width_px = int(np.max(shadow_x) - np.min(shadow_x))
        height_px = int(np.max(shadow_y) - np.min(shadow_y))
        
        # Calculate actual shadow length from edge to farthest shadow point
        # This is more accurate than centroid distance
        # Find the point in shadow that is farthest from the edge centroid
        shadow_points_x = shadow_x - edge_cx
        shadow_points_y = shadow_y - edge_cy
        shadow_distances = np.sqrt(shadow_points_x**2 + shadow_points_y**2)
        max_shadow_distance_px = shadow_distances.max() if len(shadow_distances) > 0 else 0
        
        # Estimate depth using cotangent of incidence angle
        # Physical correct formula: H = L / tan(theta) = L * cot(theta)
        # where:
        #   H = depth of pit (meters)
        #   L = shadow length on surface (meters)
        #   theta = solar incidence angle (from vertical)
        # When sun is at zenith (theta -> 0), tan(theta) -> 0, depth -> infinity
        # When sun is at horizon (theta -> 90), tan(theta) -> infinity, depth -> 0
        shadow_length_m = max_shadow_distance_px * pixel_scale
        
        # Use cotangent: depth = shadow_length / tan(incidence_angle)
        # But protect against division by zero when incidence is near 0
        tan_incidence = np.tan(incidence_rad)
        if tan_incidence < 0.01:  # Very small angle, tan ~ 0, depth -> infinity
            # Use a minimum tan value to avoid division by zero
            # At theta=0.57°, tan(0.57°) ≈ 0.01, depth = L / 0.01 = 100*L
            tan_incidence = max(tan_incidence, 0.01)
        
        depth_m = shadow_length_m / tan_incidence
        
        # Calculate width in meters
        width_m = max(width_px, height_px) * pixel_scale
        
        # Depth-to-width ratio
        depth_width_ratio = depth_m / width_m if width_m > 0 else 0
        
        # Validation logic:
        # - For a PIT: shadow should be ON the sun-facing side of the edge
        #   Vector from edge to shadow should align with solar direction
        # - For a BOULDER: shadow would be ON the opposite side
        
        # If shadow is in solar direction from edge, it's likely a pit
        is_aligned = alignment_error < 45  # Within 45 degrees
        
        # Check depth-to-width ratio for pit plausibility
        is_deep_enough = depth_width_ratio >= self.min_depth_ratio
        
        # Final determination
        is_valid_pit = is_aligned and is_deep_enough
        is_boulder = not is_aligned or not is_deep_enough
        
        # Confidence based on alignment and depth
        confidence = 0.0
        if is_valid_pit:
            confidence = (1.0 - alignment_error / 45.0) * 0.7 + (depth_width_ratio / 1.0) * 0.3
            confidence = min(confidence, 1.0)
        
        return PhysicsValidationResult(
            is_valid_pit=is_valid_pit,
            is_boulder=is_boulder,
            depth_estimate_meters=float(depth_m),
            shadow_vector_degrees=float(shadow_vector_deg),
            solar_vector_degrees=float(solar_vector_deg),
            alignment_error_degrees=float(alignment_error),
            width_pixels=width_px,
            height_pixels=height_px,
            class_confidence=float(confidence),
        )


# ---------------------------------------------------------------------------
# Complete Stage-2 Decoder
# ---------------------------------------------------------------------------

class Stage2DenseDecoder(nn.Module):
    """Complete Stage-2 Dense Prediction Decoder.
    
    Implements the full 5-phase architecture:
    1. Spatial Token Extraction
    2. Geometric Reconstruction
    3. Feature Pyramid Network
    4. Three-Class Segmentation
    5. Physics Validation (via separate validator)
    
    Example usage:
        # Initialize
        decoder = Stage2DenseDecoder(config=Stage2Config())
        validator = PhysicsValidator()
        
        # Forward pass (Phases 1-4)
        with torch.no_grad():
            # Get DINOv3 features
            features = dino_backbone.forward_features(image)
            spatial_tokens = decoder.token_extractor.extract_spatial_tokens(features)
            
            # Process through decoder
            logits = decoder(spatial_tokens)
            masks = torch.softmax(logits, dim=1)
        
        # Phase 5: Physics validation
        shadow_mask = masks[:, 1, :, :].cpu().numpy()  # True Void class
        edge_mask = masks[:, 2, :, :].cpu().numpy()    # Obstacle class
        
        result = validator.validate_detection(
            shadow_mask=shadow_mask[0],
            edge_mask=edge_mask[0],
            sub_solar_azimuth=metadata.sub_solar_azimuth,
            incidence_angle_deg=metadata.incidence_angle,
        )
    """
    
    def __init__(self, 
                 config: Stage2Config = DEFAULT_CONFIG,
                 device: Optional[str] = None):
        super().__init__()
        
        self.config = config
        self.device = device or (
            "cuda" if torch.cuda.is_available() else 
            "mps" if torch.backends.mps.is_available() else 
            "cpu"
        )
        
        # Phase 1: Token Extractor (stateless, just for interface clarity)
        self.token_extractor = SpatialTokenExtractor(
            patch_size=config.patch_size,
            hidden_dim=config.dino_hidden_dim,
        )
        
        # Phase 2: Geometric Reconstructor
        grid_size = self.token_extractor.get_spatial_grid_size(config.input_image_size)
        self.reconstructor = GeometricReconstructor(
            input_dim=config.dino_hidden_dim,
            output_dim=config.projection_dim,
            grid_size=grid_size,
        )
        
        # Phase 3: Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            start_size=grid_size,
            target_size=(config.input_image_size, config.input_image_size),
            start_channels=config.projection_dim,
            fpn_dims=config.fpn_dims,
        )
        
        # Phase 4: Classification Head
        final_channels = config.fpn_dims[-1] if config.fpn_dims else config.projection_dim // 4
        self.classification_head = ThreeClassSegmentationHead(
            in_channels=final_channels,
            num_classes=config.num_classes,
        )
        
        # Initialize all layers
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
        
        # Phase 5: Physics Validator (separate, non-learnable)
        self.physics_validator = PhysicsValidator(
            pixel_scale_meters=0.5,  # Native Stage-2: 0.5 m/pixel
            min_depth_to_width_ratio=config.min_depth_to_width_ratio,
        )
        
        log.info(f"Stage2DenseDecoder initialized on {self.device}")
        log.info(f"  Input: {config.input_image_size}x{config.input_image_size} @ {config.dino_hidden_dim} dims")
        log.info(f"  Grid: {grid_size[0]}x{grid_size[1]} patches")
        log.info(f"  FPN: {config.fpn_dims} channels")
        log.info(f"  Output: {config.num_classes} classes at native resolution")
    
    def _initialize_weights(self):
        """Initialize weights for all learnable layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass through Phases 2-4.
        
        Args:
            spatial_tokens: Spatial patch tokens from DINOv3 (B, num_patches, hidden_dim)
        
        Returns:
            Class logits: (B, num_classes, 256, 256)
        """
        # Phase 2: Unflatten and project
        features_2d = self.reconstructor(spatial_tokens)
        
        # Phase 3: Progressive upsampling
        features_upsampled = self.fpn(features_2d)
        
        # Phase 4: Classification
        logits = self.classification_head(features_upsampled)
        
        return logits
    
    def get_probability_masks(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """Get softmax probability masks for all classes.
        
        Args:
            spatial_tokens: Spatial patch tokens from DINOv3
        
        Returns:
            Probability masks: (B, num_classes, 256, 256)
        """
        logits = self.forward(spatial_tokens)
        return torch.softmax(logits, dim=1)
    
    def validate_with_physics(self, 
                             spatial_tokens: torch.Tensor,
                             sub_solar_azimuth: float,
                             incidence_angle_deg: float,
                             pixel_scale: float = 0.5) -> list:
        """Full pipeline: predict masks and validate with physics.
        
        Args:
            spatial_tokens: Spatial patch tokens (B, num_patches, hidden_dim)
            sub_solar_azimuth: Solar azimuth angle (degrees)
            incidence_angle_deg: Solar incidence angle (degrees)
            pixel_scale: Meters per pixel
        
        Returns:
            List of PhysicsValidationResult for each sample in batch
        """
        with torch.no_grad():
            prob_masks = self.get_probability_masks(spatial_tokens)
        
        results = []
        for i in range(prob_masks.shape[0]):
            # Extract masks (threshold at 0.5 for binary)
            regolith_mask = prob_masks[i, 0, :, :].cpu().numpy()
            shadow_mask = prob_masks[i, 1, :, :].cpu().numpy() > 0.5
            edge_mask = prob_masks[i, 2, :, :].cpu().numpy() > 0.5
            
            result = self.physics_validator.validate_detection(
                shadow_mask=shadow_mask,
                edge_mask=edge_mask,
                sub_solar_azimuth=sub_solar_azimuth,
                incidence_angle_deg=incidence_angle_deg,
                pixel_scale=pixel_scale,
            )
            
            # Add class probabilities to result
            shadow_conf = float(prob_masks[i, 1, :, :].max().cpu().numpy())
            edge_conf = float(prob_masks[i, 2, :, :].max().cpu().numpy())
            result.class_confidence = max(result.class_confidence, shadow_conf * edge_conf)
            
            results.append(result)
        
        return results
    
    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'device': self.device,
        }
        
        torch.save(checkpoint, path)
        log.info(f"Stage2DenseDecoder checkpoint saved to {path}")
    
    @classmethod
    def from_checkpoint(cls, path: str | Path, device: Optional[str] = None) -> 'Stage2DenseDecoder':
        """Load from checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location='cpu')
        
        config = checkpoint.get('config', DEFAULT_CONFIG)
        decoder = cls(config=config, device=device)
        decoder.load_state_dict(checkpoint['state_dict'])
        
        log.info(f"Stage2DenseDecoder loaded from {path}")
        return decoder
    
    def get_parameter_count(self) -> int:
        """Get total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Refiner Integration (for Luna Pipeline)
# ---------------------------------------------------------------------------

class Stage2Refiner:
    """Refiner class that integrates Stage-2 Decoder with Luna Pipeline.
    
    This class provides the same interface as ESSARefiner and DINORefiner
    for seamless integration into the existing pipeline.
    """
    
    CLASSES = {0: "regolith", 1: "pit", 2: "boulder"}
    
    def __init__(self, 
                 decoder: Stage2DenseDecoder,
                 dino_encoder: Optional[any] = None):
        self.decoder = decoder
        self.dino_encoder = dino_encoder
        self.device = decoder.device
    
    @classmethod
    def from_checkpoint(cls, 
                       checkpoint_path: str | Path,
                       dino_encoder: any = None,
                       device: Optional[str] = None) -> 'Stage2Refiner':
        """Load Stage2Refiner from decoder checkpoint."""
        decoder = Stage2DenseDecoder.from_checkpoint(checkpoint_path, device=device)
        return cls(decoder=decoder, dino_encoder=dino_encoder)
    
    def refine(self, 
              hits: list,
              out_dir: Optional[str | Path] = None,
              score_thr: float = 0.5,
              save_debug_plots: bool = False,
              skip_preprocess: bool = False,
              trace: Optional[dict] = None,
              **kwargs) -> list:
        """Refine candidate hits using Stage-2 dense prediction.
        
        Args:
            hits: List of CandidateHit objects from Stage-1
            out_dir: Optional directory to save outputs
            score_thr: Minimum confidence score threshold
            save_debug_plots: Whether to save visualization
            skip_preprocess: Skip preprocessing (not applicable for Stage-2)
            trace: Optional trace dictionary for metrics
            **kwargs: Additional arguments
        
        Returns:
            List of RefinedHit objects
        """
        import time
        from luna.models.essa import RefinedHit
        from luna.io.nac_reader import read_nac
        from pathlib import Path
        
        refined_hits = []
        device = self.decoder.device
        
        # For each hit, load the NAC image, extract the tile, and run Stage-2
        for hit in hits:
            try:
                # Load NAC image
                nac_path = Path(kwargs.get('nac_path', ''))
                if not nac_path.exists():
                    # Try to construct path from product_id
                    from luna.config import SCRATCH_DIR
                    nac_path = SCRATCH_DIR / f"{hit.product_id}.IMG"
                
                if not nac_path.exists():
                    log.warning(f"NAC file not found for {hit.product_id}, skipping")
                    continue
                
                # Read NAC image
                nac_img = read_nac(nac_path)
                image_data = nac_img.pixels.astype(np.float32)
                
                # Get metadata for physics validation
                sub_solar_azimuth = getattr(nac_img, 'sub_solar_azimuth', 180.0)
                incidence_angle = getattr(nac_img, 'incidence_angle', 45.0)
                pixel_scale = getattr(nac_img, 'pixel_scale', 0.5)
                
                # Normalize image
                lo, hi = image_data.min(), image_data.max()
                image_norm = (image_data - lo) / (hi - lo + 1e-6)
                
                # Convert to tensor and add batch dimension
                image_tensor = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).to(device)
                
                # Repeat grayscale to 3 channels for DINOv3
                if image_tensor.shape[1] == 1:
                    image_tensor = image_tensor.repeat(1, 3, 1, 1)
                
                # Get DINOv3 features if encoder is available
                if self.dino_encoder is not None:
                    with torch.no_grad():
                        # Normalize for DINOv3
                        from torchvision import transforms
                        normalize = transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        )
                        image_normalized = normalize(image_tensor)
                        
                        # Get forward features
                        features = self.dino_encoder._backbone_module.forward_features(image_normalized)
                        spatial_tokens = self.decoder.token_extractor.extract_spatial_tokens(features)
                else:
                    # For demo purposes, create dummy tokens
                    batch_size = 1
                    num_patches = 256  # 16x16 for 256x256 image
                    spatial_tokens = torch.randn(batch_size, num_patches, 1024).to(device)
                
                # Run Stage-2 decoder
                with torch.no_grad():
                    prob_masks = self.decoder.get_probability_masks(spatial_tokens)
                
                # Convert to numpy
                prob_masks_np = prob_masks.cpu().numpy()[0]  # Remove batch, keep (C, H, W)
                
                # Get class predictions
                class_preds = np.argmax(prob_masks_np, axis=0)
                shadow_mask = prob_masks_np[1, :, :]  # True Void
                edge_mask = prob_masks_np[2, :, :]    # Obstacle
                
                # Calculate scores
                shadow_score = float(shadow_mask.max())
                edge_score = float(edge_mask.max())
                combined_score = (shadow_score + edge_score) / 2.0
                
                # Physics validation
                shadow_binary = shadow_mask > 0.5
                edge_binary = edge_mask > 0.5
                
                physics_result = self.decoder.physics_validator.validate_detection(
                    shadow_mask=shadow_binary,
                    edge_mask=edge_binary,
                    sub_solar_azimuth=sub_solar_azimuth,
                    incidence_angle_deg=incidence_angle,
                    pixel_scale=pixel_scale,
                )
                
                # Only keep if it passes validation
                if combined_score >= score_thr and physics_result.is_valid_pit:
                    refined_hit = RefinedHit(
                        rank=hit.rank,
                        product_id=hit.product_id,
                        votes=hit.votes,
                        dino_score=hit.score,
                        lon=hit.lon,
                        lat=hit.lat,
                        x_offset=hit.x_offset,
                        y_offset=hit.y_offset,
                        essa_score=combined_score,  # Reusing field for Stage-2 score
                        essa_class="pit" if physics_result.is_valid_pit else "boulder",
                        essa_lon=hit.lon,
                        essa_lat=hit.lat,
                        dino_similarity=combined_score,
                    )
                    refined_hits.append(refined_hit)
                    
                    log.info(f"  Stage-2 CONFIRMED pit in {hit.product_id}: "
                           f"score={combined_score:.3f}, depth={physics_result.depth_estimate_meters:.1f}m, "
                           f"alignment_error={physics_result.alignment_error_degrees:.1f}°")
                else:
                    log.info(f"  Stage-2 REJECTED {hit.product_id}: "
                           f"score={combined_score:.3f}, "
                           f"boulder={physics_result.is_boulder}, "
                           f"depth/width={physics_result.depth_estimate_meters / max(physics_result.width_pixels * pixel_scale, 1):.3f}")
                
                # Save debug plots if requested
                if save_debug_plots and out_dir:
                    self._save_debug_plots(
                        out_dir=Path(out_dir),
                        product_id=hit.product_id,
                        image_data=image_data,
                        class_preds=class_preds,
                        prob_masks=prob_masks_np,
                        physics_result=physics_result,
                    )
                    
            except Exception as e:
                log.error(f"Stage-2 refinement failed for {hit.product_id}: {e}")
                continue
        
        return refined_hits
    
    def _save_debug_plots(self, 
                         out_dir: Path,
                         product_id: str,
                         image_data: np.ndarray,
                         class_preds: np.ndarray,
                         prob_masks: np.ndarray,
                         physics_result: PhysicsValidationResult) -> None:
        """Save visualization of Stage-2 results."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Normalize image for display
        img_display = image_data
        if img_display.dtype != np.uint8:
            img_display = ((img_display - img_display.min()) / 
                          (img_display.max() - img_display.min() + 1e-6) * 255).astype(np.uint8)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(img_display, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Class predictions
        cmap = plt.cm.get_cmap('viridis', 3)
        axes[0, 1].imshow(class_preds, cmap=cmap, vmin=0, vmax=2)
        axes[0, 1].set_title('Class Predictions\n(0=Regolith, 1=Pit, 2=Boulder)')
        axes[0, 1].axis('off')
        
        # Shadow probability
        axes[0, 2].imshow(prob_masks[1, :, :], cmap='Blues', vmin=0, vmax=1)
        axes[0, 2].set_title(f'True Void (Pit) Probability\nMax: {prob_masks[1].max():.3f}')
        axes[0, 2].axis('off')
        
        # Edge probability
        axes[1, 0].imshow(prob_masks[2, :, :], cmap='Reds', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Obstacle Probability\nMax: {prob_masks[2].max():.3f}')
        axes[1, 0].axis('off')
        
        # Physics info
        info_text = (
            f"Physics Validation:\n"
            f"  Valid Pit: {physics_result.is_valid_pit}\n"
            f"  Boulder: {physics_result.is_boulder}\n"
            f"  Depth: {physics_result.depth_estimate_meters:.2f} m\n"
            f"  Shadow Vector: {physics_result.shadow_vector_degrees:.1f}°\n"
            f"  Solar Vector: {physics_result.solar_vector_degrees:.1f}°\n"
            f"  Alignment Error: {physics_result.alignment_error_degrees:.1f}°\n"
            f"  Depth/Width Ratio: {physics_result.depth_estimate_meters / max(physics_result.width_pixels * 0.5, 1):.3f}"
        )
        axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes, 
                       va='center', ha='left', fontsize=10)
        axes[1, 1].set_title('Physics Validation')
        axes[1, 1].axis('off')
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(out_dir / f"stage2_{product_id}.png", dpi=150)
        plt.close()
        
        log.info(f"  Debug plot saved to {out_dir / f'stage2_{product_id}.png'}")


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def build_stage2_decoder(config: Optional[Stage2Config] = None, 
                         checkpoint_path: Optional[str | Path] = None,
                         device: Optional[str] = None) -> Stage2DenseDecoder:
    """Build or load a Stage-2 Dense Decoder."""
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        return Stage2DenseDecoder.from_checkpoint(checkpoint_path, device=device)
    else:
        cfg = config or DEFAULT_CONFIG
        return Stage2DenseDecoder(config=cfg, device=device)


def build_stage2_refiner(checkpoint_path: Optional[str | Path] = None,
                         dino_encoder: Optional[any] = None,
                         device: Optional[str] = None) -> Stage2Refiner:
    """Build or load a Stage-2 Refiner."""
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        return Stage2Refiner.from_checkpoint(checkpoint_path, dino_encoder, device=device)
    else:
        decoder = build_stage2_decoder(device=device)
        return Stage2Refiner(decoder=decoder, dino_encoder=dino_encoder)
