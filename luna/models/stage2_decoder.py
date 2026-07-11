"""Stage-2 Dense Prediction Decoder for LUNA.

Operates directly on DINOv3 spatial patch tokens in unified memory.
Implements dense semantic mask prediction and physics-based validation.
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

LUNAR_METERS_PER_DEGREE = 30323.35

def compute_lunar_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes surface distance between two points using localized flat approximation."""
    dlat = math.radians(lat1 - lat2)
    delta_lon = (lon1 % 360) - (lon2 % 360)
    if delta_lon > 180: delta_lon -= 360
    elif delta_lon < -180: delta_lon += 360
        
    dlon = math.radians(delta_lon)
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    dy = LUNAR_METERS_PER_DEGREE * 57.2957795 * dlat # Convert degrees back for distance
    dx = LUNAR_METERS_PER_DEGREE * 57.2957795 * dlon * math.cos(mean_lat)
    return math.sqrt(dx**2 + dy**2)


@dataclass
class Stage2Config:
    """Configuration for Stage-2 Dense Prediction Decoder."""
    input_image_size: int = 256
    patch_size: int = 16
    dino_hidden_dim: int = 1024
    projection_dim: int = 256
    fpn_dims: list = None
    num_classes: int = 3
    min_depth_to_width_ratio: float = 0.10
    border_margin: int = 4
    min_pit_area_px: int = 100

    def __post_init__(self):
        if self.fpn_dims is None:
            self.fpn_dims = [
                self.projection_dim,
                self.projection_dim // 2,
                self.projection_dim // 4,
                self.projection_dim // 8,
            ]


DEFAULT_CONFIG = Stage2Config()


class SpatialTokenExtractor:
    """Extracts spatial patch tokens from DINOv3 feature maps."""
    def __init__(self, patch_size: int = 16, hidden_dim: int = 1024):
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
    
    def extract_spatial_tokens(self, features: dict) -> torch.Tensor:
        if 'x_norm_patchtokens' in features:
            spatial_tokens = features['x_norm_patchtokens']
        elif 'x_prenorm' in features:
            spatial_tokens = features['x_prenorm'][:, 1:, :]
        elif 'x' in features:
            spatial_tokens = features['x'][:, 1:, :]
        else:
            raise KeyError(f"Spatial tokens not found. Available keys: {list(features.keys())}")
        return spatial_tokens
    
    def get_spatial_grid_size(self, image_size: int) -> Tuple[int, int]:
        h = w = image_size // self.patch_size
        return h, w


class GeometricReconstructor(nn.Module):
    """Reshapes flat token sequences back to 2D spatial feature maps."""
    def __init__(self, input_dim: int = 1024, output_dim: int = 256, grid_size: Tuple[int, int] = (16, 16)):
        super().__init__()
        self.grid_size = grid_size
        self.projection = nn.Linear(input_dim, output_dim)
        nn.init.kaiming_normal_(self.projection.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, input_dim = spatial_tokens.shape
        
        if input_dim != self.projection.in_features:
            new_proj = nn.Linear(input_dim, self.projection.out_features).to(spatial_tokens.device)
            nn.init.kaiming_normal_(new_proj.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(new_proj.bias)
            self.projection = new_proj
            # Force FP32 on MPS to avoid dtype mismatch with FP16 biases
            self.projection = self.projection.float()
        
        # Ensure input is FP32 for projection layer
        if spatial_tokens.dtype != self.projection.weight.dtype:
            spatial_tokens = spatial_tokens.to(dtype=self.projection.weight.dtype)
            
        projected = self.projection(spatial_tokens)
        grid_h, grid_w = self.grid_size
        return projected.permute(0, 2, 1).reshape(batch_size, -1, grid_h, grid_w)


class FPNBlock(nn.Module):
    """Upsampling block with optional cross-layer fusion."""
    def __init__(self, in_channels: int, out_channels: int, upscale_factor: int = 2, use_skip: bool = False, skip_channels: Optional[int] = None):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.use_skip = use_skip and skip_channels is not None
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
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
        x = self.upsample(x)
        if self.use_skip and skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            skip = self.skip_conv(skip)
            x = torch.cat([x, skip], dim=1)
            x = self.fuse(x)
        return x


class FeaturePyramidNetwork(nn.Module):
    """Resolution restoration cascade for dense feature generation."""
    def __init__(self, start_size: Tuple[int, int] = (16, 16), target_size: Tuple[int, int] = (256, 256), start_channels: int = 256, fpn_dims: list = None):
        super().__init__()
        if fpn_dims is None:
            fpn_dims = [256, 128, 64]
        
        self.start_size = start_size
        self.target_size = target_size
        self.blocks = nn.ModuleList()
        
        while len(fpn_dims) < 4:
            fpn_dims.append(fpn_dims[-1] // 2 if fpn_dims else start_channels // 4)
        
        self.blocks.append(FPNBlock(start_channels, fpn_dims[0], upscale_factor=2))
        self.blocks.append(FPNBlock(fpn_dims[0], fpn_dims[1], upscale_factor=2))
        self.blocks.append(FPNBlock(fpn_dims[1], fpn_dims[2], upscale_factor=2))
        self.blocks.append(FPNBlock(fpn_dims[2], fpn_dims[3], upscale_factor=2))
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(fpn_dims[-1], fpn_dims[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_dims[-1]),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_conv(x)


class ThreeClassSegmentationHead(nn.Module):
    """Maps feature maps to pixel-level classifications."""
    def __init__(self, in_channels: int = 64, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


@dataclass
class PhysicsValidationResult:
    """Contains downstream physics validation metrics."""
    is_valid_pit: bool
    is_boulder: bool
    depth_estimate_meters: float
    shadow_vector_degrees: float
    solar_vector_degrees: float
    alignment_error_degrees: float
    width_pixels: int
    height_pixels: int
    class_confidence: float
    tile_centroid_x: float = 128.0
    tile_centroid_y: float = 128.0
    lon: Optional[float] = None
    lat: Optional[float] = None


class PhysicsValidator:
    """Validates structural shadows using solar incidence geometry."""
    def __init__(self, pixel_scale_meters: float = 0.5, min_depth_to_width_ratio: float = 0.1, border_margin: int = 4, min_pit_area_px: int = 100):
        self.pixel_scale = pixel_scale_meters
        self.min_depth_ratio = min_depth_to_width_ratio
        self.border_margin = border_margin
        self.min_pit_area_px = min_pit_area_px
    
    def validate_detection(self, shadow_mask: np.ndarray, edge_mask: np.ndarray, sub_solar_azimuth: float, incidence_angle_deg: float, pixel_scale: float = 0.5, lon: Optional[float] = None, lat: Optional[float] = None) -> PhysicsValidationResult:
        incidence_rad = np.radians(incidence_angle_deg)
        
        if self.border_margin > 0:
            shadow_mask = shadow_mask.copy()
            edge_mask = edge_mask.copy()
            shadow_mask[:self.border_margin, :] = 0
            shadow_mask[-self.border_margin:, :] = 0
            shadow_mask[:, :self.border_margin] = 0
            shadow_mask[:, -self.border_margin:] = 0
            edge_mask[:self.border_margin, :] = 0
            edge_mask[-self.border_margin:, :] = 0
            edge_mask[:, :self.border_margin] = 0
            edge_mask[:, -self.border_margin:] = 0

        shadow_y, shadow_x = np.where(shadow_mask > 0)
        edge_y, edge_x = np.where(edge_mask > 0)
        
        # Check minimum pit area
        if shadow_mask.sum() < self.min_pit_area_px:
            return PhysicsValidationResult(
                is_valid_pit=False, is_boulder=True, depth_estimate_meters=0.0,
                shadow_vector_degrees=0.0, solar_vector_degrees=sub_solar_azimuth,
                alignment_error_degrees=180.0, width_pixels=0, height_pixels=0,
                class_confidence=0.0, lon=lon, lat=lat
            )
        
        if len(shadow_x) == 0 or len(edge_x) == 0:
            return PhysicsValidationResult(
                is_valid_pit=False, is_boulder=False, depth_estimate_meters=0.0,
                shadow_vector_degrees=0.0, solar_vector_degrees=sub_solar_azimuth,
                alignment_error_degrees=180.0, width_pixels=0, height_pixels=0,
                class_confidence=0.0, lon=lon, lat=lat
            )
        
        shadow_cx, shadow_cy = np.mean(shadow_x), np.mean(shadow_y)
        edge_cx, edge_cy = np.mean(edge_x), np.mean(edge_y)
        
        vec_x, vec_y = shadow_cx - edge_cx, shadow_cy - edge_cy
        shadow_vector_deg = np.degrees(np.arctan2(vec_y, vec_x))
        
        # The shadow-to-edge vector for a depression (pit) points away from the sun (azimuth + 180.0)
        expected_vector_deg = (sub_solar_azimuth + 180.0 - 90.0) % 360
        if expected_vector_deg > 180:
            expected_vector_deg -= 360
            
        solar_vector_deg = expected_vector_deg
        alignment_error = abs(shadow_vector_deg - solar_vector_deg) % 360
        alignment_error = min(alignment_error, 360 - alignment_error)
        
        width_px = int(np.max(shadow_x) - np.min(shadow_x))
        height_px = int(np.max(shadow_y) - np.min(shadow_y))
        
        # Project shadow pixels onto the shadow direction vector to get the actual shadow length
        shadow_rad = np.radians(expected_vector_deg)
        cos_dir = np.cos(shadow_rad)
        sin_dir = np.sin(shadow_rad)
        proj_shadow = shadow_x * cos_dir + shadow_y * sin_dir
        shadow_length_px = np.max(proj_shadow) - np.min(proj_shadow) if len(proj_shadow) > 0 else 0
        
        shadow_length_m = shadow_length_px * pixel_scale
        tan_incidence = max(np.tan(incidence_rad), 0.01)
        depth_m = shadow_length_m / tan_incidence
        width_m = max(width_px, height_px) * pixel_scale
        depth_width_ratio = depth_m / width_m if width_m > 0 else 0
        
        is_aligned = alignment_error < 45
        is_deep_enough = depth_width_ratio >= self.min_depth_ratio
        is_valid_pit = is_aligned and is_deep_enough
        
        confidence = 0.0
        if is_valid_pit:
            confidence = (1.0 - alignment_error / 45.0) * 0.7 + (depth_width_ratio / 1.0) * 0.3
            confidence = min(confidence, 1.0)
            
        return PhysicsValidationResult(
            is_valid_pit=is_valid_pit, is_boulder=not is_valid_pit,
            depth_estimate_meters=float(depth_m), shadow_vector_degrees=float(shadow_vector_deg),
            solar_vector_degrees=float(solar_vector_deg), alignment_error_degrees=float(alignment_error),
            width_pixels=width_px, height_pixels=height_px, class_confidence=float(confidence),
            tile_centroid_x=float(shadow_cx), tile_centroid_y=float(shadow_cy), lon=lon, lat=lat
        )


class Stage2DenseDecoder(nn.Module):
    """Complete multi-phase dense reasoning framework."""
    def __init__(self, config: Stage2Config = DEFAULT_CONFIG, device: Optional[str] = None):
        super().__init__()
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.token_extractor = SpatialTokenExtractor(patch_size=config.patch_size, hidden_dim=config.dino_hidden_dim)
        grid_size = self.token_extractor.get_spatial_grid_size(config.input_image_size)
        
        self.reconstructor = GeometricReconstructor(input_dim=config.dino_hidden_dim, output_dim=config.projection_dim, grid_size=grid_size)
        self.fpn = FeaturePyramidNetwork(start_size=grid_size, target_size=(config.input_image_size, config.input_image_size), start_channels=config.projection_dim, fpn_dims=config.fpn_dims)
        
        final_channels = config.fpn_dims[-1] if config.fpn_dims else config.projection_dim // 4
        self.classification_head = ThreeClassSegmentationHead(in_channels=final_channels, num_classes=config.num_classes)
        
        # Force FP32 for all layers on MPS to avoid FP16 bias issues
        if self.device == "mps":
            self.reconstructor = self.reconstructor.float()
            self.fpn = self.fpn.float()
            self.classification_head = self.classification_head.float()
        
        self._initialize_weights()
        self.to(self.device)
        if self.device == "cuda":
            self.half()
            
        # Ensure all layers are FP32 on MPS (FP16 not fully supported on MPS for some ops)
        if self.device == "mps":
            self.float()
            # Also ensure all submodules are FP32
            for module in self.modules():
                if hasattr(module, 'float'):
                    module.float()
        
        self.physics_validator = PhysicsValidator(pixel_scale_meters=0.5, min_depth_to_width_ratio=config.min_depth_to_width_ratio, border_margin=config.border_margin, min_pit_area_px=config.min_pit_area_px)
        
        log.info(f"Stage2DenseDecoder stacked successfully on {self.device}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        features_2d = self.reconstructor(spatial_tokens)
        features_upsampled = self.fpn(features_2d)
        return self.classification_head(features_upsampled)

    def get_probability_masks(self, spatial_tokens: torch.Tensor, temperature: float = 0.7) -> torch.Tensor:
        """Generates class probabilities from transformer tokens using temperature scaling."""
        logits = self.forward(spatial_tokens)
        scaled_logits = logits / temperature
        return torch.softmax(scaled_logits, dim=1)

    def validate_with_physics(self, spatial_tokens: torch.Tensor, sub_solar_azimuth: float, incidence_angle_deg: float, pixel_scale: float = 0.5, batch_coords: Optional[list[dict]] = None) -> list:
        with torch.no_grad():
            prob_masks = self.get_probability_masks(spatial_tokens)
        
        results = []
        for i in range(prob_masks.shape[0]):
            shadow_mask = prob_masks[i, 1, :, :].cpu().numpy() > 0.5
            edge_mask = prob_masks[i, 2, :, :].cpu().numpy() > 0.5
            
            coords = batch_coords[i] if batch_coords else {}
            result = self.physics_validator.validate_detection(
                shadow_mask, edge_mask, sub_solar_azimuth, incidence_angle_deg,
                pixel_scale, lon=coords.get("lon"), lat=coords.get("lat")
            )
            
            if shadow_mask.sum() == 0 or edge_mask.sum() == 0:
                result.class_confidence = 0.0
            else:
                result.class_confidence = max(float(prob_masks[i, 1].max().cpu().numpy()), float(prob_masks[i, 2].max().cpu().numpy()))
            results.append(result)
        return results

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'config': self.config, 'state_dict': self.state_dict(), 'device': self.device}, path)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: Optional[str] = None) -> 'Stage2DenseDecoder':
        path = Path(path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint.get('config', DEFAULT_CONFIG)
        if 'state_dict' in checkpoint and 'reconstructor.projection.weight' in checkpoint['state_dict']:
            config.dino_hidden_dim = checkpoint['state_dict']['reconstructor.projection.weight'].shape[1]
        decoder = cls(config=config, device=device)
        decoder.load_state_dict(checkpoint['state_dict'])
        return decoder

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Stage2Refiner:
    """Refiner linking Stage-1 candidates to full Stage-2 dense verification."""
    CLASSES = {0: "regolith", 1: "pit", 2: "boulder"}

    def __init__(self, decoder: Stage2DenseDecoder, dino_encoder: Optional[any] = None):
        self.decoder = decoder
        self.dino_encoder = dino_encoder

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, dino_encoder: any = None, device: Optional[str] = None) -> 'Stage2Refiner':
        decoder = Stage2DenseDecoder.from_checkpoint(checkpoint_path, device=device)
        return cls(decoder=decoder, dino_encoder=dino_encoder)

    def refine(self, hits: list, out_dir: Optional[str | Path] = None, score_thr: float = 0.5, save_debug_plots: bool = False, skip_preprocess: bool = False, trace: Optional[dict] = None, **kwargs) -> list:
        from luna.models.essa import RefinedHit
        from luna.io.nac_reader import read_nac
        from luna.io.projection import pixel_to_lonlat
        import pickle
        from luna.config import SCRATCH_DIR
        
        # Load consolidated SVM gatekeeper if trained
        svm_model = None
        svm_scaler = None
        try:
            svm_path = SCRATCH_DIR / "stage2_svm.pkl"
            if svm_path.exists():
                with open(svm_path, "rb") as f:
                    svm_data = pickle.load(f)
                    svm_scaler = svm_data["scaler"]
                    svm_model = svm_data["model"]
                log.info("Loaded consolidated SVM gatekeeper from stage2_svm.pkl")
        except Exception as e:
            log.warning(f"Could not load Stage-2 SVM classifier: {e}")
        
        refined_hits = []
        device = self.decoder.device
        nac_cache = {}

        for hit in hits:
            try:
                if hit.product_id not in nac_cache:
                    nac_path_val = kwargs.get('nac_path', '')
                    nac_path = Path(nac_path_val) if nac_path_val else None
                    if nac_path is None or not nac_path.exists() or nac_path.is_dir():
                        from luna.config import SCRATCH_DIR
                        nac_path = SCRATCH_DIR / f"{hit.product_id}.IMG"
                    
                    if not nac_path.exists(): continue
                    
                    from luna.io.projection import LinearProjection
                    nac_img = read_nac(nac_path, geometry=True)
                    try:
                        proj = LinearProjection.from_nac_geometry(
                            nac_img.geometry, samples=nac_img.samples, lines=nac_img.lines
                        )
                    except Exception:
                        proj = None
                    nac_cache[hit.product_id] = (
                        nac_img, nac_img.pixels.shape[0], nac_img.pixels.shape[1],
                        getattr(nac_img, 'sub_solar_azimuth', 180.0),
                        getattr(nac_img, 'incidence_angle', 45.0),
                        getattr(nac_img, 'pixel_scale', 0.5),
                        proj
                    )
                
                nac_img, h, w, sub_solar_azimuth, incidence_angle, pixel_scale, proj = nac_cache[hit.product_id]
                
                x0, y0 = int(hit.x_offset), int(hit.y_offset)
                x0, y0 = max(0, min(w - 1, x0)), max(0, min(h - 1, y0))
                tile = nac_img.pixels[y0:min(h, y0+256), x0:min(w, x0+256)].copy().astype(np.float32)
                
                if tile.shape[0] < 256 or tile.shape[1] < 256:
                    tile = np.pad(tile, ((0, max(0, 256 - tile.shape[0])), (0, max(0, 256 - tile.shape[1]))), mode='reflect')
                
                tile_norm = (tile - tile.min()) / (tile.max() - tile.min() + 1e-6)
                image_tensor = torch.from_numpy(tile_norm).unsqueeze(0).unsqueeze(0).to(device)
                if device == "cuda": image_tensor = image_tensor.half()
                if image_tensor.shape[1] == 1: image_tensor = image_tensor.repeat(1, 3, 1, 1)
                
                if self.dino_encoder is not None:
                    with torch.no_grad():
                        from torchvision import transforms
                        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        features = self.dino_encoder._backbone_module.forward_features(normalize(image_tensor))
                        spatial_tokens = self.decoder.token_extractor.extract_spatial_tokens(features)
                        # Force FP32 on MPS - DINO encoder outputs FP16, decoder expects FP32
                        if device == "mps":
                            spatial_tokens = spatial_tokens.float()
                        else:
                            # Match decoder dtype
                            decoder_dtype = next(self.decoder.parameters()).dtype
                            spatial_tokens = spatial_tokens.to(dtype=decoder_dtype)
                else:
                    spatial_tokens = torch.randn(1, 256, 1024).to(device)
                
                with torch.no_grad():
                    prob_masks = self.decoder.get_probability_masks(spatial_tokens, temperature=0.5)
                
                prob_masks_np = prob_masks.cpu().numpy()[0]
                class_preds = np.argmax(prob_masks_np, axis=0)
                shadow_mask, edge_mask = prob_masks_np[1, :, :], prob_masks_np[2, :, :]
                
                # Apply morphological smoothing to reduce noise
                from skimage.morphology import opening, closing, disk
                shadow_mask = opening(shadow_mask, disk(3))
                edge_mask = closing(edge_mask, disk(2))
                
                if shadow_mask.sum() == 0 or edge_mask.sum() == 0:
                    combined_score = 0.0
                else:
                    combined_score = float((shadow_mask.max() + edge_mask.max()) / 2.0)
                
                physics_result = self.decoder.physics_validator.validate_detection(
                    shadow_mask > 0.5, edge_mask > 0.5, sub_solar_azimuth, incidence_angle, pixel_scale, hit.lon, hit.lat
                )
                
                svm_passed = True
                if svm_model is not None and svm_scaler is not None:
                    try:
                        feats = self._extract_svm_features(hit, shadow_mask, edge_mask, physics_result, pixel_scale)
                        feats_scaled = svm_scaler.transform(feats.reshape(1, -1))
                        decision_score = svm_model.decision_function(feats_scaled)[0]

                        log.info(f"  SVM Gatekeeper decision for {hit.product_id}: decision_score={decision_score:.4f}")

                        if decision_score < 0.0: 
                            svm_passed = False
                    except Exception as svm_err:
                        log.warning(f"  SVM prediction failed: {svm_err}")
                
                if combined_score >= score_thr and physics_result.is_valid_pit and svm_passed:
                    pixel_dx = physics_result.tile_centroid_x - 128.0
                    pixel_dy = physics_result.tile_centroid_y - 128.0
                    
                    meter_dx = pixel_dx * pixel_scale
                    meter_dy = -pixel_dy * pixel_scale
                    
                    precise_x = x0 + physics_result.tile_centroid_x
                    precise_y = y0 + physics_result.tile_centroid_y
                    
                    try:
                        # 1. PRIMARY: SPICE (most accurate for lunar coordinates)
                        from luna.io.spice_project import image_to_ground
                        from luna.config import SCRATCH_DIR
                        target_img = SCRATCH_DIR / f"{hit.product_id}.IMG"
                        refined_lon, refined_lat = image_to_ground(str(target_img), precise_x, precise_y)
                        if refined_lon == 0.0 and refined_lat == 0.0:
                            raise ValueError("SPICE returned invalid 0.0, 0.0")
                        log.debug(f"  Coordinate projection: Using SPICE (lon={refined_lon:.6f}, lat={refined_lat:.6f})")
                    except Exception as spice_error:
                        log.warning(f"SPICE failed, falling back to bilinear: {spice_error}")
                        try:
                            # 2. FALLBACK: Bilinear Projection (LOLA DEM-registered)
                            if proj is None:
                                raise ValueError("No bilinear projection available")
                            refined_lon, refined_lat = pixel_to_lonlat(proj, precise_x, precise_y)
                            if math.isnan(refined_lon) or math.isnan(refined_lat):
                                raise ValueError("Bilinear projection returned NaN")
                            log.debug(f"  Coordinate projection: Using Bilinear (lon={refined_lon:.6f}, lat={refined_lat:.6f})")
                        except Exception as proj_error:
                            log.warning(f"Bilinear failed, using manual calculation: {proj_error}")
                            # 3. MANUAL FALLBACK (corrected!)
                            mean_lat = hit.lat
                            delta_lat = meter_dy / LUNAR_METERS_PER_DEGREE
                            delta_lon = meter_dx / (LUNAR_METERS_PER_DEGREE * math.cos(math.radians(mean_lat)))
                            refined_lat = hit.lat + delta_lat
                            refined_lon = hit.lon + delta_lon
                            log.debug(f"  Coordinate projection: Using manual calculation (lon={refined_lon:.6f}, lat={refined_lat:.6f})")
                    
                    refined_hit = RefinedHit(
                        rank=hit.rank, product_id=hit.product_id, votes=hit.votes, dino_score=hit.score,
                        lon=refined_lon, lat=refined_lat, x_offset=hit.x_offset, y_offset=hit.y_offset,
                        essa_score=combined_score, essa_class="pit", essa_lon=refined_lon, essa_lat=refined_lat,
                        dino_similarity=combined_score
                    )
                    refined_hits.append(refined_hit)
                    
                    log.info(f"  Stage-2 CONFIRMED pit in {hit.product_id}: score={combined_score:.3f}, precise_lat={refined_lat:.6f}°")
                else:
                    log.info(f"  Stage-2 REJECTED {hit.product_id}: score={combined_score:.3f}, boulder={physics_result.is_boulder}")
                
                if save_debug_plots and out_dir and physics_result.is_valid_pit:
                    self._save_debug_plots(Path(out_dir), hit.product_id, tile, class_preds, prob_masks_np, physics_result, hit.rank)
                    
            except Exception as e:
                log.error(f"Stage-2 refinement failed for {hit.product_id}: {e}")
                continue
                
        return refined_hits

    def _save_debug_plots(self, out_dir: Path, product_id: str, image_data: np.ndarray, class_preds: np.ndarray, prob_masks: np.ndarray, physics_result: PhysicsValidationResult, rank: int) -> None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        out_dir.mkdir(parents=True, exist_ok=True)
        img_display = ((image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-6) * 255).astype(np.uint8)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(img_display, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        cmap = plt.get_cmap('viridis', 3)
        axes[0, 1].imshow(class_preds, cmap=cmap, vmin=0, vmax=2)
        axes[0, 1].set_title('Class Predictions\n(0=Regolith, 1=Pit, 2=Boulder)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(prob_masks[1, :, :], cmap='Blues', vmin=0, vmax=1)
        axes[0, 2].set_title(f'True Void (Pit) Probability\nMax: {prob_masks[1].max():.3f}')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(prob_masks[2, :, :], cmap='Reds', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Obstacle Probability\nMax: {prob_masks[2].max():.3f}')
        axes[1, 0].axis('off')
        
        info_text = (
            f"Physics Validation:\n  Valid Pit: {physics_result.is_valid_pit}\n  Boulder: {physics_result.is_boulder}\n"
            f"  Depth: {physics_result.depth_estimate_meters:.2f} m\n  Alignment Error: {physics_result.alignment_error_degrees:.1f}°\n"
            f"  Centroid X/Y: {physics_result.tile_centroid_x:.1f}, {physics_result.tile_centroid_y:.1f}"
        )
        axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes, va='center', ha='left', fontsize=10)
        axes[1, 1].set_title('Physics Validation')
        axes[1, 1].axis('off')
        
        fig.delaxes(axes[1, 2])
        plt.tight_layout()
        plt.savefig(out_dir / f"stage2_{product_id}_rank_{rank}.png", dpi=150)
        plt.close()

    def _extract_svm_features(self, hit, shadow_mask: np.ndarray, edge_mask: np.ndarray, physics_result, pixel_scale: float = 0.5) -> np.ndarray:
        """Extracts 12 normalized geometric and neural features for the SVM gatekeeper."""
        dino_score = float(hit.score)
        votes = float(hit.votes)
        shadow_ratio = float(shadow_mask.sum()) / (256.0 * 256.0)
        edge_ratio = float(edge_mask.sum()) / (256.0 * 256.0)
        
        shadow_y, shadow_x = np.where(shadow_mask > 0)
        edge_y, edge_x = np.where(edge_mask > 0)
        
        shadow_cx = float(np.mean(shadow_x)) if len(shadow_x) > 0 else 128.0
        shadow_cy = float(np.mean(shadow_y)) if len(shadow_y) > 0 else 128.0
        edge_cx = float(np.mean(edge_x)) if len(edge_x) > 0 else 128.0
        edge_cy = float(np.mean(edge_y)) if len(edge_y) > 0 else 128.0
        
        centroid_dist = float(np.sqrt((shadow_cx - edge_cx) ** 2 + (shadow_cy - edge_cy) ** 2))
        aspect_ratio = float(physics_result.width_pixels) / float(physics_result.height_pixels + 1e-6)
        alignment_error = float(physics_result.alignment_error_degrees)
        
        # New features
        shadow_rad = np.radians(physics_result.solar_vector_degrees)
        cos_dir = np.cos(shadow_rad)
        sin_dir = np.sin(shadow_rad)
        proj_shadow = shadow_x * cos_dir + shadow_y * sin_dir
        max_shadow_length_px = float(np.max(proj_shadow) - np.min(proj_shadow)) if len(proj_shadow) > 0 else 0.0
        depth_width_ratio = float(physics_result.depth_estimate_meters / max(physics_result.width_pixels * pixel_scale, 1e-6))
        pit_area_px = float(shadow_mask.sum())
        shadow_edge_ratio = float(shadow_mask.sum() / max(edge_mask.sum(), 1))
        
        # Circularity (0=line, 1=perfect circle)
        from skimage.measure import label, regionprops
        labeled = label(shadow_mask > 0.5)
        if len(np.unique(labeled)) > 1:
            largest_region = max(regionprops(labeled), key=lambda r: r.area)
            circularity = 4 * np.pi * largest_region.area / (largest_region.perimeter ** 2 + 1e-6)
        else:
            circularity = 0.0
        
        feats = np.array([
            dino_score, votes, shadow_ratio, edge_ratio,
            centroid_dist, aspect_ratio, alignment_error,
            max_shadow_length_px, depth_width_ratio, pit_area_px,
            shadow_edge_ratio, circularity
        ], dtype=np.float32)
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    def train_svm(self, hits: list, catalog_path: str = "catalogs/lpa.csv") -> dict:
        """Trains the linear SVM classifier on features labeled against the LPA catalog."""
        import csv
        import pickle
        import math
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from luna.io.nac_reader import read_nac
        from luna.io.projection import LinearProjection, pixel_to_lonlat
        
        # 1. Load catalog pits for Aristarchus
        catalog_pits = []
        try:
            with open(catalog_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["host"].strip() == "Aristarchus":
                        raw_lon = float(row["longitude"])
                        if raw_lon > 180.0:
                            raw_lon -= 360.0
                        catalog_pits.append({
                            "name": row["name"].strip(),
                            "lat": float(row["latitude"]),
                            "lon": raw_lon
                        })
            log.info(f"Loaded {len(catalog_pits)} catalog pits for Aristarchus region.")
        except Exception as e:
            log.error(f"Failed to read catalog file {catalog_path}: {e}")
            return {}
        
        X = []
        y = []
        nac_cache = {}
        device = self.decoder.device
        
        log.info(f"Extracting features for {len(hits)} candidates to train SVM...")
        
        for idx, hit in enumerate(hits):
            try:
                # Load frame and cache metadata
                if hit.product_id not in nac_cache:
                    from luna.config import SCRATCH_DIR
                    nac_path = SCRATCH_DIR / f"{hit.product_id}.IMG"
                    if not nac_path.exists():
                        continue
                    nac_img = read_nac(nac_path, geometry=True)
                    try:
                        proj = LinearProjection.from_nac_geometry(
                            nac_img.geometry, samples=nac_img.samples, lines=nac_img.lines
                        )
                    except Exception:
                        proj = None
                    nac_cache[hit.product_id] = (
                        nac_img, nac_img.pixels.shape[0], nac_img.pixels.shape[1],
                        getattr(nac_img, 'sub_solar_azimuth', 180.0),
                        getattr(nac_img, 'incidence_angle', 45.0),
                        getattr(nac_img, 'pixel_scale', 0.5),
                        proj
                    )
                
                nac_img, h, w, sub_solar_azimuth, incidence_angle, pixel_scale, proj = nac_cache[hit.product_id]
                
                x0, y0 = int(hit.x_offset), int(hit.y_offset)
                x0, y0 = max(0, min(w - 1, x0)), max(0, min(h - 1, y0))
                tile = nac_img.pixels[y0:min(h, y0+256), x0:min(w, x0+256)].copy().astype(np.float32)
                if tile.shape[0] < 256 or tile.shape[1] < 256:
                    tile = np.pad(tile, ((0, max(0, 256 - tile.shape[0])), (0, max(0, 256 - tile.shape[1]))), mode='reflect')
                
                tile_norm = (tile - tile.min()) / (tile.max() - tile.min() + 1e-6)
                image_tensor = torch.from_numpy(tile_norm).unsqueeze(0).unsqueeze(0).to(device)
                if device == "cuda": image_tensor = image_tensor.half()
                if image_tensor.shape[1] == 1: image_tensor = image_tensor.repeat(1, 3, 1, 1)
                
                # Get embeddings
                if self.dino_encoder is not None:
                    with torch.no_grad():
                        from torchvision import transforms
                        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        features = self.dino_encoder._backbone_module.forward_features(normalize(image_tensor))
                        spatial_tokens = self.decoder.token_extractor.extract_spatial_tokens(features)
                        # Force FP32 on MPS - DINO encoder outputs FP16, decoder expects FP32
                        if device == "mps":
                            spatial_tokens = spatial_tokens.float()
                        else:
                            # Match decoder dtype
                            decoder_dtype = next(self.decoder.parameters()).dtype
                            spatial_tokens = spatial_tokens.to(dtype=decoder_dtype)
                else:
                    spatial_tokens = torch.randn(1, 256, 1024).to(device)
                
                with torch.no_grad():
                    prob_masks = self.decoder.get_probability_masks(spatial_tokens, temperature=0.5)
                
                prob_masks_np = prob_masks.cpu().numpy()[0]
                shadow_mask = prob_masks_np[1, :, :]
                edge_mask = prob_masks_np[2, :, :]
                
                # Apply morphological smoothing for consistency with refine()
                from skimage.morphology import opening, closing, disk
                shadow_mask = opening(shadow_mask, disk(3))
                edge_mask = closing(edge_mask, disk(2))
                
                physics_result = self.decoder.physics_validator.validate_detection(
                    shadow_mask > 0.5, edge_mask > 0.5, sub_solar_azimuth, incidence_angle, pixel_scale, hit.lon, hit.lat
                )
                
                # Calculate precise coordinates (bilinear or SPICE)
                precise_x = x0 + physics_result.tile_centroid_x
                precise_y = y0 + physics_result.tile_centroid_y
                
                try:
                    if proj is None:
                        raise ValueError("No bilinear projection available")
                    refined_lon, refined_lat = pixel_to_lonlat(proj, precise_x, precise_y)
                    if math.isnan(refined_lon) or math.isnan(refined_lat):
                        raise ValueError("Bilinear projection returned NaN")
                except Exception as proj_error:
                    try:
                        from luna.io.spice_project import image_to_ground
                        from luna.config import SCRATCH_DIR
                        target_img = SCRATCH_DIR / f"{hit.product_id}.IMG"
                        refined_lon, refined_lat = image_to_ground(str(target_img), precise_x, precise_y)
                        if refined_lon == 0.0 and refined_lat == 0.0:
                            raise ValueError("SPICE failure")
                    except Exception as spice_error:
                        # Full mathematical fallback if everything else fails
                        pixel_dx = physics_result.tile_centroid_x - 128.0
                        pixel_dy = physics_result.tile_centroid_y - 128.0
                        meter_dx = pixel_dx * pixel_scale
                        meter_dy = -pixel_dy * pixel_scale
                        delta_lat = meter_dy / LUNAR_METERS_PER_DEGREE
                        delta_lon = meter_dx / (LUNAR_METERS_PER_DEGREE * math.cos(math.radians(hit.lat)))
                        refined_lat = hit.lat + delta_lat
                        refined_lon = hit.lon + delta_lon
                
                if refined_lon > 180.0:
                    refined_lon -= 360.0
                
                # Compute distance to catalog pits
                best_dist = float('inf')
                for pit in catalog_pits:
                    dist = compute_lunar_distance(refined_lat, refined_lon, pit["lat"], pit["lon"])
                    if dist < best_dist:
                        best_dist = dist
                
                # Stricter labeling: only label as positive if within 100m of a catalog pit
                label = 1 if best_dist < 100.0 else 0
                feats = self._extract_svm_features(hit, shadow_mask, edge_mask, physics_result, pixel_scale)
                
                X.append(feats)
                y.append(label)
            except Exception as e:
                log.error(f"Failed to process candidate {idx} for SVM training: {e}")
                
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0:
            log.error("No valid features extracted. SVM training aborted.")
            return {}
            
        positives = int(y.sum())
        negatives = len(y) - positives
        log.info(f"SVM Dataset: {positives} Positives, {negatives} Negatives")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use RBF kernel for better non-linear separation
        svm = SVC(kernel='rbf', probability=True, C=1.0, class_weight='balanced', gamma='scale')
        svm.fit(X_scaled, y)
        
        # Log feature importance (for RBF, we use permutation importance)
        try:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(svm, X_scaled, y, n_repeats=10, random_state=42)
            feature_names = ["dino_score", "votes", "shadow_ratio", "edge_ratio", 
                           "centroid_dist", "aspect_ratio", "alignment_error",
                           "max_shadow_length", "depth_width_ratio", "pit_area",
                           "shadow_edge_ratio", "circularity"]
            log.info("SVM Feature Importance (Permutation):")
            for name, imp in zip(feature_names, result.importances_mean):
                log.info(f"  {name:25s}: {imp:.4f}")
        except ImportError:
            log.info("Sklearn permutation_importance not available, skipping feature importance")
            
        from luna.config import SCRATCH_DIR
        save_path = SCRATCH_DIR / "stage2_svm.pkl"
        with open(save_path, "wb") as f:
            pickle.dump({"scaler": scaler, "model": svm}, f)
            
        log.info(f"SVM classifier saved successfully to {save_path}")
        return {"positives": positives, "negatives": negatives}


def build_stage2_decoder(config: Optional[Stage2Config] = None, checkpoint_path: Optional[str | Path] = None, device: Optional[str] = None) -> Stage2DenseDecoder:
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        return Stage2DenseDecoder.from_checkpoint(checkpoint_path, device=device)
    cfg = config or DEFAULT_CONFIG
    return Stage2DenseDecoder(config=cfg, device=device)


def build_stage2_refiner(checkpoint_path: Optional[str | Path] = None, dino_encoder: Optional[any] = None, device: Optional[str] = None) -> Stage2Refiner:
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        return Stage2Refiner.from_checkpoint(checkpoint_path, dino_encoder, device=device)
    decoder = build_stage2_decoder(device=device)
    return Stage2Refiner(decoder=decoder, dino_encoder=dino_encoder)