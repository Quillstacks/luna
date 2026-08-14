# `luna.models` Reference

`luna.models` provides Vision Transformer backbones, dense prediction segmentation heads, and physics-based validation models for lunar pit detection.

---

## DINOv3 Feature Encoder (`luna.models.dinov3.dino_encoder`)

::: luna.models.dinov3.dino_encoder
    options:
      members:
        - DINOEncoder

---

## Stage-2 Dense Decoder (`luna.models.stage2_decoder`)

::: luna.models.stage2_decoder
    options:
      members:
        - Stage2DenseDecoder
        - Stage2Config
        - SpatialTokenExtractor
        - GeometricReconstructor
        - FeaturePyramidNetwork
        - ThreeClassSegmentationHead
        - PhysicsValidator
        - PhysicsValidationResult
        - build_stage2_decoder
        - build_stage2_refiner

---

## ESSA Mask R-CNN & Refined Detections (`luna.models.essa`)

::: luna.models.essa
    options:
      members:
        - RefinedHit
        - ESSARefiner

