# Vision Models & Feature Encoders

`luna.models` houses the machine learning architecture for detecting lunar collapse pits, skylights, and morphological structures in high-resolution orbital imagery.

---

## 1. Multi-Stage Pipeline Architecture

```mermaid
flowchart LR
    Raw[Raw NAC Sub-Tile 256x256] --> DINO[DINOv3 ViT Backbone]
    DINO --> Embedding[1024-d Latent Feature Vector]
    Embedding --> Binarize[Sign Binarization / Hamming Hash]
    Binarize --> PithosIndex[Pithos Fast Search]
    PithosIndex --> Candidate[Candidate Region]
    Candidate --> Stage2[Stage-2 Mask R-CNN & Refiner]
    Stage2 --> Output[Segmented Lunar Pit Polygon & Confidence]
```

---

## 2. DINOv3 Feature Backbone

`luna` employs self-supervised Vision Transformers (DINOv3) pre-trained and fine-tuned on lunar terrain:

- **Input**: Standardized $256 \times 256$ float32 reflectance patches.
- **Output**: 1024-dimensional dense feature representation.
- **Invariance**: Robust to illumination variation (phase angles from $15^\circ$ to $80^\circ$) and solar incidence.

```python
from luna.models.dinov3_encoder import DINOv3Encoder

encoder = DINOv3Encoder.from_pretrained("luna-dinov3-lunar-vit")
features = encoder.extract_features(tile_tensor)
print(f"Feature vector shape: {features.shape}")
```

---

## 3. Stage-2 Dense Decoders & Refiners

Candidate bounding boxes flagged by the screening engine are forwarded to a Stage-2 dense decoder for pixel-precise segmentation:

- **Instance Segmentation**: Mask R-CNN with Feature Pyramid Networks (FPN).
- **Geometric Ellipse Fitting**: Derives semimajor and semiminor axes, orientation angle, and rim sharpness.
- **Illumination Consistency Check**: Rejects false-positive crater shadows by validating shadow vector alignment against sub-solar azimuth.
